# SPDX-License-Identifier: Apache-2.0
"""
LingbotWorld pipeline stages for camera-conditioned video generation.

This module provides:

* :class:`CameraConditioningStage` – computes Plücker-ray camera embeddings
  from pose / intrinsic files (or accepts a pre-computed tensor) and stores
  the result in ``batch.extra["cam_plucker_emb"]``.

* :class:`LingbotWorldDenoisingStage` – extends the standard
  :class:`~fastvideo.pipelines.stages.denoising.DenoisingStage` to inject
  ``cam_plucker_emb`` into every transformer forward call.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.stages.lingbotworld_utils import (
    compute_cam_plucker_emb)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import (StageValidators as V,
                                                   VerificationResult)

logger = init_logger(__name__)


# =========================================================================
# Camera conditioning stage
# =========================================================================


class CameraConditioningStage(PipelineStage):
    """Compute Plücker-ray camera embeddings and store in ``batch.extra``.

    This stage reads camera data from ``batch.extra`` and produces
    ``batch.extra["cam_plucker_emb"]`` — a ``[1, C, F_lat, H_lat, W_lat]``
    tensor that will be forwarded to the LingbotWorld transformer.

    Expected ``batch.extra`` keys (one of the two options):

    * **Option A** – file paths:
        - ``"poses_path"`` (str): path to ``poses.npy``
        - ``"intrinsics_path"`` (str): path to ``intrinsics.npy``
        - ``"original_height"`` (int, optional, default 480)
        - ``"original_width"`` (int, optional, default 832)

    * **Option B** – pre-computed tensor:
        - ``"cam_plucker_emb"`` (torch.Tensor): already computed.
    """

    def __init__(self, vae_stride: tuple[int, int, int],
                 patch_size: tuple[int, int, int] = (1, 2, 2)) -> None:
        super().__init__()
        self.vae_stride = vae_stride
        self.patch_size = patch_size

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        # If already pre-computed, nothing to do
        if batch.extra.get("cam_plucker_emb") is not None:
            logger.info("Using pre-computed cam_plucker_emb from batch.extra")
            return batch

        poses_path = batch.extra.get("poses_path")
        intrinsics_path = batch.extra.get("intrinsics_path")

        if poses_path is None or intrinsics_path is None:
            logger.warning(
                "No camera data provided (poses_path / intrinsics_path / "
                "cam_plucker_emb). Running without camera conditioning.")
            return batch

        num_frames = batch.num_frames
        if isinstance(num_frames, list):
            num_frames = num_frames[0]
        height = batch.height
        if isinstance(height, list):
            height = height[0]
        width = batch.width
        if isinstance(width, list):
            width = width[0]

        original_height = batch.extra.get("original_height", 480)
        original_width = batch.extra.get("original_width", 832)

        device = get_local_torch_device()

        cam_plucker_emb = compute_cam_plucker_emb(
            poses_path=poses_path,
            intrinsics_path=intrinsics_path,
            num_frames=num_frames,
            height=height,
            width=width,
            vae_stride=self.vae_stride,
            patch_size=self.patch_size,
            device=device,
            dtype=torch.bfloat16,
            original_height=original_height,
            original_width=original_width,
        )

        batch.extra["cam_plucker_emb"] = cam_plucker_emb
        logger.info(
            "Computed cam_plucker_emb from files: shape=%s",
            cam_plucker_emb.shape,
        )
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.not_none)
        result.add_check("width", batch.width, V.not_none)
        result.add_check("num_frames", batch.num_frames, V.not_none)
        return result


# =========================================================================
# Denoising stage with camera conditioning
# =========================================================================


class LingbotWorldDenoisingStage(DenoisingStage):
    """Denoising stage that passes ``cam_plucker_emb`` to the transformer.

    Extends the standard :class:`DenoisingStage` to also inject the Plücker-ray
    camera embeddings stored in ``batch.extra["cam_plucker_emb"]`` into every
    transformer forward call.  The base denoising logic (CFG, boundary ratio
    for Wan2.2 dual-expert, STA, etc.) is fully preserved.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        cam_plucker_emb = batch.extra.get("cam_plucker_emb", None)

        if cam_plucker_emb is None:
            # No camera conditioning — fall back to standard Wan denoising.
            return super().forward(batch, fastvideo_args)

        # Temporarily wrap the transformer(s) so that every call receives
        # cam_plucker_emb as an extra keyword argument.  We use
        # functools.wraps so that inspect.signature still returns the
        # *original* signature (required by prepare_extra_func_kwargs in
        # the parent class).
        import functools

        models_to_patch = []
        for model in (self.transformer, self.transformer_2):
            if model is not None:
                models_to_patch.append(model)

        originals: list[Any] = []
        for model in models_to_patch:
            orig = model.forward

            @functools.wraps(orig)
            def _patched(
                *args,
                _orig_fn=orig,
                _emb=cam_plucker_emb,
                **kwargs,
            ):
                kwargs.setdefault("cam_plucker_emb", _emb)
                return _orig_fn(*args, **kwargs)

            originals.append(orig)
            model.forward = _patched  # type: ignore[method-assign]

        try:
            result = super().forward(batch, fastvideo_args)
        finally:
            # Restore original forward methods
            for model, orig in zip(models_to_patch, originals):
                model.forward = orig  # type: ignore[method-assign]

        return result
