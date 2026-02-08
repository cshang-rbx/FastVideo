"""
Ray-compatible Trainer wrapper for FastVideo training pipelines.

This module provides a ``Trainer`` class that follows the video_gen
convention (config-driven, ``use_ray`` flag, ``train()`` entry point)
while delegating all actual training logic to FastVideo's existing
training pipelines.

The pipeline class is selected via the ``pipeline`` config key, which
maps to a registry of known pipelines.

Usage from Ray Train ``train_loop_per_worker``::

    from fastvideo.training.ray.trainer import Trainer
    trainer = Trainer(cfg, use_ray=True)
    trainer.train()

Usage standalone (torchrun)::

    torchrun --nproc_per_node 8 \\
        -m fastvideo.training.ray.trainer \\
        --config configs/lingbot_finetune.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf

# NOTE: Do NOT import from fastvideo at module level.  Ray workers
# deserialise this module before CUDA is available, and
# ``fastvideo/__init__.py`` eagerly imports Triton kernels that
# require an active CUDA driver.  All fastvideo imports are deferred
# to inside methods that run after CUDA is ready.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Registry of pipeline short names → fully qualified class paths.
# The ``pipeline`` key in the YAML config selects from this map.
PIPELINE_REGISTRY: dict[str, str] = {
    "lingbotworld":
        "fastvideo.training.lingbotworld_training_pipeline.LingbotWorldTrainingPipeline",
    "wan_t2v":
        "fastvideo.training.wan_training_pipeline.WanTrainingPipeline",
    "wan_i2v":
        "fastvideo.training.wan_i2v_training_pipeline.WanI2VTrainingPipeline",
}


def _import_class(dotted_path: str) -> type:
    """Import a class from a fully qualified dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_pipeline_cls(pipeline_name: str) -> type:
    """Resolve a pipeline name to a class.

    Accepts either a short name from ``PIPELINE_REGISTRY`` or a fully
    qualified class path (e.g. ``fastvideo.training.my_pipeline.MyPipeline``).

    NOTE: This triggers fastvideo imports, so it must only be called
    after CUDA is available (i.e. inside a worker, not during Ray
    serialisation).
    """
    if pipeline_name in PIPELINE_REGISTRY:
        return _import_class(PIPELINE_REGISTRY[pipeline_name])

    # Assume it's a fully qualified path
    try:
        return _import_class(pipeline_name)
    except (ImportError, AttributeError) as exc:
        available = ", ".join(sorted(PIPELINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown pipeline '{pipeline_name}'. "
            f"Available short names: [{available}]. "
            f"Or provide a fully qualified class path."
        ) from exc


def _cfg_to_argv(cfg: DictConfig) -> list[str]:
    """Convert a flat OmegaConf DictConfig into a CLI argv list.

    The config YAML uses the same key names as FastVideo CLI args
    (underscores are converted to hyphens).  Boolean ``True`` values
    produce ``--flag True``; ``None`` values are skipped.

    Example YAML::

        model_path: /data/model
        num_gpus: 8
        log_validation: true
        inference_mode: false

    Produces::

        ["--model-path", "/data/model", "--num-gpus", "8",
         "--log-validation", "True", "--inference-mode", "False"]
    """
    argv: list[str] = []
    flat = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(flat, dict)

    # These keys are meta / launcher-only, not FastVideo CLI args.
    _SKIP_KEYS = {"pipeline", "use_ray", "ray_address", "num_workers"}

    for key, value in flat.items():
        if key in _SKIP_KEYS:
            continue
        if value is None:
            continue

        flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            # FastVideo uses StoreBoolean which accepts True/False as values
            argv.extend([flag, str(value)])
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])

    return argv


def _parse_training_args(argv: list[str]) -> argparse.Namespace:
    """Parse a CLI argv list into the argparse.Namespace that FastVideo
    ``TrainingArgs.from_cli_args`` / ``from_pretrained`` expect."""
    from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args(argv)
    args.dit_cpu_offload = False
    return args


class Trainer:
    """Ray-compatible trainer for FastVideo training pipelines.

    This is a thin adapter that bridges between the ``video_gen``-style
    config-driven ``Trainer(cfg, use_ray=True).train()`` pattern and
    FastVideo's existing training pipelines.

    The pipeline class is selected via ``cfg.pipeline``:

    - Short names: ``"lingbotworld"``, ``"wan_t2v"``, ``"wan_i2v"``
    - Fully qualified: ``"fastvideo.training.my_pipeline.MyPipeline"``

    If ``cfg.pipeline`` is not set, defaults to ``"lingbotworld"``.

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf config whose keys match FastVideo CLI arg names.
        Required keys: ``model_path``, ``pretrained_model_name_or_path``,
        ``data_path``, ``output_dir``, ``num_gpus``, ``sp_size``,
        ``learning_rate``, ``max_train_steps``, etc.
    use_ray : bool
        If True, assumes ``torch.distributed`` is already initialized
        by Ray Train.  If False, ``torchrun`` / manual init is expected.
    """

    def __init__(self, cfg: DictConfig, use_ray: bool = False) -> None:
        self.cfg = cfg
        self.use_ray = use_ray
        self.pipeline = None
        self._pipeline_cls = None  # resolved lazily in _build_pipeline

        # Store pipeline name for lazy resolution (avoid fastvideo imports
        # at construction time — CUDA may not be available yet under Ray).
        self._pipeline_name = cfg.get("pipeline", "lingbotworld")

        # Ensure inference_mode is off for training
        if "inference_mode" not in cfg:
            OmegaConf.update(cfg, "inference_mode", False)

        logger.info(
            "Trainer initialized (pipeline=%s, use_ray=%s, num_gpus=%s)",
            self._pipeline_name,
            use_ray,
            cfg.get("num_gpus", "?"),
        )

    def _build_pipeline(self):
        """Build the training pipeline from config.

        This is where fastvideo modules are first imported, so CUDA must
        be available by the time this method runs.
        """
        # Resolve pipeline class (triggers fastvideo imports)
        self._pipeline_cls = _resolve_pipeline_cls(self._pipeline_name)
        logger.info("Resolved pipeline class: %s", self._pipeline_cls.__name__)

        # Convert OmegaConf config → CLI argv → argparse.Namespace
        # This reuses FastVideo's existing argument parsing & validation.
        argv = _cfg_to_argv(self.cfg)
        logger.info("Converted config to %d CLI args", len(argv))
        logger.debug("argv: %s", argv)

        args = _parse_training_args(argv)

        # Build the pipeline (loads model, sets up FSDP, optimizer, etc.)
        self.pipeline = self._pipeline_cls.from_pretrained(
            args.pretrained_model_name_or_path, args=args)

        logger.info("%s built successfully", self._pipeline_cls.__name__)

    def train(self) -> None:
        """Run the full training loop.

        This is the entry point called by ``train_loop_per_worker``
        in Ray Train, or directly for standalone execution.
        """
        if self.pipeline is None:
            self._build_pipeline()

        assert self.pipeline is not None
        self.pipeline.train()
        logger.info("Training complete")


# ---------------------------------------------------------------------------
# Standalone entry point (for torchrun without Ray)
# ---------------------------------------------------------------------------

def _parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FastVideo Ray trainer entry point")
    parser.add_argument(
        "--config", required=True, help="Path to training config YAML.")
    parser.add_argument(
        "--num_gpus", type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="Number of GPUs (for ScalingConfig compat, unused in torchrun).")
    return parser.parse_args()


if __name__ == "__main__":
    launcher_args = _parse_launcher_args()
    cfg = OmegaConf.load(launcher_args.config)
    OmegaConf.resolve(cfg)
    trainer = Trainer(cfg, use_ray=False)
    trainer.train()
