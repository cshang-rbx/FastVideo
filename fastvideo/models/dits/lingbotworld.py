# SPDX-License-Identifier: Apache-2.0
"""
LingbotWorld Transformer model for FastVideo.

This is the FastVideo re-implementation of the lingbot-world camera-conditioned
Wan model.  It extends the base WanTransformer3DModel with:

  * **Model-level** camera conditioning:
    - ``cam_plucker_proj``: projects 6-channel Plücker ray embeddings into hidden dim
    - ``cam_hidden_layer1/2``: two-layer MLP (with SiLU) for camera hidden states

  * **Block-level** camera injection (per transformer block):
    - ``cam_injector_layer1/2``: SiLU-gated MLP that refines camera embeddings
    - ``cam_scale_layer / cam_shift_layer``: produce scale & shift that modulate
      hidden states right after the self-attention residual

The checkpoint uses the **original Wan naming convention** (not HuggingFace
Diffusers naming), so the config carries a dedicated ``param_names_mapping``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastvideo.envs as envs
from fastvideo.configs.models.dits.lingbotworld import LingbotWorldArchConfig, LingbotWorldConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.logger import init_logger
from fastvideo.models.dits.wanvideo import (
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
    WanTransformer3DModel,
)
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)


# =========================================================================
# Block with camera injection
# =========================================================================


class LingbotWorldTransformerBlock(WanTransformerBlock):
    """Wan transformer block extended with camera injection layers.

    After the self-attention residual, if ``cam_hidden_states`` is provided the
    block applies::

        h = silu(cam_injector_layer1(cam)) → cam_injector_layer2(h)
        h = h + cam   (residual)
        scale, shift = cam_scale_layer(h), cam_shift_layer(h)
        x = (1 + scale) * x + shift
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.hidden_dim

        # Camera injection layers
        self.cam_injector_layer1 = ReplicatedLinear(dim, dim, bias=True)
        self.cam_injector_layer2 = ReplicatedLinear(dim, dim, bias=True)
        self.cam_scale_layer = ReplicatedLinear(dim, dim, bias=True)
        self.cam_shift_layer = ReplicatedLinear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cam_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype

        # Modulation
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(query, key, value, freqs_cis=freqs_cis,
                                    attention_mask=attention_mask)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        # ---- Camera injection (the key difference from vanilla Wan) ----
        if cam_hidden_states is not None:
            cam_h, _ = self.cam_injector_layer1(cam_hidden_states)
            cam_h = F.silu(cam_h)
            cam_h, _ = self.cam_injector_layer2(cam_h)
            cam_h = cam_h + cam_hidden_states  # residual
            cam_scale, _ = self.cam_scale_layer(cam_h)
            cam_shift, _ = self.cam_shift_layer(cam_h)
            hidden_states = (1.0 + cam_scale) * hidden_states + cam_shift

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


# =========================================================================
# Full model
# =========================================================================


class LingbotWorldTransformer3DModel(WanTransformer3DModel):
    """LingbotWorld camera-conditioned Wan transformer.

    Differences from ``WanTransformer3DModel``:

    1. Uses ``LingbotWorldTransformerBlock`` (with per-block camera injection).
    2. Adds model-level camera conditioning layers that project Plücker ray
       embeddings into the hidden dimension.
    3. ``forward`` accepts an extra ``cam_plucker_emb`` tensor and propagates
       the processed camera hidden states to every block.
    """

    _fsdp_shard_conditions = LingbotWorldArchConfig()._fsdp_shard_conditions
    _compile_conditions = LingbotWorldArchConfig()._compile_conditions
    _supported_attention_backends = LingbotWorldArchConfig()._supported_attention_backends
    param_names_mapping = LingbotWorldArchConfig().param_names_mapping
    reverse_param_names_mapping = LingbotWorldArchConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingbotWorldArchConfig().lora_param_names_mapping

    def __init__(self, config: LingbotWorldConfig, hf_config: dict[str, Any]) -> None:
        # Skip WanTransformer3DModel.__init__ — we re-create blocks ourselves
        # but we still want CachableDiT.__init__
        from fastvideo.models.dits.base import CachableDiT
        CachableDiT.__init__(self, config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        from fastvideo.distributed.parallel_state import get_sp_world_size
        assert config.num_attention_heads % get_sp_world_size() == 0

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings (time + text; no image embedder for lingbot-world)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Camera conditioning layers (model-level)
        cam_in_dim = (config.cam_plucker_channels
                      * config.cam_plucker_dim
                      * math.prod(config.patch_size))
        self.cam_plucker_proj = ReplicatedLinear(cam_in_dim, inner_dim, bias=True)
        self.cam_hidden_layer1 = ReplicatedLinear(inner_dim, inner_dim, bias=True)
        self.cam_hidden_layer2 = ReplicatedLinear(inner_dim, inner_dim, bias=True)

        # 4. Transformer blocks (with camera injection)
        attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND
        block_cls = LingbotWorldTransformerBlock
        self.blocks = nn.ModuleList([
            block_cls(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                self._supported_attention_backends,
                prefix=f"{config.prefix}.blocks.{i}",
            )
            for i in range(config.num_layers)
        ])

        # 5. Output norm & projection
        from fastvideo.layers.layernorm import LayerNormScaleShift
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self._logged_attention_mask = False

        # TeaCache state
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.is_even = True
        self.should_calc_even = True
        self.should_calc_odd = True
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.cnt = 0
        self.__post_init__()

    # ------------------------------------------------------------------
    def _process_camera_embedding(
        self,
        cam_plucker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Project raw Plücker rays and run through the camera MLP.

        Args:
            cam_plucker_emb: ``[B, seq_len, cam_in_dim]`` already patchified.

        Returns:
            Camera hidden states ``[B, seq_len, inner_dim]``.
        """
        emb, _ = self.cam_plucker_proj(cam_plucker_emb)
        h, _ = self.cam_hidden_layer1(emb)
        h = F.silu(h)
        h, _ = self.cam_hidden_layer2(h)
        return emb + h  # residual

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        cam_plucker_emb: torch.Tensor | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with optional camera conditioning.

        Extra arg compared to ``WanTransformer3DModel.forward``:

        * ``cam_plucker_emb`` – Plücker-ray embeddings, shape
          ``[B, 6, F, H, W]`` (6 channels, at the full latent resolution
          *before* patching).  If ``None`` the model behaves identically to
          the base Wan model.
        """
        from fastvideo.forward_context import get_forward_context
        from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
        from fastvideo.distributed.communication_op import (
            sequence_model_parallel_shard,
            sequence_model_parallel_all_gather_with_unpad,
        )
        from fastvideo.distributed.parallel_state import get_sp_world_size
        from fastvideo.distributed.utils import create_attention_mask_for_padding
        from fastvideo.platforms import current_platform

        forward_batch = get_forward_context().forward_batch
        enable_teacache = (forward_batch is not None
                           and forward_batch.enable_teacache)

        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(
                encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list) and len(
                encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # RoPE
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=(torch.float32
                   if current_platform.is_mps() else torch.float64),
            rope_theta=10000,
        )
        freqs_cis = (freqs_cos.to(hidden_states.device).float(),
                     freqs_sin.to(hidden_states.device).float())

        # Patch embed
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Process camera embedding (patchify + MLP) if provided
        cam_hidden_states = None
        if cam_plucker_emb is not None:
            # cam_plucker_emb: [B, 6, F, H, W] or already patchified
            if cam_plucker_emb.dim() == 5:
                # Patchify: rearrange to [B, (F'*H'*W'), (6*cam_dim*p_t*p_h*p_w)]
                B, C_cam, F_cam, H_cam, W_cam = cam_plucker_emb.shape
                cam_dim = C_cam  # 6
                # Reshape into patches
                cam = cam_plucker_emb.reshape(
                    B, cam_dim,
                    post_patch_num_frames, p_t,
                    post_patch_height, p_h,
                    post_patch_width, p_w,
                )
                # → [B, F', H', W', cam_dim * p_t * p_h * p_w]
                cam = cam.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
                    B,
                    post_patch_num_frames * post_patch_height * post_patch_width,
                    cam_dim * p_t * p_h * p_w,
                )
            else:
                cam = cam_plucker_emb

            # cam: [B, seq_len, cam_in_dim]  – but cam_in_dim is 6*p_t*p_h*p_w
            # need to match cam_plucker_proj input: 6 * 64 * prod(patch_size)
            # If the plucker embedding has spatial dim 64, reshape accordingly
            cam_hidden_states = self._process_camera_embedding(cam)

        # Sequence parallel shard
        hidden_states, original_seq_len = sequence_model_parallel_shard(
            hidden_states, dim=1)

        if cam_hidden_states is not None:
            cam_hidden_states, _ = sequence_model_parallel_shard(
                cam_hidden_states, dim=1)

        # Attention mask for padding
        current_seq_len = hidden_states.shape[1]
        sp_world_size = get_sp_world_size()
        padded_seq_len = current_seq_len * sp_world_size

        if padded_seq_len > original_seq_len:
            if not self._logged_attention_mask:
                logger.info("Padding applied, original seq len: %d, padded: %d",
                            original_seq_len, padded_seq_len)
                self._logged_attention_mask = True
            attention_mask = create_attention_mask_for_padding(
                seq_len=original_seq_len,
                padded_seq_len=padded_seq_len,
                batch_size=batch_size,
                device=hidden_states.device,
            )
        else:
            if not self._logged_attention_mask:
                logger.info("Padding not applied")
                self._logged_attention_mask = True
            attention_mask = None

        # Timestep handling
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len))

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if current_platform.is_mps() or current_platform.is_npu():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)

        assert encoder_hidden_states.dtype == orig_dtype

        # Transformer blocks
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb)

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            if enable_teacache:
                original_hidden_states = hidden_states.clone()

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for block in self.blocks:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states,
                        timestep_proj, freqs_cis, attention_mask,
                        cam_hidden_states)
            else:
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states,
                        timestep_proj, freqs_cis, attention_mask,
                        cam_hidden_states=cam_hidden_states)

            if enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)

        # Output
        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0)
                            + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table
                            + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(
            hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height,
            post_patch_width, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


# Entry point for model registry
EntryClass = LingbotWorldTransformer3DModel
