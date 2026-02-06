# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])

@dataclass
class LingbotWorldArchConfig(WanVideoArchConfig):
    """Architecture config for the LingbotWorld camera-conditioned Wan model.

    The checkpoint uses original-Wan naming (``self_attn.q``, ``cross_attn.k``,
    ``ffn.0``, …) rather than the HuggingFace Diffusers convention.  The
    ``param_names_mapping`` below translates those names into the FastVideo
    internal names.
    """

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    # ------------------------------------------------------------------
    # Mapping: original-Wan checkpoint key  →  FastVideo internal key
    # ------------------------------------------------------------------
    param_names_mapping: dict = field(default_factory=lambda: {
        # --- patch embedding ---
        r"^patch_embedding\.(.*)$":
            r"patch_embedding.proj.\1",

        # --- camera conditioning (model-level) ---
        r"^patch_embedding_wancamctrl\.(.*)$":
            r"cam_plucker_proj.\1",
        r"^c2ws_hidden_states_layer1\.(.*)$":
            r"cam_hidden_layer1.\1",
        r"^c2ws_hidden_states_layer2\.(.*)$":
            r"cam_hidden_layer2.\1",

        # --- time embedding / projection ---
        r"^time_embedding\.0\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_in.\1",
        r"^time_embedding\.2\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_out.\1",
        r"^time_projection\.1\.(.*)$":
            r"condition_embedder.time_modulation.linear.\1",

        # --- text embedding ---
        r"^text_embedding\.0\.(.*)$":
            r"condition_embedder.text_embedder.fc_in.\1",
        r"^text_embedding\.2\.(.*)$":
            r"condition_embedder.text_embedder.fc_out.\1",

        # --- head ---
        r"^head\.modulation$":
            r"scale_shift_table",
        r"^head\.norm\.(.*)$":
            r"norm_out.norm.\1",
        r"^head\.head\.(.*)$":
            r"proj_out.\1",

        # --- blocks: self-attention ---
        r"^blocks\.(\d+)\.self_attn\.q\.(.*)$":
            r"blocks.\1.to_q.\2",
        r"^blocks\.(\d+)\.self_attn\.k\.(.*)$":
            r"blocks.\1.to_k.\2",
        r"^blocks\.(\d+)\.self_attn\.v\.(.*)$":
            r"blocks.\1.to_v.\2",
        r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.to_out.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
            r"blocks.\1.norm_q.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
            r"blocks.\1.norm_k.\2",

        # --- blocks: cross-attention ---
        r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$":
            r"blocks.\1.attn2.to_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$":
            r"blocks.\1.attn2.to_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$":
            r"blocks.\1.attn2.to_v.\2",
        r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn2.norm_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn2.norm_k.\2",

        # --- blocks: norms ---
        # norm3 (cross-attn-norm, has params) → self_attn_residual_norm.norm
        r"^blocks\.(\d+)\.norm3\.(.*)$":
            r"blocks.\1.self_attn_residual_norm.norm.\2",
        # norm2 (ffn pre-norm, no params for affine=False) → cross_attn_residual_norm.norm
        # included for completeness; if affine variants appear the mapping works
        r"^blocks\.(\d+)\.norm2\.(.*)$":
            r"blocks.\1.cross_attn_residual_norm.norm.\2",

        # --- blocks: FFN ---
        r"^blocks\.(\d+)\.ffn\.0\.(.*)$":
            r"blocks.\1.ffn.fc_in.\2",
        r"^blocks\.(\d+)\.ffn\.2\.(.*)$":
            r"blocks.\1.ffn.fc_out.\2",

        # --- blocks: modulation ---
        r"^blocks\.(\d+)\.modulation$":
            r"blocks.\1.scale_shift_table",

        # --- blocks: camera injection (names kept identical) ---
        # cam_injector_layer1, cam_injector_layer2, cam_scale_layer, cam_shift_layer
        # are already named the same in the checkpoint and model → no mapping needed
    })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    # Model architecture (A14B I2V defaults)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 36      # I2V: 16 (video) + 16 (image latent) + 4 (mask)
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None  # No CLIP image embedding in lingbot-world
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024

    # Camera conditioning
    cam_plucker_channels: int = 6   # Plücker ray: 6 channels
    cam_plucker_dim: int = 64       # Spatial dim per channel

    # MoE
    boundary_ratio: float | None = None


@dataclass
class LingbotWorldConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingbotWorldArchConfig)
    prefix: str = "LingbotWorld"
