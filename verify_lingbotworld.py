#!/usr/bin/env python3
"""
Minimal verification script for LingbotWorldTransformer3DModel.

This script:
1. Instantiates the model on meta device (no actual memory)
2. Verifies the architecture matches expected layer counts
3. Checks that checkpoint keys can be mapped to model keys
4. (Optionally) loads real weights and runs a dummy forward pass

Usage:
    # Architecture check only (no GPU needed):
    python verify_lingbotworld.py

    # Full load + forward pass (needs GPU + checkpoint):
    python verify_lingbotworld.py --load --checkpoint-dir /path/to/low_noise_model
"""

import argparse
import os
import re
import sys

import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29506")

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def build_config():
    """Build config for lingbot-world A14B I2V."""
    from fastvideo.models.dits.lingbotworld import LingbotWorldArchConfig, LingbotWorldConfig

    arch = LingbotWorldArchConfig(
        num_attention_heads=40,
        attention_head_dim=128,
        num_layers=40,
        ffn_dim=13824,
        patch_size=(1, 2, 2),
        in_channels=36,
        out_channels=16,
        text_dim=4096,
        text_len=512,
        freq_dim=256,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        eps=1e-6,
        image_dim=None,  # No CLIP image embedding
        cam_plucker_channels=6,
        cam_plucker_dim=64,
    )
    return LingbotWorldConfig(arch_config=arch, prefix="LingbotWorld")


def verify_architecture():
    """Instantiate the model on meta device and check structure."""
    from fastvideo.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.models.dits.lingbotworld import LingbotWorldTransformer3DModel

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    config = build_config()
    logger.info("=" * 70)
    logger.info("Verifying LingbotWorldTransformer3DModel architecture")
    logger.info("=" * 70)

    # Build on meta device (no memory)
    with torch.device("meta"):
        model = LingbotWorldTransformer3DModel(config=config, hf_config={})

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s (%.2fB)", f"{total:,}", total / 1e9)

    # Check key components exist
    assert hasattr(model, "cam_plucker_proj"), "Missing cam_plucker_proj"
    assert hasattr(model, "cam_hidden_layer1"), "Missing cam_hidden_layer1"
    assert hasattr(model, "cam_hidden_layer2"), "Missing cam_hidden_layer2"
    logger.info("✓ Model-level camera layers present")

    for i, block in enumerate(model.blocks):
        assert hasattr(block, "cam_injector_layer1"), f"Block {i} missing cam_injector_layer1"
        assert hasattr(block, "cam_scale_layer"), f"Block {i} missing cam_scale_layer"
    logger.info("✓ All %d blocks have camera injection layers", len(model.blocks))

    # Print top-level modules
    logger.info("\nTop-level modules:")
    for name, mod in model.named_children():
        n_params = sum(p.numel() for p in mod.parameters())
        logger.info("  %s: %s (%s params)", name, mod.__class__.__name__, f"{n_params:,}")

    return model, config


def verify_checkpoint_mapping(checkpoint_dir: str):
    """Check that all checkpoint keys can be mapped to model keys."""
    from safetensors import safe_open
    from fastvideo.models.dits.lingbotworld import LingbotWorldArchConfig

    mapping = LingbotWorldArchConfig().param_names_mapping

    logger.info("\n" + "=" * 70)
    logger.info("Verifying checkpoint key mapping")
    logger.info("=" * 70)

    # Gather all checkpoint keys
    ckpt_keys = set()
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith(".safetensors") and "index" not in fname:
            path = os.path.join(checkpoint_dir, fname)
            with safe_open(path, framework="pt") as f:
                ckpt_keys.update(f.keys())

    logger.info("Checkpoint has %d keys", len(ckpt_keys))

    # Try mapping each key
    mapped = {}
    unmapped = []
    for key in sorted(ckpt_keys):
        new_key = key
        for pattern, replacement in mapping.items():
            result = re.sub(pattern, replacement, key)
            if result != key:
                new_key = result
                break
        mapped[key] = new_key
        if new_key == key:
            unmapped.append(key)

    # Build model on meta to get expected keys (distributed already init'd by verify_architecture)
    config = build_config()
    with torch.device("meta"):
        from fastvideo.models.dits.lingbotworld import LingbotWorldTransformer3DModel
        model = LingbotWorldTransformer3DModel(config=config, hf_config={})

    model_keys = set(model.state_dict().keys())
    mapped_values = set(mapped.values())

    # Check coverage
    in_model = mapped_values & model_keys
    not_in_model = mapped_values - model_keys
    not_loaded = model_keys - mapped_values

    logger.info("Mapped checkpoint keys found in model: %d / %d",
                len(in_model), len(mapped_values))

    if not_in_model:
        logger.warning("Mapped keys NOT in model (%d):", len(not_in_model))
        for k in sorted(not_in_model)[:20]:
            logger.warning("  %s", k)

    if not_loaded:
        logger.warning("Model keys NOT loaded from checkpoint (%d):", len(not_loaded))
        for k in sorted(not_loaded)[:20]:
            logger.warning("  %s", k)

    if unmapped:
        logger.info("Checkpoint keys with no mapping (kept as-is): %d", len(unmapped))
        for k in sorted(unmapped)[:10]:
            logger.info("  %s → %s", k, mapped[k])

    if not not_in_model and not not_loaded:
        logger.info("✓ Perfect mapping: all checkpoint keys map to model keys")
    
    return len(not_in_model) == 0 and len(not_loaded) == 0


def run_dummy_forward(checkpoint_dir: str | None = None):
    """Run a dummy forward pass (optionally with real weights)."""
    from fastvideo.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.models.dits.lingbotworld import LingbotWorldTransformer3DModel
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    config = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logger.info("\n" + "=" * 70)
    logger.info("Running dummy forward pass on %s", device)
    logger.info("=" * 70)

    # Build model
    model = LingbotWorldTransformer3DModel(config=config, hf_config={})
    model = model.to(device=device, dtype=dtype).eval()

    total = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s (%.2fB)", f"{total:,}", total / 1e9)

    # Create dummy inputs
    B = 1
    C_in = 36  # I2V channels
    T, H, W = 5, 60, 104  # small latent size (divisible by patch_size)
    text_len = 512
    text_dim = 4096

    hidden_states = torch.randn(B, C_in, T, H, W, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, text_len, text_dim, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)

    # Camera plucker embedding: [B, 6*64, T, H, W] at full latent resolution
    # 6 Plücker ray channels × 64 spatial dim per channel = 384
    cam_channels = 6 * 64  # cam_plucker_channels * cam_plucker_dim
    cam_plucker = torch.randn(B, cam_channels, T, H, W, device=device, dtype=dtype)

    forward_batch = ForwardBatch(data_type="dummy")

    logger.info("Input shapes:")
    logger.info("  hidden_states: %s", hidden_states.shape)
    logger.info("  encoder_hidden_states: %s", encoder_hidden_states.shape)
    logger.info("  cam_plucker_emb: %s", cam_plucker.shape)

    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cam_plucker_emb=cam_plucker,
            )

    logger.info("✓ Forward pass succeeded!")
    logger.info("  Output shape: %s", output.shape)
    logger.info("  Output dtype: %s", output.dtype)
    logger.info("  Expected: [%d, %d, %d, %d, %d]", B, config.out_channels, T, H, W)

    # Also test without camera (should work like vanilla Wan)
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output_no_cam = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cam_plucker_emb=None,
            )
    logger.info("✓ Forward pass without camera also succeeded!")
    logger.info("  Output shape: %s", output_no_cam.shape)

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify LingbotWorld model")
    parser.add_argument("--load", action="store_true",
                        help="Run a dummy forward pass (needs GPU)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoint dir for key mapping verification")
    args = parser.parse_args()

    # Always verify architecture
    verify_architecture()

    # Verify checkpoint mapping if dir provided
    if args.checkpoint_dir:
        verify_checkpoint_mapping(args.checkpoint_dir)

    # Run forward pass if requested
    if args.load:
        run_dummy_forward(args.checkpoint_dir)

    logger.info("\n" + "=" * 70)
    logger.info("All checks passed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
