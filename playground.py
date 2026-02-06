# SPDX-License-Identifier: Apache-2.0
"""
Playground script to create and load Wan2.2-I2V-A14B model using FastVideo.

This script demonstrates how to:
1. Load the full MoE I2V pipeline with both transformers
2. Print model architecture and parameter information
3. Generate videos using the loaded pipeline

Usage:
    python playground.py                      # Load pipeline and print model info
    python playground.py --generate           # Load pipeline and generate a video
    python playground.py --generate --prompt "your prompt" --image path/to/image.jpg
"""

import argparse
import glob
import os
import time

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

# Set distributed environment variables (required even for single GPU)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29504"

from fastvideo.configs.pipelines.lingbotworld import LingbotWorldT2VBaseConfig
from fastvideo.fastvideo_args import FastVideoArgs, ExecutionMode
from fastvideo.logger import init_logger
from fastvideo.pipelines import build_pipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

# ============================================================================
# Configuration for Wan2.2-I2V-A14B (Image-to-Video MoE model)
# ============================================================================

# HuggingFace model ID (Diffusers format)
MODEL_ID = "/home/builder/dev/data/lingbot-world-base-cam"

# Local directory to cache the model
# set default path to ~/dev/Wan-AI/Wan2.2-I2V-A14B-Diffusers
# LOCAL_DIR = os.path.join(os.path.expanduser("~"), "dev", MODEL_ID.replace("/", "-"))
LOCAL_DIR = "/home/builder/dev/data/lingbot-world-base-cam"

# Output directory for generated videos
OUTPUT_DIR = "video_samples_wan2_2_14B_i2v"

# Example prompts and images (from Wan2.2 official examples)
EXAMPLE_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing atmosphere of the seaside."
)
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
EXAMPLE_IMAGE = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"


def print_model_info(
    model,
    model_name: str = "Model",
    checkpoint_params: int | None = None,
):
    """
    Print detailed model architecture and parameter information.
    """
    logger.info("=" * 70)
    logger.info("%s Architecture Information", model_name)
    logger.info("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Total Parameters (runtime): %s (%.2fB)", f"{total_params:,}", total_params / 1e9)
    if checkpoint_params is not None:
        logger.info(
            "Total Parameters (checkpoint): %s (%.2fB)",
            f"{checkpoint_params:,}",
            checkpoint_params / 1e9,
        )
    logger.info("Trainable Parameters: %s (%.2fB)", f"{trainable_params:,}", trainable_params / 1e9)
    logger.info("Model dtype: %s", next(model.parameters()).dtype)
    
    # Print model architecture summary
    logger.info("\nModel Architecture:")
    logger.info("-" * 70)
    
    # Get top-level modules
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        logger.info("  %s: %s (%s params)", name, module.__class__.__name__, f"{num_params:,}")
    
    # Print detailed layer counts
    logger.info("\nLayer Statistics:")
    logger.info("-" * 70)
    
    layer_types = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in layer_types:
            layer_types[module_type] = 0
        layer_types[module_type] += 1
    
    # Sort by count and print top layer types
    sorted_layers = sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:15]
    for layer_type, count in sorted_layers:
        logger.info("  %s: %d", layer_type, count)
    
    logger.info("=" * 70)


def count_checkpoint_params(model_dir: str) -> int | None:
    """
    Count parameters from safetensors checkpoints without loading tensors into memory.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors not available; skipping checkpoint param count.")
        return None

    tensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not tensor_files:
        logger.warning("No .safetensors files found in %s", model_dir)
        return None

    total_params = 0
    for file_path in tensor_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                numel = 1
                for dim in shape:
                    numel *= dim
                total_params += numel
    return total_params


def load_pipeline():
    """
    Load the full MoE I2V pipeline with both transformers.
    
    The A14B I2V model is a Mixture of Experts (MoE) model with:
    - transformer: handles low-noise timesteps (t < boundary)
    - transformer_2: handles high-noise timesteps (t >= boundary)
    - text_encoder: UMT5-XXL for text encoding
    - vae: AutoencoderKLWan for encoding/decoding
    
    Returns:
        tuple: (pipeline, fastvideo_args)
    """
    logger.info("=" * 70)
    logger.info("Loading Wan2.2-I2V-A14B MoE Pipeline")
    logger.info("=" * 70)
    
    # Download model
    logger.info("Downloading/loading model from: %s", MODEL_ID)
    model_path = maybe_download_model(MODEL_ID, local_dir=LOCAL_DIR)
    logger.info("Model path: %s", model_path)
    
    # Create the I2V pipeline config
    pipeline_config = LingbotWorldT2VBaseConfig()
    logger.info("Pipeline config: %s", pipeline_config.__class__.__name__)
    logger.info("  - boundary_ratio: %s", pipeline_config.boundary_ratio)
    logger.info("  - flow_shift: %s", pipeline_config.flow_shift)
    print("pipeline_config: %s", pipeline_config)
    
    # Create FastVideo args
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        hsdp_replicate_dim=1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pipeline_config=pipeline_config,
    )
    
    # Build pipeline - this loads all modules
    logger.info("\nBuilding pipeline...")
    pipeline = build_pipeline(fastvideo_args)
    
    # Count parameters directly from checkpoints (full model size)
    transformer_ckpt_params = count_checkpoint_params(
        os.path.join(model_path, "transformer")
    )
    transformer_2_ckpt_params = count_checkpoint_params(
        os.path.join(model_path, "transformer_2")
    )

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Loaded Successfully!")
    logger.info("=" * 70)
    logger.info("Loaded modules: %s", list(pipeline.modules.keys()))
    
    # Print model info for each transformer
    if "transformer" in pipeline.modules:
        print_model_info(
            pipeline.modules["transformer"],
            "Transformer (Low-Noise Expert)",
            checkpoint_params=transformer_ckpt_params,
        )
    
    if "transformer_2" in pipeline.modules:
        print_model_info(
            pipeline.modules["transformer_2"],
            "Transformer_2 (High-Noise Expert)",
            checkpoint_params=transformer_2_ckpt_params,
        )
    
    # Print total parameters across all modules
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Module Summary")
    logger.info("=" * 70)
    
    total_params = 0
    for name, module in pipeline.modules.items():
        if hasattr(module, 'parameters'):
            try:
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                logger.info("  %s: %s parameters (%.2fB)", name, f"{params:,}", params / 1e9)
            except Exception:
                logger.info("  %s: (non-parametric module)", name)
    
    logger.info("-" * 70)
    logger.info(
        "Total Pipeline Parameters (runtime): %s (%.2fB)",
        f"{total_params:,}",
        total_params / 1e9,
    )

    if transformer_ckpt_params is not None or transformer_2_ckpt_params is not None:
        ckpt_total = sum(
            p for p in [transformer_ckpt_params, transformer_2_ckpt_params] if p is not None
        )
        logger.info(
            "Total Transformers Parameters (checkpoint): %s (%.2fB)",
            f"{ckpt_total:,}",
            ckpt_total / 1e9,
        )
    logger.info("=" * 70)
    
    return pipeline, fastvideo_args


def generate_video(
    pipeline,
    fastvideo_args: FastVideoArgs,
    prompt: str = None,
    image_path: str = None,
    output_path: str = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    """
    Generate a video using the loaded pipeline.
    
    This function creates a ForwardBatch and runs sampling through the pipeline,
    similar to how the official Wan2.2 generate.py works.
    
    Args:
        pipeline: The loaded FastVideo pipeline
        fastvideo_args: FastVideo arguments used to build the pipeline
        prompt: Text description for video generation
        image_path: Path or URL to the input image for I2V
        output_path: Path to save the generated video
        height: Output video height (default: 480)
        width: Output video width (default: 832)
        num_frames: Number of frames to generate (default: 81, should be 4n+1)
        num_inference_steps: Denoising steps (default: 40)
        guidance_scale: Classifier-free guidance scale (default: 5.0)
        seed: Random seed for reproducibility
    
    Returns:
        Generated video frames as numpy arrays
    """
    # Use defaults if not provided
    if prompt is None:
        prompt = EXAMPLE_PROMPT
    if image_path is None:
        image_path = EXAMPLE_IMAGE
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"generated_seed{seed}.mp4")
    
    logger.info("=" * 70)
    logger.info("Wan2.2-I2V-A14B Video Generation")
    logger.info("=" * 70)
    logger.info("Prompt: %s...", prompt[:100])
    logger.info("Image: %s", image_path)
    logger.info("Output: %s", output_path)
    logger.info("Resolution: %dx%d", width, height)
    logger.info("Frames: %d", num_frames)
    logger.info("Steps: %d", num_inference_steps)
    logger.info("Guidance Scale: %.1f", guidance_scale)
    logger.info("Seed: %d", seed)
    logger.info("-" * 70)
    
    # Calculate latent dimensions
    height_latents = height // 8
    width_latents = width // 8
    num_latent_frames = (num_frames - 1) // 4 + 1
    n_tokens = num_latent_frames * height_latents * width_latents
    
    logger.info("\nLatent dimensions:")
    logger.info("  - height_latents: %d", height_latents)
    logger.info("  - width_latents: %d", width_latents)
    logger.info("  - num_latent_frames: %d", num_latent_frames)
    logger.info("  - n_tokens: %d", n_tokens)
    
    # Create ForwardBatch for the pipeline
    batch = ForwardBatch(
        data_type="video",
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        n_tokens=n_tokens,
        output_path=output_path,
        save_video=True,
        fps=16,
    )
    
    # Run inference through the pipeline
    logger.info("\nRunning inference...")
    start_time = time.perf_counter()

    print("+" * 70)
    print("pipeline.modules.keys()")
    print(pipeline.modules.keys())
    if "transformer" in pipeline.modules:
        print(f"{type(pipeline.modules['transformer'])}")
    if "transformer_2" in pipeline.modules:
        print(f"type(pipeline.modules['transformer_2']): {type(pipeline.modules['transformer_2'])}")
    print("+" * 70)
    
    output_batch = pipeline.forward(batch, fastvideo_args)
    
    gen_time = time.perf_counter() - start_time
    logger.info("Generation completed in %.2f seconds", gen_time)
    
    # Process outputs
    samples = output_batch.output
    if samples is not None:
        logger.info("Output tensor shape: %s", samples.shape)
        logger.info("Output tensor dtype: %s", samples.dtype)
        
        # Convert to frames
        videos = rearrange(samples, "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))
        
        # Save video
        if batch.save_video:
            imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
            logger.info("Saved video to: %s", output_path)
        
        logger.info("\n" + "=" * 70)
        logger.info("Video Generation Complete!")
        logger.info("=" * 70)
        
        return frames
    else:
        logger.warning("No output generated")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wan2.2-I2V-A14B Playground - Load MoE pipeline and generate videos"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a video after loading the pipeline"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to input image for I2V"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for generated video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output video height (default: 480)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Output video width (default: 832)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=41,
        help="Number of frames to generate (default: 81, should be 4n+1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 40)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load the pipeline
    pipeline, fastvideo_args = load_pipeline()
    
    # Generate video if requested
    if args.generate:
        generate_video(
            pipeline=pipeline,
            fastvideo_args=fastvideo_args,
            prompt=args.prompt,
            image_path=args.image,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
