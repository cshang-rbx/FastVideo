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

# Set distributed environment variables (defaults for single GPU;
# torchrun sets these automatically for multi-GPU)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29504")

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
# s5cmd cp s3://3dfm-videogen/models/fastvideo-lingbot-world-base-cam/\* fastvideo-lingbot-world-base-cam/
MODEL_ID = "/home/builder/workspace/weights/fastvideo-lingbot-world-base-cam"

# Local directory to cache the model
LOCAL_DIR = "/home/builder/workspace/weights/fastvideo-lingbot-world-base-cam"

# Output directory for generated videos
OUTPUT_DIR = "video_samples_fastvideo-lingbot-world-base-cam"

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


def load_pipeline(
    num_gpus: int = 1,
    tp_size: int = 1,
    sp_size: int = 1,
    hsdp_shard_dim: int = 1,
    hsdp_replicate_dim: int = 1,
):
    """
    Load the full MoE I2V pipeline with both transformers.
    
    The A14B I2V model is a Mixture of Experts (MoE) model with:
    - transformer: handles low-noise timesteps (t < boundary)
    - transformer_2: handles high-noise timesteps (t >= boundary)
    - text_encoder: UMT5-XXL for text encoding
    - vae: AutoencoderKLWan for encoding/decoding
    
    Args:
        num_gpus: Total number of GPUs to use (default: 1)
        tp_size: Tensor parallelism size (default: 1)
        sp_size: Sequence parallelism size (default: 1, set >1 for SP)
        hsdp_shard_dim: FSDP shard dimension for HSDP (default: 1)
        hsdp_replicate_dim: HSDP replication dimension (default: 1)

    Returns:
        tuple: (pipeline, fastvideo_args)
    """
    logger.info("=" * 70)
    logger.info("Loading LingbotWorld T2V Base Pipeline")
    logger.info("=" * 70)
    logger.info("Parallelism: num_gpus=%d, tp=%d, sp=%d, hsdp_shard=%d, hsdp_rep=%d",
                num_gpus, tp_size, sp_size, hsdp_shard_dim, hsdp_replicate_dim)
    
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
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        hsdp_shard_dim=hsdp_shard_dim,
        hsdp_replicate_dim=hsdp_replicate_dim,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pipeline_config=pipeline_config,
        # The model_index.json has _class_name="WanImageToVideoPipeline",
        # but we want the LingbotWorld pipeline which adds camera conditioning.
        override_pipeline_cls_name="LingbotWorldI2VPipeline",
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


def draw_camera_joystick_on_frames(
    frames: list[np.ndarray],
    poses_path: str,
    num_video_frames: int,
) -> list[np.ndarray]:
    """Draw a joystick-style camera motion indicator on each video frame.

    Overlays two widgets:

    * **Bottom-right – Joystick**: shows per-frame camera translation
      direction in camera-local coordinates (right/left ↔ x, forward/back ↔ y
      on screen).  A bright dot + line from centre indicates direction and
      magnitude.

    * **Bottom-left – Trajectory minimap**: top-down (world XZ) view of the
      full camera path.  The traversed portion is highlighted and a heading
      arrow shows which way the camera is facing.

    Args:
        frames: List of ``(H, W, 3)`` uint8 numpy frames.
        poses_path: Path to ``poses.npy`` — ``(N, 4, 4)`` c2w matrices
            (OpenCV convention).
        num_video_frames: Number of video frames (poses are truncated /
            repeated to match).

    Returns:
        New list of frames with the overlay drawn.
    """
    from PIL import Image, ImageDraw, ImageFont

    # ------------------------------------------------------------------
    # Load & align poses to video frames
    # ------------------------------------------------------------------
    c2ws = np.load(poses_path).astype(np.float64)  # (N, 4, 4)
    n_poses = len(c2ws)

    # Build a pose index for every video frame (clamp if poses < frames)
    pose_indices = np.linspace(0, n_poses - 1, num_video_frames)
    pose_indices = np.clip(np.round(pose_indices).astype(int), 0, n_poses - 1)

    positions = c2ws[:, :3, 3]       # (N, 3)
    rotations = c2ws[:, :3, :3]      # (N, 3, 3)

    # ------------------------------------------------------------------
    # Per-frame camera-local translation (for joystick)
    # ------------------------------------------------------------------
    local_deltas = np.zeros((n_poses, 3))
    for i in range(1, n_poses):
        world_delta = positions[i] - positions[i - 1]
        # World → camera-local: R^T @ Δp
        local_deltas[i] = rotations[i].T @ world_delta

    # OpenCV camera: x=right, y=down, z=forward
    # Joystick mapping: screen_x = local_x (right), screen_y = -local_z (forward → up)
    joy_vecs = np.stack([local_deltas[:, 0], -local_deltas[:, 2]], axis=-1)  # (N, 2)

    max_mag = np.max(np.linalg.norm(joy_vecs, axis=-1))
    if max_mag > 1e-8:
        joy_vecs_norm = joy_vecs / max_mag  # in [-1, 1]
    else:
        joy_vecs_norm = np.zeros_like(joy_vecs)

    # ------------------------------------------------------------------
    # Trajectory minimap (world XZ top-down)
    # ------------------------------------------------------------------
    traj_xz = positions[:, [0, 2]]  # world X, Z
    traj_range = max(np.ptp(traj_xz, axis=0).max(), 1e-8)
    traj_center = (traj_xz.min(axis=0) + traj_xz.max(axis=0)) / 2.0
    traj_norm = (traj_xz - traj_center) / (traj_range * 0.6)  # normalised with margin

    # Camera forward direction projected onto XZ (for heading arrow)
    cam_fwd_world = rotations[:, :3, 2]   # z-column of R = forward in world
    cam_fwd_xz = cam_fwd_world[:, [0, 2]]
    fwd_len = np.linalg.norm(cam_fwd_xz, axis=-1, keepdims=True)
    fwd_len = np.where(fwd_len < 1e-8, 1.0, fwd_len)
    cam_fwd_xz = cam_fwd_xz / fwd_len

    # ------------------------------------------------------------------
    # Drawing parameters
    # ------------------------------------------------------------------
    H, W = frames[0].shape[:2]
    radius = max(min(H, W) // 10, 30)
    pad = 16

    joy_cx = W - radius - pad          # joystick centre (bottom-right)
    joy_cy = H - radius - pad
    map_cx = radius + pad               # minimap centre (bottom-left)
    map_cy = H - radius - pad

    # Colours (RGBA)
    BG       = (0, 0, 0, 110)
    RING     = (220, 220, 220, 180)
    CROSS    = (255, 255, 255, 45)
    JOY_LINE = (0, 190, 255, 200)
    JOY_DOT  = (0, 210, 255, 255)
    TRAJ_DIM = (120, 120, 120, 80)
    TRAJ_LIT = (0, 190, 255)
    POS_DOT  = (255, 70, 70, 255)
    START_DOT = (0, 230, 80, 220)
    HEADING  = (255, 200, 60, 220)

    # ------------------------------------------------------------------
    # Try to load a small font for labels (fall back to default)
    # ------------------------------------------------------------------
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    # ------------------------------------------------------------------
    # Render each frame
    # ------------------------------------------------------------------
    result_frames: list[np.ndarray] = []
    for fi, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        pi = pose_indices[min(fi, len(pose_indices) - 1)]

        # ============== Joystick (bottom-right) ==============
        # Background
        draw.ellipse(
            [joy_cx - radius, joy_cy - radius, joy_cx + radius, joy_cy + radius],
            fill=BG, outline=RING, width=2,
        )
        # Crosshairs
        cr = radius - 6
        draw.line([(joy_cx - cr, joy_cy), (joy_cx + cr, joy_cy)], fill=CROSS, width=1)
        draw.line([(joy_cx, joy_cy - cr), (joy_cx, joy_cy + cr)], fill=CROSS, width=1)
        # Centre dot (rest position)
        draw.ellipse(
            [joy_cx - 2, joy_cy - 2, joy_cx + 2, joy_cy + 2],
            fill=(255, 255, 255, 60),
        )

        # Joystick knob
        jv = joy_vecs_norm[pi]
        knob_x = joy_cx + jv[0] * (radius - 10)
        knob_y = joy_cy + jv[1] * (radius - 10)  # already flipped (forward = -screen_y)
        # Line from centre to knob
        draw.line([(joy_cx, joy_cy), (knob_x, knob_y)], fill=JOY_LINE, width=3)
        # Knob dot
        kr = 7
        draw.ellipse(
            [knob_x - kr, knob_y - kr, knob_x + kr, knob_y + kr],
            fill=JOY_DOT, outline=(255, 255, 255, 255), width=1,
        )
        # Label
        draw.text((joy_cx - radius, joy_cy - radius - 15), "CAM", fill=(255, 255, 255, 180), font=font)

        # ============== Trajectory minimap (bottom-left) ==============
        draw.ellipse(
            [map_cx - radius, map_cy - radius, map_cx + radius, map_cy + radius],
            fill=BG, outline=RING, width=2,
        )

        def _map_pt(idx):
            """Map pose index to minimap pixel coords."""
            mx = map_cx + traj_norm[idx, 0] * (radius - 6)
            # world Z → screen y (negate so +Z = up on screen)
            my = map_cy - traj_norm[idx, 1] * (radius - 6)
            return mx, my

        # Full trajectory (dimmed)
        for j in range(1, n_poses):
            x1, y1 = _map_pt(j - 1)
            x2, y2 = _map_pt(j)
            draw.line([(x1, y1), (x2, y2)], fill=TRAJ_DIM, width=1)

        # Traversed trajectory (bright, alpha ramp)
        if pi > 0:
            for j in range(1, pi + 1):
                alpha = int(80 + 175 * j / max(pi, 1))
                x1, y1 = _map_pt(j - 1)
                x2, y2 = _map_pt(j)
                draw.line([(x1, y1), (x2, y2)], fill=(*TRAJ_LIT, alpha), width=2)

        # Start marker
        sx, sy = _map_pt(0)
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=START_DOT)

        # Current position + heading arrow
        cx, cy = _map_pt(pi)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=POS_DOT, outline=(255, 255, 255, 255), width=1)
        # Heading arrow (camera forward direction on XZ plane)
        arrow_len = radius * 0.18
        ax = cx + cam_fwd_xz[pi, 0] * arrow_len
        ay = cy - cam_fwd_xz[pi, 1] * arrow_len  # negate Z for screen
        draw.line([(cx, cy), (ax, ay)], fill=HEADING, width=2)
        # Arrowhead
        dx, dy = ax - cx, ay - cy
        perp_x, perp_y = -dy * 0.4, dx * 0.4
        draw.polygon(
            [(ax, ay), (ax - dx * 0.35 + perp_x, ay - dy * 0.35 + perp_y),
             (ax - dx * 0.35 - perp_x, ay - dy * 0.35 - perp_y)],
            fill=HEADING,
        )

        # Label
        draw.text((map_cx - radius, map_cy - radius - 15), "PATH", fill=(255, 255, 255, 180), font=font)
        # Frame counter
        draw.text(
            (map_cx - radius, map_cy + radius + 4),
            f"f {fi+1}/{len(frames)}",
            fill=(180, 180, 180, 160), font=font,
        )

        result_frames.append(np.array(img.convert("RGB")))

    return result_frames


def generate_video(
    pipeline,
    fastvideo_args: FastVideoArgs,
    prompt: str = None,
    image_path: str = None,
    output_path: str = None,
    action_path: str = None,
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
        output_path = os.path.join(OUTPUT_DIR, f"generated_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    
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
    
    # Build extra dict for camera conditioning (if action_path provided)
    extra = {}
    if action_path is not None:
        extra["poses_path"] = os.path.join(action_path, "poses.npy")
        extra["intrinsics_path"] = os.path.join(action_path, "intrinsics.npy")
        # If the intrinsics were calibrated for a different resolution than
        # 480×832, override here:
        # extra["original_height"] = 480
        # extra["original_width"] = 832
        logger.info("Camera conditioning enabled from: %s", action_path)

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
        extra=extra,
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
        
        # Overlay camera joystick if camera data was provided
        if action_path is not None:
            poses_file = os.path.join(action_path, "poses.npy")
            if os.path.isfile(poses_file):
                logger.info("Drawing camera joystick overlay...")
                frames = draw_camera_joystick_on_frames(
                    frames, poses_file, num_frames)
                logger.info("Camera overlay applied to %d frames", len(frames))

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
        "--action_path",
        type=str,
        default=None,
        help="Path to directory containing poses.npy and intrinsics.npy for camera conditioning"
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

    # ---- Parallelism args ----
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)"
    )
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="Sequence parallelism size (default: 1, set >1 to enable SP)"
    )
    parser.add_argument(
        "--hsdp_shard_dim",
        type=int,
        default=1,
        help="FSDP shard dimension for HSDP (default: 1)"
    )
    parser.add_argument(
        "--hsdp_replicate_dim",
        type=int,
        default=1,
        help="HSDP replication dimension (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Load the pipeline
    pipeline, fastvideo_args = load_pipeline(
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_size=args.sp_size,
        hsdp_shard_dim=args.hsdp_shard_dim,
        hsdp_replicate_dim=args.hsdp_replicate_dim,
    )
    
    # Generate video if requested
    if args.generate:
        generate_video(
            pipeline=pipeline,
            fastvideo_args=fastvideo_args,
            prompt=args.prompt,
            image_path=args.image,
            output_path=args.output,
            action_path=args.action_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

"""
# Single GPU:
python playground.py --generate \
    --prompt "A dragon flying to a tower" \
    --image "assets/lingbot/image.jpg" \
    --action_path "assets/lingbot/" \
    --height 480 --width 832 \
    --num_frames 81 --steps 40 \
    --guidance_scale 5.0 --seed 5 2>&1 | tee lingbot.log

# 2-GPU sequence parallel:
torchrun --nproc_per_node=2 playground.py --generate \
    --num_gpus 2 --sp_size 2 --hsdp_shard_dim 2 --hsdp_replicate_dim 1 \
    --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls." \
    --image "assets/lingbot/image.jpg" \
    --action_path "assets/lingbot/" \
    --height 480 --width 832 \
    --num_frames 481 --steps 40 \
    --guidance_scale 5.0 --seed 8 2>&1 | tee lingbot_sp2.log

# 4-GPU HSDP (2-way FSDP shard x 2-way replication):
torchrun --nproc_per_node=4 playground.py --generate \
    --num_gpus 4 --sp_size 2 --hsdp_shard_dim 2 --hsdp_replicate_dim 2 \
    --prompt "A dragon flying to a tower" \
    --image "assets/lingbot/image.jpg" \
    --action_path "assets/lingbot/" \
    --height 480 --width 832 \
    --num_frames 81 --steps 40 \
    --guidance_scale 5.0 --seed 5 2>&1 | tee lingbot_hsdp4.log
"""