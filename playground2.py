"""
Playground2 script for Wan2.2-I2V-A14B.

This script demonstrates:
1) Loading the I2V training pipeline modules (without dataset wiring)
2) Running a minimal dummy training step (forward + loss + backward)
3) Running a simple I2V generation pass (image + text)

Notes:
- The training path intentionally bypasses dataset/dataloader setup to keep it lightweight.
- For multi-GPU, launch with torchrun and set SP/HSDP sizes via CLI or env vars.
"""

import argparse
import os
import time

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

# Distributed env defaults (torchrun will override)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29505")

from fastvideo.configs.pipelines.wan import Wan2_2_I2V_A14B_Config
from fastvideo.distributed import get_local_torch_device
from fastvideo.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs, TrainingArgs, WorkloadType
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines import build_pipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.utils import maybe_download_model
from fastvideo.training.wan_i2v_training_pipeline import WanI2VTrainingPipeline
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.platforms import current_platform

logger = init_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
# LOCAL_DIR = os.path.join("data", MODEL_ID.replace("/", "-"))
LOCAL_DIR = os.path.join(os.path.expanduser("~"), "data", MODEL_ID.replace("/", "-"))
OUTPUT_DIR = "video_samples_wan2_2_14B_i2v"

EXAMPLE_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing atmosphere of the seaside."
)
EXAMPLE_IMAGE = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _resolve_model_path() -> str:
    logger.info("Downloading/loading model: %s", MODEL_ID)
    model_path = maybe_download_model(MODEL_ID, local_dir=LOCAL_DIR)
    logger.info("Model path: %s", model_path)
    return model_path


# ============================================================================
# Inference pipeline (generation)
# ============================================================================


def load_inference_pipeline(num_gpus: int = 1) -> tuple[object, FastVideoArgs]:
    model_path = _resolve_model_path()

    pipeline_config = Wan2_2_I2V_A14B_Config()
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=num_gpus,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        hsdp_replicate_dim=1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pipeline_config=pipeline_config,
    )

    logger.info("Building inference pipeline...")
    pipeline = build_pipeline(fastvideo_args)
    return pipeline, fastvideo_args


def generate_video_from_image_text(
    pipeline,
    fastvideo_args: FastVideoArgs,
    prompt: str,
    image_path: str,
    output_path: str | None = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"generated_seed{seed}.mp4")

    height_latents = height // 8
    width_latents = width // 8
    num_latent_frames = (num_frames - 1) // 4 + 1
    n_tokens = num_latent_frames * height_latents * width_latents

    batch = ForwardBatch(
        data_type="video",
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
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

    logger.info("Running inference...")
    start_time = time.perf_counter()
    output_batch = pipeline.forward(batch, fastvideo_args)
    gen_time = time.perf_counter() - start_time
    logger.info("Generation completed in %.2f seconds", gen_time)

    samples = output_batch.output
    if samples is None:
        logger.warning("No output generated")
        return None

    videos = rearrange(samples, "b c t h w -> t b c h w")
    frames = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    if batch.save_video:
        imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
        logger.info("Saved video to: %s", output_path)
    return frames


# ============================================================================
# Minimal training step (dummy data)
# ============================================================================


def _build_training_args(
    model_path: str,
    num_gpus: int,
    tp_size: int,
    sp_size: int,
    hsdp_shard_dim: int,
    hsdp_replicate_dim: int,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    train_batch_size: int,
) -> TrainingArgs:
    pipeline_config = Wan2_2_I2V_A14B_Config()
    pipeline_config.dit_precision = "fp32"

    num_latent_t = (num_frames - 1) // 4 + 1

    training_args = TrainingArgs(
        model_path=model_path,
        mode=ExecutionMode.FINETUNING,
        workload_type=WorkloadType.I2V,
        inference_mode=False,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        hsdp_shard_dim=hsdp_shard_dim,
        hsdp_replicate_dim=hsdp_replicate_dim,
        pipeline_config=pipeline_config,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        seed=seed,
        learning_rate=1e-4,
        betas="0.9,0.999",
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        train_batch_size=train_batch_size,
        train_sp_batch_size=train_batch_size,
        num_latent_t=num_latent_t,
        num_height=height,
        num_width=width,
        num_frames=num_frames,
        max_train_steps=1,
    )
    training_args.boundary_ratio = pipeline_config.boundary_ratio
    return training_args


def _init_training_components(
    pipeline: WanI2VTrainingPipeline,
    training_args: TrainingArgs,
) -> None:
    pipeline.training_args = training_args
    pipeline.device = get_local_torch_device()

    # Use the low-noise transformer for the dummy step to keep it simple.
    pipeline.transformer = pipeline.get_module("transformer")
    pipeline.transformer_2 = None
    pipeline.train_transformer_2 = False

    pipeline.set_trainable()
    pipeline.transformer.train()

    params_to_optimize = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    pipeline.optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=training_args.learning_rate,
        betas=tuple(float(x.strip()) for x in training_args.betas.split(",")),
        weight_decay=training_args.weight_decay,
        eps=1e-8,
    )

    pipeline.noise_scheduler = FlowMatchEulerDiscreteScheduler()
    if training_args.boundary_ratio is not None:
        pipeline.boundary_timestep = (
            training_args.boundary_ratio * pipeline.noise_scheduler.num_train_timesteps
        )
    else:
        pipeline.boundary_timestep = None

    pipeline.noise_random_generator = torch.Generator(device="cpu").manual_seed(
        training_args.seed or 0
    )
    pipeline.noise_gen_cuda = torch.Generator(
        device=current_platform.device_name
    ).manual_seed(training_args.seed or 0)

    # Keep scheduler module consistent with training pipeline expectation.
    pipeline.modules["scheduler"] = FlowUniPCMultistepScheduler(
        shift=training_args.pipeline_config.flow_shift
    )


def _dummy_training_batch(
    pipeline: WanI2VTrainingPipeline,
    training_args: TrainingArgs,
    batch_size: int,
    image_token_len: int = 1,
) -> TrainingBatch:
    device = get_local_torch_device()
    dtype = torch.bfloat16 if current_platform.is_cuda() else torch.float32

    transformer = pipeline.get_module("transformer")
    config = transformer.config

    vae_arch = training_args.pipeline_config.vae_config.arch_config
    latent_channels = getattr(vae_arch, "z_dim", 16)
    in_channels = latent_channels
    text_len = getattr(config, "text_len", 512)
    text_dim = getattr(config, "text_dim", 4096)
    image_dim = getattr(config, "image_dim", None)

    height_latent = training_args.num_height // 8
    width_latent = training_args.num_width // 8
    num_latent_t = training_args.num_latent_t

    latents = torch.randn(
        batch_size,
        in_channels,
        num_latent_t,
        height_latent,
        width_latent,
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        text_len,
        text_dim,
        device=device,
        dtype=dtype,
    )
    encoder_attention_mask = torch.ones(
        batch_size,
        text_len,
        device=device,
        dtype=torch.int64,
    )


    image_embeds = None
    if image_dim is not None:
        image_embeds = torch.randn(
            batch_size,
            image_token_len,
            image_dim,
            device=device,
            dtype=dtype,
        )
    image_latents = torch.randn(
        batch_size,
        in_channels,
        num_latent_t,
        height_latent,
        width_latent,
        device=device,
        dtype=dtype,
    )

    training_batch = TrainingBatch()
    training_batch.current_timestep = 0
    training_batch.current_vsa_sparsity = 0.0
    training_batch.latents = latents
    training_batch.encoder_hidden_states = encoder_hidden_states
    training_batch.encoder_attention_mask = encoder_attention_mask
    training_batch.image_embeds = image_embeds
    training_batch.image_latents = image_latents
    return training_batch


def minimal_dummy_train_step(
    pipeline: WanI2VTrainingPipeline,
    training_args: TrainingArgs,
    batch_size: int = 1,
    image_token_len: int = 1,
    do_optimizer_step: bool = True,
) -> TrainingBatch:
    pipeline.optimizer.zero_grad()

    training_batch = _dummy_training_batch(
        pipeline=pipeline,
        training_args=training_args,
        batch_size=batch_size,
        image_token_len=image_token_len,
    )
    # Initialize loss accumulator expected by training pipeline.
    training_batch.total_loss = 0.0
    training_batch = pipeline._normalize_dit_input(training_batch)
    training_batch = pipeline._prepare_dit_inputs(training_batch)
    training_batch = pipeline._build_attention_metadata(training_batch)

    # If image_dim is None, the model has no image_embedder and must not receive
    # encoder_hidden_states_image.
    if getattr(pipeline.transformer.config, "image_dim", None) is None:
        training_batch.input_kwargs = {
            "hidden_states": training_batch.noisy_model_input,
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "timestep": training_batch.timesteps.to(
                get_local_torch_device(), dtype=torch.bfloat16
            ),
            "encoder_attention_mask": training_batch.encoder_attention_mask,
            "return_dict": False,
        }
    else:
        training_batch = pipeline._build_input_kwargs(training_batch)
    training_batch = pipeline._transformer_forward_and_compute_loss(training_batch)

    if do_optimizer_step:
        training_batch = pipeline._clip_grad_norm(training_batch)
        pipeline.optimizer.step()

    return training_batch


def load_training_pipeline_for_dummy_step(
    num_gpus: int,
    tp_size: int,
    sp_size: int,
    hsdp_shard_dim: int,
    hsdp_replicate_dim: int,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    train_batch_size: int,
) -> tuple[WanI2VTrainingPipeline, TrainingArgs]:
    model_path = _resolve_model_path()

    # Initialize distributed + model parallel groups early.
    maybe_init_distributed_environment_and_model_parallel(tp_size, sp_size)

    training_args = _build_training_args(
        model_path=model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        hsdp_shard_dim=hsdp_shard_dim,
        hsdp_replicate_dim=hsdp_replicate_dim,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        train_batch_size=train_batch_size,
    )

    pipeline = WanI2VTrainingPipeline(model_path, training_args)
    _init_training_components(pipeline, training_args)
    return pipeline, training_args


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-I2V-A14B Playground2")

    parser.add_argument("--generate", action="store_true", help="Run I2V generation")
    parser.add_argument("--train-step", action="store_true", help="Run a dummy training step")

    parser.add_argument("--prompt", type=str, default=EXAMPLE_PROMPT, help="Prompt")
    parser.add_argument("--image", type=str, default=EXAMPLE_IMAGE, help="Image path or URL")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=832, help="Output video width")
    parser.add_argument("--num-frames", type=int, default=81, help="Num frames (4n+1)")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    # Dummy training args
    parser.add_argument("--train-height", type=int, default=64, help="Dummy train height")
    parser.add_argument("--train-width", type=int, default=64, help="Dummy train width")
    parser.add_argument("--train-frames", type=int, default=9, help="Dummy train frames")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Dummy train batch size")
    parser.add_argument("--image-token-len", type=int, default=1, help="Dummy image token length")

    # Parallel settings
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    parser.add_argument("--num-gpus", type=int, default=world_size, help="Total GPUs")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--sp-size", type=int, default=world_size, help="SP size")
    parser.add_argument("--hsdp-shard-dim", type=int, default=world_size, help="HSDP shard dim")
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1, help="HSDP replicate dim")

    args = parser.parse_args()

    if args.generate:
        pipeline, fastvideo_args = load_inference_pipeline(num_gpus=1)
        generate_video_from_image_text(
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

    if args.train_step:
        training_pipeline, training_args = load_training_pipeline_for_dummy_step(
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            sp_size=args.sp_size,
            hsdp_shard_dim=args.hsdp_shard_dim,
            hsdp_replicate_dim=args.hsdp_replicate_dim,
            seed=args.seed,
            num_frames=args.train_frames,
            height=args.train_height,
            width=args.train_width,
            train_batch_size=args.train_batch_size,
        )
        training_batch = minimal_dummy_train_step(
            pipeline=training_pipeline,
            training_args=training_args,
            batch_size=args.train_batch_size,
            image_token_len=args.image_token_len,
            do_optimizer_step=True,
        )
        logger.info("Dummy training step loss: %.6f", training_batch.total_loss or 0.0)


if __name__ == "__main__":
    main()

"""
torchrun --nproc_per_node=2 playground2.py \
    --train-step \
    --num-gpus 2 --sp-size 2 --hsdp-shard-dim 2 --hsdp-replicate-dim 1
"""