# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Ray training launcher.

Unified entry point that supports three launch modes:

1. **Local torchrun** (default) — spawns ``torchrun`` as a subprocess.
2. **Ray local** (``--ray_local``) — uses ``ray.train.torch.TorchTrainer``
   on all GPUs of the current node.
3. **Ray remote** (``--remote``) — submits a job to a remote Ray cluster
   via ``roblox_ml.training.launch_distributed_training``.

Examples
--------

Local (torchrun)::

    python -m fastvideo.training.ray.launcher \\
        --trainer fastvideo.training.ray.trainer \\
        --config examples/training/finetune/lingbotworld/finetune_base_ray.yaml \\
        --num_gpus 8

Ray local::

    python -m fastvideo.training.ray.launcher \\
        --ray_local \\
        --trainer fastvideo.training.ray.trainer \\
        --config examples/training/finetune/lingbotworld/finetune_base_ray.yaml \\
        --num_gpus 8

Ray remote::

    python -m fastvideo.training.ray.launcher \\
        --remote \\
        --trainer fastvideo.training.ray.trainer \\
        --config examples/training/finetune/lingbotworld/finetune_base_ray.yaml \\
        --num_gpus 16 \\
        --name lingbotworld-finetune \\
        --gpu_type H100 \\
        --wandb_api_key $WANDB_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# NOTE: Do NOT import from fastvideo at module level.  Ray workers
# deserialise this module before CUDA is available, and
# ``fastvideo/__init__.py`` eagerly imports Triton kernels that
# require an active CUDA driver.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# FastVideo project root (3 levels up from this file)
_FASTVIDEO_ROOT = Path(__file__).resolve().parents[3]


def _prepare_config_paths(config_path: str) -> tuple[str, str]:
    """Return (absolute_path, relative_path) for a config file."""
    abs_path = Path(config_path)
    if not abs_path.is_absolute():
        abs_path = (Path.cwd() / abs_path).resolve()
    try:
        rel_path = abs_path.relative_to(_FASTVIDEO_ROOT)
    except ValueError:
        rel_path = abs_path
    return str(abs_path), rel_path.as_posix()


# ── Mode 1: Local torchrun ─────────────────────────────────────────

def launch_local(
    trainer_module: str, config_file: str, num_gpus: int
) -> None:
    """Spawn ``torchrun`` on the current node."""
    command = [
        "torchrun",
        "--nproc_per_node", str(num_gpus),
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_port", "21501",
        "-m", trainer_module,
        "--config", config_file,
        "--num_gpus", str(num_gpus),
    ]
    logger.info("Launching local training:\n  %s", " ".join(command))
    subprocess.run(command, check=True)


# ── Mode 2: Ray local ─────────────────────────────────────────────

def launch_ray_local(
    trainer_module: str, config_file: str, num_gpus: int
) -> None:
    """Use ``ray.train.torch.TorchTrainer`` on local GPUs."""
    import torch
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from fastvideo.training.ray.main import (
        load_config,
        sanity_check_per_worker,
        train_loop_per_worker,
    )

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available for Ray local launch.")

    if num_gpus > 0 and num_gpus != available_gpus:
        logger.warning(
            "--num_gpus=%d ignored; using all %d local GPU(s).",
            num_gpus, available_gpus,
        )
    num_gpus = available_gpus
    logger.info("Using %d local GPU(s) via Ray.", num_gpus)

    logger.info("Loading config from: %s", config_file)
    cfg = load_config(config_file)
    logger.info("Config loaded successfully. Keys: %s", list(cfg.keys())[:10])

    # Convert OmegaConf DictConfig to plain dict before Ray serialization.
    # This prevents fastvideo imports during deserialization (before CUDA is ready).
    # The dict will be converted back to DictConfig in train_loop_per_worker
    # after CUDA is initialized.
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    train_loop_config = {
        "cfg": cfg_dict,  # Plain dict, not DictConfig
        "trainer_module": trainer_module,
    }

    scaling_config = ScalingConfig(num_workers=num_gpus, use_gpu=True)
    run_config = RunConfig(failure_config=FailureConfig(max_failures=2))

    # Sanity check
    TorchTrainer(
        sanity_check_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    ).fit()

    # Real training
    TorchTrainer(
        train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    ).fit()


# ── Mode 3: Ray remote (cluster submission) ────────────────────────

def launch_remote(
    trainer_module: str,
    config_file: str,
    num_gpus: int,
    job_name: str,
    gpu_type: str,
    wandb_api_key: str,
    docker_image: str,
    tier_selection: str,
) -> None:
    """Submit to a remote Ray cluster via ``roblox_ml``."""
    try:
        from roblox_ml import cluster
        from roblox_ml.training import launch_distributed_training
    except ImportError as exc:
        raise ImportError(
            "Remote launch requires the 'roblox_ml' package. "
            "Install it or use --ray_local instead."
        ) from exc

    entry_point = [
        "python", "-m", "fastvideo.training.ray.main",
        "--trainer", trainer_module,
        "--config", config_file,
        "--num_gpus", str(num_gpus),
    ]

    envvars = {
        "RAY_CHDIR_TO_TRIAL_DIR": "0",
        "FI_PROVIDER": "efa",
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "PYTHONPATH": "./:${PYTHONPATH}",
        "WANDB_API_KEY": wandb_api_key,
        "WANDB_BASE_URL": os.environ.get(
            "WANDB_BASE_URL", "https://api.wandb.ai"),
        "TORCH_CPP_LOG_LEVEL": "INFO",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "NCCL_DEBUG": "INFO",
        "OMP_NUM_THREADS": "1",
    }

    # Forward AWS credentials from cluster secrets or env
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        secret_name = f"{key}_VIDEOGEN"
        try:
            envvars[key] = cluster.get_secret(secret_name)
        except Exception:
            val = os.environ.get(key, "")
            if val:
                envvars[key] = val

    # EFA libraries for multi-node NCCL
    if num_gpus >= 8:
        envvars["LD_LIBRARY_PATH"] = (
            "/opt/gdrcopy/lib:"
            "/usr/local/cuda/extras/CUPTI/lib64:"
            "/usr/local/nccl-rdma-sharp-plugins/lib:"
            "/usr/lib64:"
            "/usr/local/nvidia/lib:"
            "/usr/local/nvidia/lib64:"
            "/opt/amazon/openmpi/lib:"
            "/opt/nccl/build/lib:"
            "/opt/amazon/efa/lib:"
            "/opt/aws-ofi-nccl/install/lib"
        )

    worker_environments = {
        "RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper": "true"
    }

    logger.info("Submitting remote job '%s' with %d GPUs (%s)",
                job_name, num_gpus, gpu_type)
    launch_distributed_training(
        job_name=job_name,
        num_gpus=num_gpus,
        entry_point=entry_point,
        working_dir=str(_FASTVIDEO_ROOT),
        env_vars=envvars,
        gpu_type=gpu_type,
        full_image_name=docker_image,
        pip_packages=["wandb"],
        worker_environments=worker_environments,
        tier_selection=tier_selection,
        ray_version="2.50.0",
        enable_nvme=True,
        mount_efs=False,
    )


# ── CLI ────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser("FastVideo Ray launcher")

    # Launch mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--remote", action="store_true",
        help="Submit to a remote Ray cluster.")
    mode.add_argument(
        "--ray_local", action="store_true",
        help="Use Ray TorchTrainer on local GPUs.")

    # Common args
    parser.add_argument(
        "--trainer", type=str, required=True,
        help="Python module path to the Trainer class "
             "(e.g. fastvideo.training.ray.trainer).")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML.")
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use.")

    # Remote-only args
    parser.add_argument(
        "--name", type=str, default=None,
        help="Remote job name (required for --remote).")
    parser.add_argument(
        "--gpu_type", type=str, default="H100",
        choices=["H200_ML", "H100", "A100_80G", "A100_40G"],
        help="GPU type for remote execution.")
    parser.add_argument(
        "--docker_image", type=str,
        default="docker.artifactory.rbx.com/mlblox/3dfm-video-gen-ray:1.0.47",
        help="Docker image for the remote job.")
    parser.add_argument(
        "--wandb_api_key", type=str,
        default=os.environ.get("WANDB_API_KEY"),
        help="Weights & Biases API key.")
    parser.add_argument(
        "--tier_selection", type=str, default="0_1_2",
        help="Job tier selection (e.g. 0_1_2).")

    args = parser.parse_args(argv)
    local_config, remote_config = _prepare_config_paths(args.config)

    if args.remote:
        # Validate remote-specific args
        if args.num_gpus % 8 != 0:
            raise ValueError("--num_gpus must be divisible by 8 for remote runs.")
        if not args.name:
            raise ValueError("Remote runs require --name.")
        if not args.wandb_api_key:
            raise ValueError("Remote runs require --wandb_api_key.")

        launch_remote(
            args.trainer,
            remote_config,
            args.num_gpus,
            args.name,
            args.gpu_type,
            args.wandb_api_key,
            args.docker_image,
            args.tier_selection,
        )

    elif args.ray_local:
        launch_ray_local(args.trainer, local_config, args.num_gpus)

    else:
        launch_local(args.trainer, local_config, args.num_gpus)


if __name__ == "__main__":
    main()
