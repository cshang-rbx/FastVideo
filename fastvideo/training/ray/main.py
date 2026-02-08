# SPDX-License-Identifier: Apache-2.0
"""
Ray Train worker entry point for FastVideo training.

This module is the Ray cluster-side orchestrator.  It is invoked by the
launcher (``launcher.py``) either directly via ``python -m`` on a remote
Ray head node, or programmatically from ``launch_ray_local``.

It provides:

* ``train_loop_per_worker`` — the function executed on each Ray worker
  (one per GPU).  It imports the ``Trainer`` class, constructs it from
  the config, and calls ``trainer.train()``.

* ``sanity_check_per_worker`` — a lightweight NCCL all-reduce test that
  runs before the real training to catch infra issues early.

* ``ray_main`` — the top-level function that initialises the Ray cluster,
  configures ``TorchTrainer``, and calls ``.fit()``.

Usage (on Ray head node)::

    python -m fastvideo.training.ray.main \\
        --trainer fastvideo.training.ray.trainer \\
        --config configs/lingbot_finetune.yaml \\
        --num_gpus 16
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Error classification ────────────────────────────────────────────

_NCCL_OOM_PATTERNS = [
    r"NCCL",
    r"ncclRemoteError",
    r"remote process exited",
    r"ProcessGroupNCCL",
    r"error or timeout",
    r"watchdog.*stuck",
    r"CUDA out of memory",
    r"OutOfMemoryError",
]


def _looks_like_nccl_or_oom(exc: BaseException) -> bool:
    txt = str(exc) + "\n" + traceback.format_exc()
    return any(re.search(p, txt, re.I) for p in _NCCL_OOM_PATTERNS)


# ── InfiniBand HCA detection ───────────────────────────────────────

def _get_active_hcas() -> str:
    """Return comma-separated list of active InfiniBand HCA device names."""
    if not Path("/sys/class/infiniband").exists():
        return ""
    try:
        output = subprocess.check_output(
            ["ibv_devinfo"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    hcas, current = [], None
    for line in output.splitlines():
        if line.startswith("hca_id:"):
            current = line.split()[1]
        elif "state:" in line and "PORT_ACTIVE" in line and current:
            hcas.append(current)
            current = None
    return ",".join(hcas)


# ── Config loader ──────────────────────────────────────────────────

def load_config(config_path: str):
    """Load an OmegaConf config with ``_base_`` inheritance support."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if "_base_" in cfg:
        base_cfg = OmegaConf.load(Path(config_path).parent / cfg._base_)
        del cfg._base_
        cfg = OmegaConf.merge(base_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


# ── Sanity check worker ───────────────────────────────────────────

def sanity_check_per_worker(train_loop_config: dict) -> None:
    """Lightweight NCCL / GPU sanity check run before real training."""
    import torch
    import torch.distributed as dist

    _ = train_loop_config  # unused, required by Ray signature

    device = torch.device("cpu")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_count = max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local_rank % device_count)
        device = torch.device("cuda", local_rank % device_count)

    if dist.is_available() and dist.is_initialized():
        tensor = torch.ones(1, device=device) * (dist.get_rank() + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        if dist.get_rank() == 0:
            logger.info(
                "Sanity all-reduce OK (expected=%.0f, got=%.0f)",
                expected, tensor.item(),
            )
    else:
        logger.info("Distributed not initialized; skipping collective check.")


# ── Training worker ────────────────────────────────────────────────

def train_loop_per_worker(train_loop_config: dict) -> None:
    """Per-worker training function executed by Ray TorchTrainer.

    Ray Train has already set ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``,
    ``MASTER_ADDR``, ``MASTER_PORT`` and called
    ``torch.distributed.init_process_group()``.
    """
    # Set InfiniBand HCA list for optimal NCCL transport
    hca_list = _get_active_hcas()
    if hca_list:
        os.environ["NCCL_IB_HCA"] = hca_list
        logger.info("[Worker %s] NCCL_IB_HCA = %s", os.uname().nodename, hca_list)
    else:
        logger.info("[Worker %s] No active HCAs found", os.uname().nodename)

    # Dynamic import of the Trainer module
    trainer_module_path = train_loop_config["trainer_module"]
    trainer_module = importlib.import_module(trainer_module_path)
    trainer_cls = getattr(trainer_module, "Trainer")

    cfg = train_loop_config["cfg"]
    trainer = trainer_cls(cfg, use_ray=True)

    try:
        trainer.train()
    except BaseException as exc:
        if _looks_like_nccl_or_oom(exc):
            logger.error("[NCCL/OOM] %s: %s", type(exc).__name__, exc)
        else:
            logger.error("[ERROR] %s: %s", type(exc).__name__, exc)
        raise


# ── Cluster-level orchestrator ─────────────────────────────────────

def ray_main(args: argparse.Namespace) -> None:
    """Initialise the Ray cluster, run sanity check, then launch training."""
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    cfg = load_config(args.config)

    # Ray timeout tuning
    os.environ.setdefault("RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS", "3600")
    os.environ.setdefault("RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB", "100")
    os.environ.setdefault("RAY_AGENT_REGISTER_TIMEOUT_MS", "3600000")

    # Propagate credentials and debugging env vars to workers
    env_vars = {
        # NCCL tuning
        "NCCL_NET_ENABLE_PXN": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
        "NCCL_TIMEOUT": "120",
        "NCCL_MIN_NCHANNELS": "32",
        "NCCL_MAX_NCHANNELS": "64",
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "RAY_CHDIR_TO_TRIAL_DIR": "0",
        "OMP_NUM_THREADS": "1",
        # Debugging
        "TORCH_CPP_LOG_LEVEL": "INFO",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    }
    # Forward credentials if set
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "WANDB_API_KEY", "WANDB_BASE_URL", "HF_TOKEN"):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val

    # Determine working_dir: the FastVideo project root
    fastvideo_root = Path(__file__).resolve().parents[3]

    ray.init(
        address=getattr(args, "ray_address", "auto"),
        runtime_env={
            "env_vars": env_vars,
            "working_dir": fastvideo_root.as_posix(),
            "excludes": [
                "checkpoints",
                ".git",
                "wandb",
                "*.egg-info",
                "__pycache__",
            ],
        },
    )
    logger.info("Ray initialized: %s", ray.cluster_resources())

    train_loop_config = {
        "cfg": cfg,
        "trainer_module": args.trainer,
    }

    scaling_config = ScalingConfig(
        num_workers=int(args.num_gpus), use_gpu=True)
    run_config = RunConfig(
        failure_config=FailureConfig(max_failures=2))
    torch_config = TorchConfig(backend="nccl", timeout_s=300)

    # Step 1: Sanity check (lightweight NCCL + GPU test)
    logger.info("Running sanity check with %d workers ...", args.num_gpus)
    TorchTrainer(
        sanity_check_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
        torch_config=torch_config,
    ).fit()
    logger.info("Sanity check passed.")

    # Step 2: Real training
    logger.info("Starting training with %d workers ...", args.num_gpus)
    TorchTrainer(
        train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
        torch_config=torch_config,
    ).fit()
    logger.info("Training finished.")


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FastVideo Ray training orchestrator")
    parser.add_argument(
        "--trainer", required=True,
        help="Python module path to the Trainer class "
             "(e.g. fastvideo.training.ray.trainer).")
    parser.add_argument(
        "--config", required=True,
        help="Path to training config YAML.")
    parser.add_argument(
        "--num_gpus", required=True, type=int,
        help="Total number of GPU workers across the cluster.")
    parser.add_argument(
        "--ray_address", default="auto",
        help="Ray cluster address (default: auto-detect).")

    ray_main(parser.parse_args())
