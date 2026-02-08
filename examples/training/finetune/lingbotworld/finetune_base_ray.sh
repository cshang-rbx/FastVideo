#!/bin/bash
# LingbotWorld finetuning via Ray trainer.
#
# This is the Ray equivalent of finetune_base.sh.
# The training config lives in finetune_base_ray.yaml — all training
# hyperparameters, model paths, and parallelism settings are defined there.
#
# Three launch modes:
#   1. Ray local  — uses Ray TorchTrainer on the current node
#   2. torchrun   — standard single-node launch (no Ray cluster needed)
#   3. Ray remote — submit to a remote Ray cluster (uncomment below)

set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export TOKENIZERS_PARALLELISM=false
# export WANDB_API_KEY=<your-wandb-api-key>
# export AWS_SECRET_ACCESS_KEY=<your-aws-secret>
# export AWS_ACCESS_KEY_ID=<your-aws-key-id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/finetune_base_ray.yaml"
NUM_GPUS=8

# ── Optional: download model to fast local storage ──
FAST_DIR="/dev/shm/"
if [ ! -d "$FAST_DIR" ]; then
  FAST_DIR="$HOME"
fi

export MODEL_PATH="${FAST_DIR}/fastvideo-lingbot-world-base-cam/"
if [ ! -d "$MODEL_PATH" ]; then
  echo "Downloading model to ${MODEL_PATH} ..."
  s5cmd cp "s3://3dfm-videogen/models/fastvideo-lingbot-world-base-cam/*" "$MODEL_PATH"
fi

# ── Pick a launch mode ──
# Default: Ray local. Set MODE=torchrun or MODE=ray_remote to switch.
MODE="${MODE:-ray_local}"

case "$MODE" in
  ray_local)
    echo "[Launch] Ray local with ${NUM_GPUS} GPUs"
    python -m fastvideo.training.ray.trainer \
      --config "$CONFIG"
    ;;

  torchrun)
    echo "[Launch] torchrun with ${NUM_GPUS} GPUs"
    torchrun \
      --nnodes 1 \
      --nproc_per_node "$NUM_GPUS" \
      -m fastvideo.training.ray.trainer \
      --config "$CONFIG"
    ;;

  ray_remote)
    echo "[Launch] Ray remote with ${NUM_GPUS} GPUs"
    # Requires video_gen ray_launcher and a running Ray cluster.
    # Adjust --trainer, --name, --gpu_type as needed.
    python -m trainer.ray_launcher \
      --remote \
      --trainer fastvideo.training.ray.trainer \
      --config "$CONFIG" \
      --num_gpus "$NUM_GPUS" \
      --name "lingbotworld-finetune" \
      --gpu_type H100 \
      --wandb_api_key "${WANDB_API_KEY:-}"
    ;;

  *)
    echo "Unknown MODE=${MODE}. Use ray_local, torchrun, or ray_remote."
    exit 1
    ;;
esac
