#!/bin/bash
# LingbotWorld finetuning via Ray trainer.
#
# This is the Ray equivalent of finetune_base.sh.
# The training config lives in finetune_base_ray.yaml — all training
# hyperparameters, model paths, and parallelism settings are defined there.
#
# Three launch modes (set MODE env var):
#   MODE=torchrun   — standard single-node launch (default, no Ray needed)
#   MODE=ray_local  — uses Ray TorchTrainer on the current node
#   MODE=ray_remote — submit to a remote Ray cluster

set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export TOKENIZERS_PARALLELISM=false
# export WANDB_API_KEY=<your-wandb-api-key>
# export AWS_SECRET_ACCESS_KEY=<your-aws-secret>
# export AWS_ACCESS_KEY_ID=<your-aws-key-id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/finetune_base_ray.yaml"
NUM_GPUS=8

# ── cd to FastVideo project root (needed for module resolution) ──
FASTVIDEO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
cd "$FASTVIDEO_ROOT"

# ── Optional: stage model + data to fast local storage ──
FAST_DIR="/dev/shm"
if [ ! -d "$FAST_DIR" ]; then
  FAST_DIR="$HOME"
fi

# Model
export MODEL_PATH="${FAST_DIR}/fastvideo-lingbot-world-base-cam/"
if [ ! -d "$MODEL_PATH" ]; then
  echo "Downloading model to ${MODEL_PATH} ..."
  s5cmd cp "s3://3dfm-videogen/models/fastvideo-lingbot-world-base-cam/*" "$MODEL_PATH"
fi

# Data
DATA_DIR="data/crush-smol_processed_i2v/combined_parquet_dataset/"
if [ -d "$DATA_DIR" ] && [ ! -d "${FAST_DIR}/combined_parquet_dataset/" ]; then
  echo "Copying data to ${FAST_DIR}/combined_parquet_dataset/ ..."
  cp -r "$DATA_DIR" "${FAST_DIR}/combined_parquet_dataset/"
fi
if [ -d "${FAST_DIR}/combined_parquet_dataset/" ]; then
  export DATA_PATH="${FAST_DIR}/combined_parquet_dataset/"
else
  export DATA_PATH="$DATA_DIR"
fi

# Validation
export VALIDATION_DATASET_FILE="${SCRIPT_DIR}/validation.json"

# ── Pick a launch mode ──
# Default: torchrun (no Ray dependency). Set MODE=ray_local or MODE=ray_remote.
MODE="${MODE:-torchrun}"

case "$MODE" in
  torchrun)
    echo "[Launch] torchrun with ${NUM_GPUS} GPUs"
    torchrun \
      --nnodes 1 \
      --nproc_per_node "$NUM_GPUS" \
      -m fastvideo.training.ray.trainer \
      --config "$CONFIG"
    ;;

  ray_local)
    echo "[Launch] Ray local with ${NUM_GPUS} GPUs"
    python -m fastvideo.training.ray.launcher \
      --ray_local \
      --trainer fastvideo.training.ray.trainer \
      --config "$CONFIG" \
      --num_gpus "$NUM_GPUS"
    ;;

  ray_remote)
    echo "[Launch] Ray remote with ${NUM_GPUS} GPUs"
    python -m fastvideo.training.ray.launcher \
      --remote \
      --trainer fastvideo.training.ray.trainer \
      --config "$CONFIG" \
      --num_gpus "$NUM_GPUS" \
      --name "lingbotworld-finetune" \
      --gpu_type H100 \
      --wandb_api_key "${WANDB_API_KEY:-}"
    ;;

  *)
    echo "Unknown MODE=${MODE}. Use torchrun, ray_local, or ray_remote."
    exit 1
    ;;
esac
