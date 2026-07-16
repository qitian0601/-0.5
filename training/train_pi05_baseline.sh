#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO_ID:?Set DATASET_REPO_ID to a local LeRobot dataset path or Hub repo id}"
: "${PRETRAINED_PATH:?Set PRETRAINED_PATH to the Pi0.5 base checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_nero_baseline}"
JOB_NAME="${JOB_NAME:-pi05_nero_baseline}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${PRETRAINED_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.optimizer_lr="${LEARNING_RATE}" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.scheduler_decay_lr=3e-6 \
  --wandb.enable="${WANDB_ENABLE}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}"

