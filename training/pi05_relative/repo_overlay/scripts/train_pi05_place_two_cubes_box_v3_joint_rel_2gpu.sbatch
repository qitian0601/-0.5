#!/usr/bin/env bash
#SBATCH --job-name=pi05-place-cubes-v3-joint-rel-2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=/public/home/chenglongyan/workspace/lerobot
DATASET_ROOT=$REPO_ROOT/hf_downloads/datasets/place_two_cubes_box_lerobot_v3
OUTPUT_DIR=$REPO_ROOT/outputs/train/pi05_place_two_cubes_box_v3_joint_rel_2gpu
CONDA_ROOT=/public/home/chenglongyan/apps/miniconda3

cd "$REPO_ROOT"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lerobot

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export JOB_CACHE_ROOT="${TMPDIR:-/tmp}/lerobot_place_cubes_v3_train_${SLURM_JOB_ID:-manual}"
export HF_HOME="$JOB_CACHE_ROOT/hf_home"
export HF_DATASETS_CACHE="$JOB_CACHE_ROOT/hf_datasets"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" outputs/slurm
trap 'rm -rf "$JOB_CACHE_ROOT"' EXIT

echo "host=$(hostname)"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "gpus=${SLURM_GPUS_ON_NODE:-2}"
echo "cpus=${SLURM_CPUS_PER_TASK:-64}"
echo "cache=$JOB_CACHE_ROOT"
echo "start=$(date -Is)"

python - <<'PY'
import json
from pathlib import Path

import torch

root = Path("/public/home/chenglongyan/workspace/lerobot/hf_downloads/datasets/place_two_cubes_box_lerobot_v3")
info = json.loads((root / "meta" / "info.json").read_text())
stats = json.loads((root / "meta" / "stats.json").read_text())

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit(
        f"CUDA preflight failed: available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
    )
assert info["features"]["observation.state"]["shape"] == [16]
assert info["features"]["action"]["shape"] == [16]
assert info["total_tasks"] == 1
assert stats["observation.state"]["count"] == [19170]
assert stats["action"]["count"] == [828650], stats["action"]["count"]
print(f"CUDA preflight ok: {torch.cuda.device_count()} visible devices")
print("Dataset preflight ok: 16D joint, task=pickplace, relative action stats count=828650")
PY

accelerate launch --num_processes=2 "$(command -v lerobot-train)" \
  --dataset.repo_id=local/place_two_cubes_box_lerobot_v3 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_place_two_cubes_box_v3_joint_rel_2gpu \
  --policy.pretrained_path=hf_downloads/models/pi05_base \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.use_relative_actions=true \
  --policy.relative_action_type=joint \
  --policy.relative_exclude_joints='["gripper"]' \
  --steps=12000 \
  --save_freq=2000 \
  --log_freq=1 \
  --eval_freq=20000 \
  --policy.device=cuda \
  --batch_size=32 \
  --num_workers=8 \
  --seed=1000

echo "end=$(date -Is)"
