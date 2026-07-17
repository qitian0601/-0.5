#!/usr/bin/env bash
#SBATCH --job-name=upload-pick-ee-mix56
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: sbatch $0 <checkpoint_step>" >&2
  exit 2
fi

CKPT_STEP="$1"
REPO_ROOT=/public/home/chenglongyan/workspace/lerobot
TRAIN_DIR=$REPO_ROOT/outputs/train/pi05_pickplace_human_56_mixed_ee_so3_2gpu_any
LOCAL_DIR=$TRAIN_DIR/checkpoints/$CKPT_STEP/pretrained_model
REPO_ID=bryce301best/pickplace_ee_mix_56
CONDA_ROOT=/public/home/chenglongyan/apps/miniconda3

cd "$REPO_ROOT"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lerobot

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1
export JOB_CACHE_ROOT="${TMPDIR:-/tmp}/lerobot_upload_pick_ee_mix56_${CKPT_STEP}_${SLURM_JOB_ID:-manual}"
export HF_HOME="$JOB_CACHE_ROOT/hf_home"
export HF_DATASETS_CACHE="$JOB_CACHE_ROOT/hf_datasets"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" outputs/slurm
trap 'rm -rf "$JOB_CACHE_ROOT"' EXIT

SOURCE_HF_HOME=/public/home/chenglongyan/.cache/huggingface
if [[ -f "$SOURCE_HF_HOME/token" ]]; then
  cp "$SOURCE_HF_HOME/token" "$HF_HOME/token"
fi
if [[ -f "$SOURCE_HF_HOME/stored_tokens" ]]; then
  cp "$SOURCE_HF_HOME/stored_tokens" "$HF_HOME/stored_tokens"
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "missing checkpoint dir: $LOCAL_DIR" >&2
  exit 1
fi

echo "host=$(hostname)"
echo "ckpt=$CKPT_STEP"
echo "local_dir=$LOCAL_DIR"
echo "repo_id=$REPO_ID"
echo "cache=$JOB_CACHE_ROOT"
echo "start=$(date -Is)"

python scripts/upload_hf_with_host_patch.py \
  --repo-id "$REPO_ID" \
  --repo-type model \
  --local-dir "$LOCAL_DIR" \
  --path-in-repo "checkpoints/$CKPT_STEP/pretrained_model" \
  --commit-message "Upload PI0.5 pickplace EE mixed 56 checkpoint $CKPT_STEP"

echo "end=$(date -Is)"
