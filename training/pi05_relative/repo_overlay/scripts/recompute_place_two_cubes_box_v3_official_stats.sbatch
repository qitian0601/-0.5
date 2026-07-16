#!/bin/bash
#SBATCH --job-name=place-cubes-v3-rel-stats
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=/public/home/chenglongyan/workspace/lerobot
DATASET_ROOT=$REPO_ROOT/hf_downloads/datasets/place_two_cubes_box_lerobot_v3
PYTHON=/public/home/chenglongyan/apps/miniconda3/envs/lerobot/bin/python
CONDA_PREFIX=/public/home/chenglongyan/apps/miniconda3/envs/lerobot

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONHOME="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export JOB_CACHE_ROOT="${TMPDIR:-/tmp}/lerobot_place_cubes_v3_stats_${SLURM_JOB_ID:-manual}"
export HF_HOME="$JOB_CACHE_ROOT/hf_home"
export HF_DATASETS_CACHE="$JOB_CACHE_ROOT/hf_datasets"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
trap 'rm -rf "$JOB_CACHE_ROOT"' EXIT

echo "host=$(hostname)"
echo "dataset=$DATASET_ROOT"
echo "cpus=${SLURM_CPUS_PER_TASK:-16}"
echo "cache=$JOB_CACHE_ROOT"
echo "start=$(date -Is)"

run_recompute() {
  "$PYTHON" -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/place_two_cubes_box_lerobot_v3 \
    --root "$DATASET_ROOT" \
    --new_repo_id local/place_two_cubes_box_lerobot_v3 \
    --new_root "$DATASET_ROOT" \
    --operation.type recompute_stats \
    --operation.overwrite true \
    --operation.skip_image_video true \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints '["gripper"]'
}

recompute_status=1
for attempt in 1 2 3; do
  set +e
  run_recompute
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    recompute_status=0
    break
  fi
  echo "recompute attempt $attempt failed with status $status" >&2
  sleep 5
done
if [[ "$recompute_status" -ne 0 ]]; then
  exit "$recompute_status"
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/public/home/chenglongyan/workspace/lerobot/hf_downloads/datasets/place_two_cubes_box_lerobot_v3")
info = json.loads((root / "meta" / "info.json").read_text())
stats = json.loads((root / "meta" / "stats.json").read_text())
tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
data = pd.read_parquet(root / "data/chunk-000/file-000.parquet", columns=["task_index", "action"])
episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet", columns=["tasks", "length"])
expected_relative_count = sum(max(0, int(length) - 50 + 1) for length in episodes["length"]) * 50

assert info["total_tasks"] == 1
assert tasks.index.name == "task"
assert list(tasks.index) == ["pickplace"]
assert tasks["task_index"].tolist() == [0]
assert data["task_index"].nunique() == 1 and int(data["task_index"].iloc[0]) == 0
assert episodes["tasks"].map(lambda x: x == ["pickplace"]).all()
assert stats["observation.state"]["count"] == [info["total_frames"]]
assert stats["action"]["count"] == [expected_relative_count], (stats["action"]["count"], expected_relative_count)
raw_action = np.stack(data["action"].to_numpy()).astype(np.float64)
raw_mean = raw_action.mean(axis=0)
stats_mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
max_mean_diff = float(np.max(np.abs(raw_mean - stats_mean)))
if max_mean_diff <= 0.05:
    raise AssertionError(f"action stats still look absolute: max_mean_diff={max_mean_diff}")

print("verified total_frames", info["total_frames"])
print("verified total_tasks", info["total_tasks"])
print("verified task", list(tasks.index)[0])
print("verified state_count", stats["observation.state"]["count"])
print("verified action_count", stats["action"]["count"])
print("verified expected_relative_count", expected_relative_count)
print("verified raw_abs_action_mean_diff", max_mean_diff)
PY

echo "end=$(date -Is)"
