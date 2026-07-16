#!/bin/bash
#SBATCH --job-name=pickplace-human56-ee-so3-stats
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.out
#SBATCH --error=/public/home/chenglongyan/workspace/lerobot/outputs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=/public/home/chenglongyan/workspace/lerobot
DATASET_ROOT=$REPO_ROOT/hf_downloads/datasets/pickplace_flange_pose_new_001_camera_rotvec_14d_human_56_mixed
PYTHON=/public/home/chenglongyan/apps/miniconda3/envs/lerobot/bin/python
CONDA_PREFIX=/public/home/chenglongyan/apps/miniconda3/envs/lerobot

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONHOME="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export JOB_CACHE_ROOT="${TMPDIR:-/tmp}/lerobot_pickplace_human56_stats_${SLURM_JOB_ID:-manual}"
export HF_HOME="$JOB_CACHE_ROOT/hf_home"
export HF_DATASETS_CACHE="$JOB_CACHE_ROOT/hf_datasets"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
trap 'rm -rf "$JOB_CACHE_ROOT"' EXIT

echo "host=$(hostname)"
echo "dataset=$DATASET_ROOT"
echo "cpus=${SLURM_CPUS_PER_TASK:-64}"
echo "cache=$JOB_CACHE_ROOT"
echo "start=$(date -Is)"

run_recompute() {
  "$PYTHON" scripts/recompute_ee_so3_relative_stats.py \
    --dataset-root "$DATASET_ROOT" \
    --repo-id local/pickplace_flange_pose_new_001_camera_rotvec_14d_human_56_mixed \
    --chunk-size 50 \
    --num-workers "${SLURM_CPUS_PER_TASK:-64}"
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
import pyarrow.parquet as pq

root = Path("/public/home/chenglongyan/workspace/lerobot/hf_downloads/datasets/pickplace_flange_pose_new_001_camera_rotvec_14d_human_56_mixed")
stats = json.loads((root / "meta" / "stats.json").read_text())
info = json.loads((root / "meta" / "info.json").read_text())
episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", columns=["length"])
lengths = episodes["length"].to_pylist()
expected_count = sum(max(0, int(length) - 50 + 1) for length in lengths) * 50

assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
assert len(stats["observation.state"]["mean"]) == 14
assert len(stats["action"]["mean"]) == 14
assert stats["action"]["count"] == [expected_count], (stats["action"]["count"], expected_count)

raw = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet", columns=["action"])
raw_action = np.asarray(raw["action"].to_pylist(), dtype=np.float64)
raw_mean = raw_action.mean(axis=0)
stats_mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
max_mean_diff = float(np.max(np.abs(raw_mean - stats_mean)))
if max_mean_diff <= 0.05:
    raise AssertionError(f"action stats still look like absolute action stats: max_mean_diff={max_mean_diff}")

print("verified action_count", stats["action"]["count"])
print("verified expected_count", expected_count)
print("verified action_mean_abs", float(np.mean(np.abs(stats_mean))))
print("verified action_std_mean", float(np.mean(stats["action"]["std"])))
print("verified raw_abs_action_mean_diff", max_mean_diff)
PY

echo "end=$(date -Is)"
