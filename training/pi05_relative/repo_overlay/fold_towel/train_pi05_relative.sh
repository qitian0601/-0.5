#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_DIR}/config.env}"
LEROBOT_ENV_BIN="${LEROBOT_ENV_BIN:-/public/home/chenglongyan/apps/miniconda3/envs/lerobot/bin}"
PYTHON_BIN="${PYTHON_BIN:-${LEROBOT_ENV_BIN}/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${LEROBOT_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${LEROBOT_ENV_BIN}/lerobot-train}"
LEROBOT_ENV_DIR="$(cd "${LEROBOT_ENV_BIN}/.." && pwd)"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${PROJECT_DIR}/config.template.env"
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

required_vars=(
  DATASET_REPO_ID
  DATASET_ROOT
  OUTPUT_DIR
  JOB_NAME
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    printf 'Missing required value: %s\n' "${var_name}" >&2
    printf 'Fill it in %s before running training.\n' "${CONFIG_PATH}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"

cache_suffix="${SLURM_JOB_ID:-manual}"
export HF_HOME="${HF_HOME:-/tmp/chenglongyan_hf_home_${cache_suffix}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/chenglongyan_hf_datasets_${cache_suffix}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LD_LIBRARY_PATH="${LEROBOT_ENV_DIR}/lib:${LEROBOT_ENV_DIR}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ "${DEVICE:-cuda}" == cuda* ]]; then
  "${PYTHON_BIN}" - "${NUM_PROCESSES:-1}" <<'PY'
import sys

import torch

expected = int(sys.argv[1])
available = torch.cuda.is_available()
count = torch.cuda.device_count()
if not available or count < expected:
    raise SystemExit(
        f"CUDA preflight failed: torch.cuda.is_available()={available}, "
        f"torch.cuda.device_count()={count}, expected at least {expected}."
    )
print(f"CUDA preflight ok: {count} visible device(s), expected {expected}.")
PY
fi

if [[ "${RUN_TRAIN_PREFLIGHT:-true}" == "true" ]]; then
  preflight_ok=false
  for attempt in $(seq 1 "${TRAIN_PREFLIGHT_ATTEMPTS:-8}"); do
    printf 'Training preflight attempt %s/%s...\n' "${attempt}" "${TRAIN_PREFLIGHT_ATTEMPTS:-8}" >&2
    if "${PYTHON_BIN}" - "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${POLICY_PRETRAINED_PATH}" "${RELATIVE_ACTION_TYPE:-ee_so3}" <<'PY'
import sys
from pathlib import Path

import pandas
import scipy
import sympy
import torch
import torchvision

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = sys.argv[1]
dataset_root = Path(sys.argv[2])
policy_path = Path(sys.argv[3])
relative_action_type = sys.argv[4]
expected_dim_by_relative_type = {
    "ee_so3": 14,
    "ee_local_se3": 16,
}
expected_dim = expected_dim_by_relative_type.get(relative_action_type)

if not policy_path.exists():
    raise FileNotFoundError(policy_path)

tasks = pandas.read_parquet(dataset_root / "meta" / "tasks.parquet")
if tasks.index.name != "task" or list(tasks.columns) != ["task_index"]:
    raise ValueError(
        "meta/tasks.parquet must use task strings as the pandas index and "
        f"only task_index as a column, got index={tasks.index.name!r}, columns={list(tasks.columns)!r}"
    )
if not isinstance(tasks.iloc[0].name, str):
    raise TypeError(f"Task index entry must be a string, got {type(tasks.iloc[0].name).__name__}")

dataset = LeRobotDataset(
    repo_id=repo_id,
    root=dataset_root,
    download_videos=False,
    return_uint8=True,
    video_backend="torchcodec",
)
if expected_dim is not None:
    if tuple(dataset.meta.features["observation.state"]["shape"]) != (expected_dim,):
        raise ValueError(dataset.meta.features["observation.state"]["shape"])
    if tuple(dataset.meta.features["action"]["shape"]) != (expected_dim,):
        raise ValueError(dataset.meta.features["action"]["shape"])
task = dataset.meta.tasks.iloc[int(dataset.hf_dataset[0]["task_index"])].name
if not isinstance(task, str):
    raise TypeError(f"Dataset task mapping must return a string, got {type(task).__name__}")
required_stats = [
    "action",
    "observation.state",
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
missing_stats = [key for key in required_stats if key not in dataset.meta.stats]
if missing_stats:
    raise ValueError(f"Missing dataset stats: {missing_stats}")
print(
    "Training preflight ok:",
    f"frames={dataset.num_frames}",
    f"episodes={dataset.num_episodes}",
    f"task={task!r}",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"pandas={pandas.__version__}",
    f"scipy={scipy.__version__}",
    f"sympy={sympy.__version__}",
)
PY
    then
      preflight_ok=true
      break
    fi
    sleep "${TRAIN_PREFLIGHT_SLEEP:-20}"
  done

  if [[ "${preflight_ok}" != true ]]; then
    printf 'Training preflight failed after %s attempts.\n' "${TRAIN_PREFLIGHT_ATTEMPTS:-8}" >&2
    exit 3
  fi
fi

train_args=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --policy.type="${POLICY_TYPE:-pi05}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --policy.pretrained_path="${POLICY_PRETRAINED_PATH:-hf_downloads/models/pi05_base}"
  --policy.compile_model="${COMPILE_MODEL:-false}"
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}"
  --wandb.enable="${WANDB_ENABLE:-false}"
  --policy.push_to_hub="${PUSH_TO_HUB:-false}"
  --policy.dtype="${DTYPE:-bfloat16}"
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER:-false}"
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY:-false}"
  --policy.chunk_size="${CHUNK_SIZE:-50}"
  --policy.n_action_steps="${N_ACTION_STEPS:-50}"
  --policy.use_relative_actions="${USE_RELATIVE_ACTIONS:-true}"
  --policy.relative_action_type="${RELATIVE_ACTION_TYPE:-ee_so3}"
  --policy.relative_exclude_joints="${RELATIVE_EXCLUDE_JOINTS:-[\"gripper\"]}"
  --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE:-false}"
  --dataset.image_transforms.max_num_transforms="${IMAGE_TRANSFORMS_MAX_NUM_TRANSFORMS:-3}"
  --dataset.image_transforms.random_order="${IMAGE_TRANSFORMS_RANDOM_ORDER:-false}"
  --steps="${STEPS:-5000}"
  --save_freq="${SAVE_FREQ:-1000}"
  --log_freq="${LOG_FREQ:-1}"
  --eval_freq="${EVAL_FREQ:-20000}"
  --policy.device="${DEVICE:-cuda}"
  --batch_size="${BATCH_SIZE:-32}"
  --num_workers="${NUM_WORKERS:-8}"
  --seed="${SEED:-1000}"
)

if (( ${NUM_PROCESSES:-1} > 1 )); then
  exec "${ACCELERATE_BIN}" launch --num_processes="${NUM_PROCESSES}" "${LEROBOT_TRAIN_BIN}" "${train_args[@]}"
fi

exec "${LEROBOT_TRAIN_BIN}" "${train_args[@]}"
