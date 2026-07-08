#!/usr/bin/env bash
set -euo pipefail

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_ROOT=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table

datasets=(
  "${DATA_ROOT}/bus_table_01_ep000_joints"
  "${DATA_ROOT}/bus_table_01_ep001_joints"
  "${DATA_ROOT}/bus_table_01_ep002_joints"
  "${DATA_ROOT}/bus_table_01_ep003_joints"
  "${DATA_ROOT}/bus_table_01_ep004_joints"
)

common_args=(
  --robot.type=nero_dual
  --robot.right.connection.channel=nero_right
  --robot.left.connection.channel=nero_left
  --robot.right.connection.reset_on_connect=false
  --robot.left.connection.reset_on_connect=false
  --takeover_time_s=4.0
  --takeover_dt_s=0.02
  --high_rate_control=true
  --high_rate_dt_s=0.005556
  --max_replay_step_rad=0.08
  --max_replay_gripper_step_m=0.05
  --max_executor_step_rad=0.005
  --max_executor_gripper_step_m=0.004
)

for dataset in "${datasets[@]}"; do
  if [[ ! -d "${dataset}" ]]; then
    echo "Missing dataset: ${dataset}" >&2
    exit 1
  fi

  echo
  echo "Ready to replay: ${dataset}"
  read -r -p "Press Enter to start, or Ctrl-C to stop..."

  nero-replay-dual-joint \
    "${common_args[@]}" \
    --dataset.root="${dataset}" \
    --dataset.episode=0
done

echo
echo "Finished replaying first 5 bus_table joint episodes."
