#!/usr/bin/env bash
set -euo pipefail

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table/bus_table_01
episode=${1:-0}

common_args=(
  --robot.type=nero_dual
  --robot.right.connection.channel=nero_right
  --robot.left.connection.channel=nero_left
  --robot.right.connection.reset_on_connect=false
  --robot.left.connection.reset_on_connect=false
  --robot.right.connection.speed_percent=20
  --robot.left.connection.speed_percent=20
  --robot.right.command.move_method=move_js
  --robot.left.command.move_method=move_js
  --robot.right.command.max_step_rad=0.05
  --robot.left.command.max_step_rad=0.05
  --dataset.fps=20
  --takeover_time_s=3.0
  --takeover_dt_s=0.02
)

replay_episode() {
  local episode="$1"

  if [[ ! -d "${DATASET}" ]]; then
    echo "Missing dataset: ${DATASET}" >&2
    exit 1
  fi

  echo
  echo "Ready to replay ${DATASET} episode ${episode} with move_js."
  read -r -p "Press Enter to start, or Ctrl-C to stop..."

  nero-replay-dual-joint \
    "${common_args[@]}" \
    --dataset.root="${DATASET}" \
    --dataset.episode="${episode}"
}

if [[ "${episode}" == "all" ]]; then
  for episode_idx in $(seq 0 29); do
    replay_episode "${episode_idx}"
  done
else
  replay_episode "${episode}"
fi
