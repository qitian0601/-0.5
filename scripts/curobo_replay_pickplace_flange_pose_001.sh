#!/usr/bin/env bash
set -euo pipefail

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

DATASET=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/pickplace_flange_pose_001

python ./scripts/replay_nero_dual_ee_pose.py \
  --episode=all \
  --right-channel=nero_right \
  --left-channel=nero_left \
  --firmware-version=V120 \
  --speed-percent=20 \
  --ready-wait-s=3.0 \
  --takeover-time-s=3.0 \
  --ik-backend=curobo \
  --curobo-robot=nero_custom.yml \
  --max-joint-step-rad=1 \
  --fps=30 \
  --interpolate-first-target \
  --joint-target-tolerance-rad=0.06 \
  --joint-wait-timeout-s=0.3 \
  --joint-timeout-error-rad=0.12 \
  --control-dt-s=0.05 \
  --profile-csv=/home/chenglong/workplace/nero_teleop_ws/lerobot/artifacts/curobo_pickplace_flange_pose_001_profile.csv \
  "${DATASET}"
