#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂 pickplace_new joint 异步推理 client。
# 请先在另一个终端启动 run_nero_pickplace_new_infer_server.sh。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线，避免运行中访问 Hugging Face。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 可通过环境变量覆盖，不需要改脚本文件。
POLICY_PATH="${NERO_PICKPLACE_NEW_POLICY_PATH:-/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_new/checkpoints/012000/pretrained_model}"
TASK="${NERO_PICKPLACE_NEW_TASK:-pickplace}"

args=(
  # 使用 Nero 双臂机器人配置
  --robot.type=nero_dual
  # 连接本机 policy server
  --server_address=127.0.0.1:8080
  # policy 类型
  --policy_type=pi05
  # 本地 pickplace_new joint PI05 checkpoint
  --policy_path="${POLICY_PATH}"
  # server 侧 policy 推理设备
  --policy_device=cuda
  # client 收到 action 后使用的设备
  --client_device=cpu
  # 任务文本指令
  --task="${TASK}"
  # 三台 RealSense 相机及推理分辨率
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30, warmup_s: 3}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30, warmup_s: 3}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30, warmup_s: 3}}"
  # 右臂 CAN 口
  --robot.right.connection.channel=nero_right
  # 左臂 CAN 口
  --robot.left.connection.channel=nero_left
  # 连接右臂时不 reset
  --robot.right.connection.reset_on_connect=false
  # 连接左臂时不 reset
  --robot.left.connection.reset_on_connect=false
  # 固定 ready pose
  --safety.fixed_ready_pose='{"right_nero_joint_1":-0.01864011641129944,"right_nero_joint_2":-1.72533,"right_nero_joint_3":0.005689773361501515,"right_nero_joint_4":1.8416539734118966,"right_nero_joint_5":-0.055553830090979514,"right_nero_joint_6":-0.028972465583105872,"right_nero_joint_7":1.550797,"right_gripper_width":0.02853,"left_nero_joint_1":0.03324852225049198,"left_nero_joint_2":-1.72533,"left_nero_joint_3":-3.490658503988659e-05,"left_nero_joint_4":1.796868824805722,"left_nero_joint_5":-0.017174039839624202,"left_nero_joint_6":-0.06607816548050531,"left_nero_joint_7":1.550797,"left_gripper_width":0.023507}'
  # 启动前平滑接管到 ready pose 的时间
  --safety.takeover_time_s=6.0
  # policy 目标单步最大关节变化
  --safety.max_policy_step_rad=0.05
  # policy 目标单步最大夹爪变化
  --safety.max_gripper_step_m=0.05
  # false 表示真实控制机械臂
  --safety.dry_run=false
  # 开启 180Hz 高速执行器
  --safety.high_rate_control=true
  # 高速执行器周期，约 180Hz
  --safety.high_rate_dt_s=0.005556
  # 高速执行器单步最大关节变化
  --safety.max_executor_step_rad=0.005
  # 高速执行器单步最大夹爪变化
  --safety.max_executor_gripper_step_m=0.004
  # 开启推理录制
  --trace.enabled=true
  # 录制输出根目录
  --trace.dir=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records
  # trace 每 100 条 flush 一次
  --trace.flush_every=100
  # client 控制/观测频率
  --fps=30
  # ready pose 同步完成后等待按回车再开始 policy control
  --wait_for_enter=true
  # 允许键盘停止
  --keyboard_stop=true
  # 每个 policy chunk 取 50 个 action
  --actions_per_chunk=50
  # action queue 低于 80% chunk 时发送新观测
  --chunk_size_threshold=0.8
  # action 聚合方式；average = 0.5 old + 0.5 new
  --aggregate_fn_name=average
)

python -m lerobot_robot_nero.async_client "${args[@]}"
