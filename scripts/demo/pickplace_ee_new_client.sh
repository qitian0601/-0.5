#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂 pickplace EE-local-SE3 异步推理 client。
# 请先在另一个终端启动 pickplace_ee_new_server.sh。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线，避免运行中访问 Hugging Face。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 可通过环境变量覆盖，不需要改脚本文件。
POLICY_PATH="${NERO_PICKPLACE_EE_POLICY_PATH:-/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee_new/7.7/pickplace_001_ee_local_se3_4gpu/checkpoints/012000/pretrained_model}"
RIGHT_CAN="${NERO_RIGHT_CAN:-nero_right}"
LEFT_CAN="${NERO_LEFT_CAN:-nero_left}"
DRY_RUN="${NERO_DRY_RUN:-false}"
TASK="${NERO_PICKPLACE_EE_TASK:-Pick up the cube and place it in the target area. If no cube is visible, stay in the ready pose.}"

args=(
  # 使用 Nero 双臂机器人配置。
  --robot.type=nero_dual
  # 连接本机 policy server。
  --server_address=127.0.0.1:8080
  # policy 类型。
  --policy_type=pi05
  # pickplace EE-local-SE3 PI05 checkpoint，默认 016000。
  --policy_path="${POLICY_PATH}"
  # server 侧 policy 推理设备。
  --policy_device=cuda
  # client 收到 action 后使用的设备；控制逻辑留在 CPU。
  --client_device=cpu
  # 任务文本指令，与训练时 task 保持一致。
  --task="${TASK}"
  # 启用 EE-local-SE3 action 模式：policy 输出 16D EE action，再经 cuRobo IK 转 joint action。
  --action_mode=ee_local_se3
  # 三台 RealSense 相机及 EE 模型训练时的推理分辨率。
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30, warmup_s: 3}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30, warmup_s: 3}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30, warmup_s: 3}}"
  # 右臂 CAN 口；默认 nero_right，可用 NERO_RIGHT_CAN 覆盖。
  --robot.right.connection.channel="${RIGHT_CAN}"
  # 左臂 CAN 口；默认 nero_left，可用 NERO_LEFT_CAN 覆盖。
  --robot.left.connection.channel="${LEFT_CAN}"
  # 连接右臂时不 reset。
  --robot.right.connection.reset_on_connect=false
  # 连接左臂时不 reset。
  --robot.left.connection.reset_on_connect=false
  # hand-eye 文件保存的是 camera -> base；client 内部会按需要取逆。
  "--right_handeye_camera_to_base_yaml=/home/chenglong/workplace/nero_teleop_ws/lerobot/相机_机械臂标定/handeye_result_right(1).yml"
  "--left_handeye_camera_to_base_yaml=/home/chenglong/workplace/nero_teleop_ws/lerobot/相机_机械臂标定/handeye_result_left(1).yml"
  # Nero SDK 读出的末端姿态是 Euler，当前按 xyz 解释。
  --ee_euler_order=xyz
  # EE policy 里保留的 base/head 两维，目前固定为 0。
  --ee_base_or_head_x=0.0
  --ee_base_or_head_y=0.0
  # cuRobo IK 配置；输出 joint target 后仍走原 joint safety/high-rate executor/move_js。
  --curobo_robot_file=nero_custom.yml
  --curobo_num_seeds=64
  --curobo_position_threshold=0.05
  --curobo_rotation_threshold=0.08
  --curobo_device=cuda
  # 固定 ready pose，仍是 joint pose，用于启动前对齐两条臂。
  --safety.fixed_ready_pose='{"right_nero_joint_1":-0.01864011641129944,"right_nero_joint_2":-1.72533,"right_nero_joint_3":0.005689773361501515,"right_nero_joint_4":1.8416539734118966,"right_nero_joint_5":-0.055553830090979514,"right_nero_joint_6":-0.028972465583105872,"right_nero_joint_7":1.550797,"right_gripper_width":0.02853,"left_nero_joint_1":0.03324852225049198,"left_nero_joint_2":-1.72533,"left_nero_joint_3":-3.490658503988659e-05,"left_nero_joint_4":1.796868824805722,"left_nero_joint_5":-0.017174039839624202,"left_nero_joint_6":-0.06607816548050531,"left_nero_joint_7":1.550797,"left_gripper_width":0.023507}'
  # 启动前平滑接管到 ready pose 的时间。
  --safety.takeover_time_s=6.0
  # IK 前的 EE 空间前置限幅，先把 policy 末端目标裁成更容易求解的小步。
  --safety.max_ee_position_step_m=0.04
  --safety.max_ee_rotation_step_rad=0.15
  # IK 后的 policy joint target 单步最大关节变化；保留为 joint 空间兜底。
  --safety.max_policy_step_rad=0.05
  # policy target 单步最大夹爪变化。
  --safety.max_gripper_step_m=0.05
  # false 表示真实控制机械臂；可用 NERO_DRY_RUN=true 覆盖。
  --safety.dry_run="${DRY_RUN}"
  # 暂时关闭 IK 失败回 ready 恢复，先单独测试 EE 前置限幅效果。
  --safety.recover_on_ik_failure=false
  # 开启 180Hz 高速执行器。
  --safety.high_rate_control=true
  # 高速执行器周期，约 180Hz。
  --safety.high_rate_dt_s=0.005556
  # 高速执行器单步最大关节变化。
  --safety.max_executor_step_rad=0.005
  # 高速执行器单步最大夹爪变化。
  --safety.max_executor_gripper_step_m=0.004
  # 开启推理录制。
  --trace.enabled=true
  # 录制输出根目录。
  --trace.dir=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records
  # trace 每 100 条 flush 一次。
  --trace.flush_every=100
  # client 控制/观测频率。
  --fps=30
  # ready pose 同步完成后等待按回车再开始 policy control。
  --wait_for_enter=true
  # 允许键盘停止。
  --keyboard_stop=true
  # 每个 policy chunk 取 50 个 action。
  --actions_per_chunk=50
  # action queue 低于 80% chunk 时发送新观测。
  --chunk_size_threshold=0.8
  # action 聚合方式；average = 0.5 old + 0.5 new，比 weighted_average 更平滑。
  --aggregate_fn_name=average
)

python -m lerobot_robot_nero.async_client "${args[@]}"
