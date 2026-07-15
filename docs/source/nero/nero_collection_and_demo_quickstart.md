
## 1. 进入环境

```bash
cd /home/chenglong/workplace/nero_teleop_ws
source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
```

如果命令需要在 `lerobot` 目录下运行：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
```

## 2. 双臂手动主从控制
同时控制双臂：

```bash
cd /home/chenglong/workplace/nero_teleop_ws
conda activate lerobot
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair both
```

指定底层运动接口为 `move_js`：

```bash
cd /home/chenglong/workplace/nero_teleop_ws
conda activate lerobot
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair both --move-method move_js
```

## 3. 保存末端位姿数据集

### 3.1 只保存末端位姿数据集

这个版本使用 `nero-record-dual-joint`，采集结束后会按当前记录流程做裁剪/预处理。`--record_joint_dataset=false` 表示不保存 joint dataset，只保存 `flange_dataset`。

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
nero-record-dual-joint \
  --right_leader.type=so101_8dof_leader \
  --right_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00 \
  --right_leader.id=so101_8dof_wzh \
  --left_leader.type=so101_8dof_leader \
  --left_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00 \
  --left_leader.id=so101_8dof_pair1 \
  --robot.type=nero_dual \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=chenglong/pickplace_joint_new_003 \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_joint_new_003 \
  --flange_dataset.repo_id=chenglong/pickplace_flange_pose_new_003 \
  --flange_dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_flange_pose_new_003 \
  --dataset.single_task="pickplace" \
  --dataset.fps=30 \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=200 \
  --dataset.push_to_hub=false \
  --flange_dataset.push_to_hub=false \
  --dataset.video=true \
  --flange_dataset.video=false \
  --record_joint_dataset=true \
  --manual_stop=true
```

如果要同时保存裁剪后的 joint dataset 和 flange pose dataset，把最后的 `--record_joint_dataset=false` 改成：

```bash
--record_joint_dataset=true
```

### 3.2 保存未裁剪的原始数据

这个版本使用 `nero-record-dual-joint-raw`，用于保留静止段和完整原生 episode。下面命令会同时保存 raw joint dataset 和 raw flange pose dataset。

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
nero-record-dual-joint-raw \
  --right_leader.type=so101_8dof_leader \
  --right_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00 \
  --right_leader.id=so101_8dof_wzh \
  --left_leader.type=so101_8dof_leader \
  --left_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00 \
  --left_leader.id=so101_8dof_pair1 \
  --robot.type=nero_dual \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=chenglong/pickplace_joint_new_003_raw \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_joint_new_003_raw \
  --flange_dataset.repo_id=chenglong/pickplace_flange_pose_new_003_raw \
  --flange_dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_flange_pose_new_003_raw \
  --dataset.single_task="pickplace" \
  --dataset.fps=30 \
  --dataset.num_episodes=80 \
  --dataset.episode_time_s=200 \
  --dataset.push_to_hub=false \
  --flange_dataset.push_to_hub=false \
  --dataset.video=true \
  --flange_dataset.video=false \
  --record_joint_dataset=true \
  --manual_stop=true
```

## 4. scripts/demo 推理脚本

所有 demo 推理脚本都在：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/demo
```

统一运行方式是先启动 server，再在第二个终端启动 client。

### 4.2 pickplace_new joint 推理

终端 1：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/pickplace_new_server.sh
```

终端 2：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/pickplace_new_client.sh
```

常用覆盖项：

```bash
NERO_PICKPLACE_NEW_POLICY_PATH=/path/to/pretrained_model \
NERO_PICKPLACE_NEW_TASK="pickplace" \
bash scripts/demo/pickplace_new_client.sh
```

### 4.3 pickplace EE 推理

终端 1：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/pickplace_ee_new_server.sh
```

终端 2：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/pickplace_ee_new_client.sh
```

常用覆盖项：

```bash
NERO_PICKPLACE_EE_POLICY_PATH=/path/to/pretrained_model \
NERO_RIGHT_CAN=nero_right \
NERO_LEFT_CAN=nero_left \
NERO_DRY_RUN=true \
bash scripts/demo/pickplace_ee_client.sh
```

### 4.4 叠毛巾 joint 推理

终端 1：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/towel_server.sh
```

终端 2：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/towel_client.sh
```

### 4.5 叠毛巾 EE 推理

终端 1：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/towel_ee_server.sh
```

终端 2：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/demo/towel_ee_client.sh
```

常用覆盖项：

```bash
NERO_EE_POLICY_PATH=/path/to/pretrained_model \
NERO_RIGHT_CAN=nero_right \
NERO_LEFT_CAN=nero_left \
NERO_DRY_RUN=true \
bash scripts/demo/towel_ee_client.sh
```

## 5. 运行前检查

确认 CAN 和相机：

```bash
ip link show nero_right
ip link show nero_left
lerobot-find-cameras
```

当前固定相机序列号：

```text
front: 324422301659
left_wrist: 244222077114
right_wrist: 244222070153
```

推理 client 会先移动到 fixed ready pose，并在 `--wait_for_enter=true` 时等待按回车。推荐等双臂稳定、目标物摆好、三路相机画面正常后再按回车进入 policy control。
