# Nero 双臂关节数据采集

本文档记录 Nero 双臂关节数据采集的当前可用流程，覆盖硬件检查、双臂同步采集、数据验证、回放和部署推理。

执行采集、回放或推理前，必须确认双臂周围安全。回放和推理会直接驱动双 Nero 运动。

如果只需要当前常用命令，优先看精简版：[Nero 采集和 Demo 脚本速查](nero_collection_and_demo_quickstart.md)。

## Note
hf 镜像站: https://hf-mirror.com/

## 1. 当前配置

### 1.1 数据字段

当前数据集只保存 Nero 侧数据，不把 SO101 原始读数作为训练字段：

```text
observation.state: Nero 实际关节状态，rad；夹爪宽度，m
action: 实际发送给 Nero 的绝对关节目标，rad；夹爪宽度，m
image: RealSense 彩色图像
```

双臂字段应包含：

```text
action:
  right_nero_joint_1 ... right_nero_joint_7
  left_nero_joint_1 ... left_nero_joint_7
  right_gripper_width
  left_gripper_width

observation.state:
  right_nero_joint_1 ... right_nero_joint_7
  left_nero_joint_1 ... left_nero_joint_7
  right_gripper_width
  left_gripper_width

images:
  observation.images.front
  observation.images.left_wrist
  observation.images.right_wrist
```

### 1.2 硬件绑定

当前双臂和三路 RealSense 按固定设备 ID 绑定：

```text
右臂 pair0:
  SO101 port: /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00
  SO101 id: so101_8dof_wzh
  Nero CAN: nero_right

左臂 pair1:
  SO101 port: /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00
  SO101 id: so101_8dof_pair1
  Nero CAN: nero_left

front camera:
  RealSense serial_number_or_name: 324422301659

left_wrist camera:
  RealSense serial_number_or_name: 244222077114

right_wrist camera:
  RealSense serial_number_or_name: 244222070153
```

### 1.3 SO101 校准文件

SO101 8DOF leader 使用 LeRobot `so_leader` 校准目录：

```text
/home/chenglong/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

如果主从同步位置明显不对，优先检查：

```text
so_leader/so101_8dof_wzh.json
so_leader/so101_8dof_pair1.json
```

### 1.4 单臂/双臂手动主从控制

进入环境：

```bash
cd /home/chenglong/workplace/nero_teleop_ws
conda activate lerobot
```

单独控制右臂 `pair0`：

```bash
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair pair0
```

单独控制左臂 `pair1`：

```bash
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair pair1
```

同时控制双臂：

```bash
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair both
```

如果要指定控制方式，例如 `move_js`：

```bash
python tools/so101_j1to7_to_real_nero_j1to7_movejs.py --pair both --move-method move_js
```

## 2. 采集前检查

### 2.1 进入环境

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
conda activate lerobot
```

### 2.2 检查硬件连接

检查串口、CAN 和相机：

```bash
ls -l /dev/serial/by-id/
ip link show nero_right
ip link show nero_left
lerobot-find-cameras
```

确认相机 serial 与当前配置一致：

```text
front: 324422301659
left_wrist: 244222077114
right_wrist: 244222070153
```

### 2.3 串口权限

给两个 SO101 串口临时授权：

```bash
scripts/nero-grant-so101-permissions.sh \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00 \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00
```

长期授权可以把当前用户加入 `dialout`，然后重新登录：

```bash
sudo usermod -aG dialout $USER
```

## 3. 双臂数据采集
（手眼标定流程）1.录制一段采集视频，用机械臂抓着标定板，停顿15次。保存完数据之后，让codex分析出停顿的范围，例如xxx帧到xxx帧是停顿的。然后让他挑选出停顿最稳定的那一帧。将这一帧的信息保留下来，包括图片，末端位姿。末端位姿需要保留原生的欧拉角以及转换后的四元数。

### 3.1 采集命令

下面命令是当前可直接运行的双臂采集模板：

旧模板，采集完立刻执行裁剪预处理，然后保存
```bash
nero-record-dual-joint \
  --right_leader.type=so101_8dof_leader \
  --right_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00 \
  --right_leader.id=so101_8dof_wzh \
  --left_leader.type=so101_8dof_leader \
  --left_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00 \
  --left_leader.id=so101_8dof_pair1 \
  --robot.type=nero_dual \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=chenglong/fold_towel_0695 \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_0695 \
  --dataset.single_task="fold_towel" \
  --dataset.fps=30 \
  --dataset.num_episodes=40 \
  --dataset.episode_time_s=200 \
  --dataset.push_to_hub=false \
  --manual_stop=true
```

新模板，录制后直接保存，不进行预处理，之后会进行后处理

```bash
nero-record-dual-joint-raw \
  --right_leader.type=so101_8dof_leader \
  --right_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00 \
  --right_leader.id=so101_8dof_wzh \
  --left_leader.type=so101_8dof_leader \
  --left_leader.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532789-if00 \
  --left_leader.id=so101_8dof_pair1 \
  --robot.type=nero_dual \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=chenglong/fold_towel_new01 \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_new01 \
  --dataset.single_task="fold_towel" \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=200 \
  --dataset.push_to_hub=false \
  --manual_stop=true


```

离线后处理模板，从 raw dataset 读取数据，裁剪每条 episode 的静止头尾，并写成新的 dataset：

```bash
nero-trim-dual-joint-dataset \
  --repo_id=chenglong/fold_towel_06 \
  --root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_06 \
  --new_repo_id=chenglong/fold_towel_06_trimmed \
  --new_root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_06_trimmed \
  --overwrite=false
```

只检查裁剪结果、不写入新数据集时，加：

```bash
--dry_run=true
```

通常需要按任务修改以下参数：

```text
--dataset.repo_id: 数据集仓库名
--dataset.root: 本地数据集保存目录
--dataset.single_task: LeRobot task 文本
--dataset.num_episodes: 计划采集 episode 数
--dataset.episode_time_s: 单条 episode 最大时长
--new_repo_id: 后处理输出数据集仓库名
--new_root: 后处理输出数据集本地目录
```

### 3.2 只保存末端位姿数据集

双臂录制脚本可以同时生成 joint 数据集和 EE/flange pose 数据集。只想保存末端位姿数据集时，不需要改代码，在命令里加入：

```bash
--record_joint_dataset=false
```

`dataset.*` 参数仍然需要保留，因为录制脚本会继续从这里读取 `single_task`、`fps`、`num_episodes`、`episode_time_s` 等通用录制参数；真正要保存的末端位姿数据集路径由 `flange_dataset.*` 指定。

只保存 `pickplace_flange_pose_006` 的模板：

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
  --dataset.repo_id=chenglong/pickplace_joint_new_0077 \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_joint_new_0077 \
  --flange_dataset.repo_id=chenglong/pickplace_flange_pose_new_0077 \
  --flange_dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new/pickplace_flange_pose_new_0077 \
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

如果要录原始版、不在录制时裁剪，把命令里的 `nero-record-dual-joint` 换成 `nero-record-dual-joint-raw`，其他参数保持一致。

### 3.3 交互流程

1. 程序连接双 SO101、双 Nero 和相机。
2. 程序提示即将同步时，确认双臂周围安全后按 `ENTER`。
3. Nero 会平滑移动到当前两个 SO101 对应的位置。
4. 进入待机主从同步，按 `ENTER` 开始当前 episode。
5. 操作完成后按 `ENTER` 结束当前 episode。
6. 如果使用旧模板，程序裁剪静止片段后提示 `ENTER=保存，r=重录当前 episode，q=结束并保存已有数据`。
7. 如果使用新 raw 模板，程序不裁剪，提示 `ENTER=保存完整 episode，r=重录当前 episode，q=结束并保存已有数据`。
8. 输入 `r` 后回车会丢弃刚录完的数据并重录当前 episode，episode 编号不增加。
9. 输入 `q` 后回车会提前结束并保存已有 episode；刚录完但未保存的 episode 会被丢弃。
10. episode 之间仍保持主从同步，可手动复位。
11. 全部 episode 录完后，程序保持主从同步，复位完成后按 `ENTER` 关闭。

旧模板会在保存前裁剪 episode 开头和结尾的静止片段；新 raw 模板会完整保存，之后用 `nero-trim-dual-joint-dataset` 统一后处理。

## 4. 频率和性能

主从控制频率和数据集采样频率是分开的：

```text
Nero 主从控制周期: control_dt_s=0.006，约 166 Hz
数据集保存频率: --dataset.fps=30
相机频率: camera fps=30
```

推荐保持：

```text
--dataset.fps=30
camera fps=30
camera resolution=640x480
```

录制时如果看到控制循环变慢的 warning，含义是 Python 循环被相机读取、状态读取或写缓存拖慢，没有稳定达到 166 Hz。

处理建议：

1. 如果 replay 轨迹正常，数据仍可用于初步训练。
2. 如果操作手感明显变差，优先降低相机分辨率或采样频率。
3. 不建议通过增大 `control_dt_s` 解决采集性能问题。

## 5. 数据验证

采集完成后检查数据目录，例如：

```bash
find /home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_02 -maxdepth 3 -type f | sort
```

验证重点：

1. `meta/`、episode 数据和图像数据存在。
2. `action` 字段是发送给 Nero 的绝对关节目标。
3. `observation.state` 字段是 Nero 实际关节状态。
4. 三路相机图像字段存在：`front`、`left_wrist`、`right_wrist`。

当前已验证的回放方式是：平滑同步到第一帧 action，然后按 `30 Hz` 将 episode 中的 `action` 发送给 `NeroDualRobot`。

最近一次双臂两相机测试保存了 1 条 episode、406 帧，回放后末端关节误差约为：

```text
right max error: 0.014 rad
left max error: 0.017 rad
```

## 6. 上传数据集

使用 Hugging Face 镜像站上传本地数据集，例如把 `fold_towel_05` 上传到 `bryce301best/ttttttt`：

```bash
cd /home/chenglong/workplace/nero_teleop_ws

HF_ENDPOINT=https://hf-mirror.com \
hf upload bryce301best/ttttttt \
  /home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_05 \
  . \
  --repo-type dataset
```

如果还没有登录：

```bash
HF_ENDPOINT=https://hf-mirror.com hf auth login
```

## 7. 双臂数据回放

使用 Nero 专用回放命令：

```bash
nero-replay-dual-joint \
  --robot.type=nero_dual \
  --dataset.root=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_02 \
  --dataset.episode=0
```

参数说明：

```text
--dataset.root: 本地 LeRobot 数据集目录
--dataset.episode: 要回放的 episode_index，从 0 开始
--dataset.fps: 可选，覆盖 meta/info.json 中的 fps

```

回放流程：

1. 读取指定 episode 的 `action`。
2. 连接双 Nero，不连接 SO101，不打开相机。
3. 先用 `takeover_time_s=2.0` 平滑移动到该 episode 的第一帧 action。
4. 按数据集 `meta/info.json` 中的 `fps` 回放；如果需要覆盖，可加 `--dataset.fps=30`。
5. 回放结束后自动断开双 Nero。


## 8. 部署推理

部署推理分为 policy server 和 Nero client 两部分。先启动 server，再启动 client。

### 8.1 启动 policy server

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
```

### 8.2 启动 Nero client

当前记录了两组 client 命令：`move_js` 和 `move_j`。

`move_js` 使用三路相机、安全 ready pose 和高频执行器参数：

```bash
python -m lerobot_robot_nero.async_client \
  --robot.type=nero_dual \
  --server_address=127.0.0.1:8080 \
  --policy_path=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_bus_table_01_rel_ckpts/checkpoints/005000/pretrained_model \
  --task="Bus table" \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30}}" \
  --safety.fixed_ready_pose='{"right_nero_joint_1":-0.01864011641129944,"right_nero_joint_2":-1.72533,"right_nero_joint_3":0.005689773361501515,"right_nero_joint_4":1.8416539734118966,"right_nero_joint_5":-0.055553830090979514,"right_nero_joint_6":-0.028972465583105872,"right_nero_joint_7":1.550797,"right_gripper_width":0.02853,"left_nero_joint_1":0.03324852225049198,"left_nero_joint_2":-1.72533,"left_nero_joint_3":-3.490658503988659e-05,"left_nero_joint_4":1.796868824805722,"left_nero_joint_5":-0.017174039839624202,"left_nero_joint_6":-0.06607816548050531,"left_nero_joint_7":1.550797,"left_gripper_width":0.023507}' \
  --safety.takeover_time_s=6.0 \
  --safety.max_policy_step_rad=0.05 \
  --safety.max_gripper_step_m=0.05 \
  --safety.dry_run=false \
  --safety.high_rate_control=true \
  --safety.high_rate_dt_s=0.005556 \
  --safety.max_executor_step_rad=0.005 \
  --safety.max_executor_gripper_step_m=0.004 \
  --trace.enabled=true \
  --trace.dir=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records \
  --fps=30 \
  --actions_per_chunk=50
```

当前 fold_towel 推理推荐使用脚本：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_infer_server.sh
bash scripts/run_nero_infer_client.sh
```

### 8.2.1 EE 版本推理脚本

2026-06-22 新增 EE-local-SE3 版本推理脚本，用于
`bryce301best/fold_towel_ee` 下载得到的新版 PI05 EE checkpoint。
这组脚本不会替换旧 joint-action 推理脚本；旧脚本仍是
`run_nero_infer_server.sh` / `run_nero_infer_client.sh`。

两个终端分别启动：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_ee_infer_server.sh
```

另一个终端：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_ee_infer_client.sh
```

EE 版本脚本路径：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_server.sh
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_client.sh
```

EE client 默认关键参数：

```text
policy_path: /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best/pi05_fold_towel_ee_local_se3_rel_8gpu/checkpoints/020000/pretrained_model
action_mode: ee_local_se3
policy_device: cuda
client_device: cpu
curobo_device: cuda
right arm CAN: nero_right
left arm CAN: nero_left
task: fold_towel
actions_per_chunk: 50
chunk_size_threshold: 0.8
aggregate_fn_name: average
trace.enabled: true
```

EE 版本链路：

```text
RealSense images + Nero flange pose observation
-> 16D EE-local-SE3 observation.state
-> PI05 EE checkpoint outputs 16D EE action
-> camera-frame EE target 转回 Nero base frame
-> cuRobo IK 解成左右臂 7D joint target
-> 复用原 joint safety / high-rate executor
-> NeroDualRobot.send_action()
-> move_js
```

可以通过环境变量临时覆盖，不需要改脚本文件：

```bash
# 先 dry-run，不真实驱动机械臂
NERO_DRY_RUN=true bash scripts/run_nero_ee_infer_client.sh

# 如果现场 CAN 线临时需要覆盖，可按右 nero_right、左 nero_left 显式指定
NERO_RIGHT_CAN=nero_right NERO_LEFT_CAN=nero_left bash scripts/run_nero_ee_infer_client.sh

# 如果要测试其它 EE checkpoint
NERO_EE_POLICY_PATH=/path/to/pretrained_model bash scripts/run_nero_ee_infer_client.sh
```

EE server 脚本仍启动通用 async policy server，并保留当前 RTC 运行参数：

```text
--obs_similarity_atol=0.09
--async_rtc.enabled=true
--async_rtc.latency_quantile=0.9
--async_rtc.debug_dump.enabled=true
```

注意：当前检查到 `020000` EE checkpoint 的 `config.json` 中
`rtc_config = null`，所以 server 会保留 async RTC 开关，但实际 RTC
guidance 取决于 checkpoint 是否带有启用的 `rtc_config`。

### 8.2.2 pick place 任务 EE 版本推理命令

本节是 **pick place 任务** 的 EE-local-SE3 推理命令，不是 fold towel 命令。
pick place EE-local-SE3 推理脚本用于
`bryce301best/pickplace_ee` 下载得到的 PI05 EE checkpoint。
默认使用 `016000`，也可以通过环境变量切换到 `014000`。

pick place 任务的 server 启动命令：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_pickplace_ee_infer_server.sh
```

pick place 任务的 client 启动命令：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_pickplace_ee_infer_client.sh
```

pick place EE 脚本路径：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_pickplace_ee_infer_server.sh
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_pickplace_ee_infer_client.sh
```

pick place EE client 默认关键参数：

```text
policy_path: /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee/checkpoints/016000/pretrained_model
action_mode: ee_local_se3
policy_device: cuda
client_device: cpu
curobo_device: cuda
right arm CAN: nero_right
left arm CAN: nero_left
task: First place the yellow cube into the front box with the right arm, then place the green cube into the front box with the left arm.
actions_per_chunk: 50
chunk_size_threshold: 0.8
aggregate_fn_name: average
trace.enabled: true
```

可以通过环境变量临时覆盖，不需要改脚本文件：

```bash
# 先 dry-run，不真实驱动机械臂
NERO_DRY_RUN=true bash scripts/run_nero_pickplace_ee_infer_client.sh

# 如果现场 CAN 线临时需要覆盖，可按右 nero_right、左 nero_left 显式指定
NERO_RIGHT_CAN=nero_right NERO_LEFT_CAN=nero_left bash scripts/run_nero_pickplace_ee_infer_client.sh

# 如果要测试 014000 checkpoint
NERO_PICKPLACE_EE_POLICY_PATH=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee/checkpoints/014000/pretrained_model \
  bash scripts/run_nero_pickplace_ee_infer_client.sh

# 如果要临时覆盖任务文本
NERO_PICKPLACE_EE_TASK="First place the yellow cube into the front box with the right arm, then place the green cube into the front box with the left arm." \
  bash scripts/run_nero_pickplace_ee_infer_client.sh
```

### 8.2.3 当前 joint 推理脚本内容备份

以下内容备份自当前两个推理启动脚本，便于脚本丢失或误删后恢复。

`scripts/run_nero_infer_server.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂异步推理 server。先启动这个脚本，再在另一个终端启动 client 脚本。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线加载本地 checkpoint，避免 server 卡在 huggingface.co 网络请求。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  # 监听所有网卡，client 可通过 127.0.0.1:8080 连接
  --host=0.0.0.0
  # gRPC server 端口
  --port=8080
  # server 侧环境频率，用于 RTC 延迟 step 换算
  --fps=30
  # 观测相似过滤阈值，越小越容易触发新推理
  --obs_similarity_atol=0.08
  # 开启 async RTC 调度
  --async_rtc.enabled=true
  # RTC 延迟估计使用最近延迟的 p90，避免首包慢推理长期锁死 inference_delay
  --async_rtc.latency_quantile=0.9
  # 将 RTC 内部统计写入本次录制目录 rtc_debug.jsonl
  --async_rtc.debug_dump.enabled=true
)

python -m lerobot.async_inference.policy_server "${args[@]}"
```

`scripts/run_nero_infer_client.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

# Nero 双臂异步推理 client。请先在另一个终端启动 run_nero_infer_server.sh。

cd /home/chenglong/workplace/nero_teleop_ws/lerobot

source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 强制离线，避免运行中访问 Hugging Face。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  # 使用 Nero 双臂机器人配置
  --robot.type=nero_dual
  # 连接本机 policy server
  --server_address=127.0.0.1:8080
  # policy 类型
  --policy_type=pi05
  # 本地 PI05 checkpoint
  --policy_path=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model/checkpoints/012000/pretrained_model
  # server 侧 policy 推理设备
  --policy_device=cuda
  # client 收到 action 后使用的设备
  --client_device=cpu
  # 任务文本指令
  --task=fold_towel
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
  # action 聚合方式；average = 0.5 old + 0.5 new，比 weighted_average 更平滑
  --aggregate_fn_name=average
)

python -m lerobot_robot_nero.async_client "${args[@]}"
```

如果需要展开成完整 client 命令，当前参数如下，包含 `reset_on_connect=false`、trace 记录、`actions_per_chunk=50`、`chunk_size_threshold=0.8` 和 `aggregate_fn_name=average`：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
conda activate lerobot

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m lerobot_robot_nero.async_client \
  --robot.type=nero_dual \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --policy_path=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model/checkpoints/012000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --task="fold_towel" \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '324422301659', width: 1280, height: 800, fps: 30, warmup_s: 3}, left_wrist: {type: intelrealsense, serial_number_or_name: '244222077114', width: 640, height: 480, fps: 30, warmup_s: 3}, right_wrist: {type: intelrealsense, serial_number_or_name: '244222070153', width: 640, height: 480, fps: 30, warmup_s: 3}}" \
  --robot.right.connection.channel=nero_right \
  --robot.left.connection.channel=nero_left \
  --robot.right.connection.reset_on_connect=false \
  --robot.left.connection.reset_on_connect=false \
  --safety.fixed_ready_pose='{"right_nero_joint_1":-0.01864011641129944,"right_nero_joint_2":-1.72533,"right_nero_joint_3":0.005689773361501515,"right_nero_joint_4":1.8416539734118966,"right_nero_joint_5":-0.055553830090979514,"right_nero_joint_6":-0.028972465583105872,"right_nero_joint_7":1.550797,"right_gripper_width":0.02853,"left_nero_joint_1":0.03324852225049198,"left_nero_joint_2":-1.72533,"left_nero_joint_3":-3.490658503988659e-05,"left_nero_joint_4":1.796868824805722,"left_nero_joint_5":-0.017174039839624202,"left_nero_joint_6":-0.06607816548050531,"left_nero_joint_7":1.550797,"left_gripper_width":0.023507}' \
  --safety.takeover_time_s=6.0 \
  --safety.max_policy_step_rad=0.05 \
  --safety.max_gripper_step_m=0.05 \
  --safety.dry_run=false \
  --safety.high_rate_control=true \
  --safety.high_rate_dt_s=0.005556 \
  --safety.max_executor_step_rad=0.005 \
  --safety.max_executor_gripper_step_m=0.004 \
  --trace.enabled=true \
  --trace.dir=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records \
  --trace.flush_every=100 \
  --fps=30 \
  --wait_for_enter=true \
  --keyboard_stop=true \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.8 \
  --aggregate_fn_name=average
```

### 8.3 退出推理

推理启动后，`async_client` 支持键盘退出。需要退出时，在 Nero client 终端输入以下任一命令并回车：

```text
q
s
stop
e
exit
```

`--keyboard_stop=true` 表示启动一个键盘监听线程。推理运行时，在 Nero client 终端输入退出词并回车，它会调用 `client.stop()`，停止 policy 控制循环、断开 Nero client 和 policy server 连接。

原始推理退出流程不会自动回到额外 pose，也不会主动 disable 夹爪或双臂。若需要回收、下使能或额外移动，应在推理完全退出后用单独工具执行，并在命令和文档中明确标注，避免和推理 ready pose 混用。

部署时通常需要修改：

```text
--server_address: policy server 地址
--policy_path: 训练输出的 pretrained_model 目录
--task: policy 使用的任务文本
--fps: client 请求和执行频率
--actions_per_chunk: 每次请求的动作块长度
```

开启 `--trace.enabled=true` 后，每次推理会生成一个目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/YYYYMMDD_HHMMSS/
  meta.json
  trace.jsonl
  replay_actions.jsonl
```

`trace.jsonl` 记录 policy action chunk、队列中实际执行的 policy action、安全限步后的目标、高频执行器每一步、每条臂进入 SDK 前的 `move_js` / `move_j` 指令、以及 SDK 下发后立即读取到的 Nero 关节反馈。

`replay_actions.jsonl` 只记录最终高频执行器发送给双 Nero 的 16 维 action，用于直接回放：

```bash
python -m lerobot_robot_nero.replay_inference_trace \
  --robot.type=nero_dual \
  --trace.path=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/YYYYMMDD_HHMMSS/replay_actions.jsonl \
  --trace.fps=180 \
  --takeover_time_s=2.0
```

## 9. 最近新增文件和目录整理

本节只整理近期 Nero 双臂采集、推理、RTC 调试相关的常用文件和目录，方便之后定位脚本、录制数据和代码改动。

### 9.1 推理启动脚本

两个终端分别启动 server 和 client：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_infer_server.sh
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_infer_client.sh
```

EE-local-SE3 版本推理脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_server.sh
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_client.sh
```

EE 版本用于新版 fold_towel_ee checkpoint，默认右臂 `nero_right`、左臂 `nero_left`，
并开启 `--action_mode=ee_local_se3`、hand-eye 转换和 cuRobo IK。

server 脚本包含：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
--obs_similarity_atol=0.09
--async_rtc.enabled=true
--async_rtc.latency_quantile=0.9
--async_rtc.debug_dump.enabled=true
```

client 脚本包含当前实机推理关键参数：

```text
right arm: nero_right
left arm: nero_left
front camera: 324422301659, 1280x800@30
left_wrist camera: 244222077114, 640x480@30
right_wrist camera: 244222070153, 640x480@30
actions_per_chunk: 50
chunk_size_threshold: 0.8
aggregate_fn_name: average
safety.high_rate_control: true
safety.high_rate_dt_s: 0.005556
reset_on_connect: false
trace.enabled: true
```

这两个脚本已经显式 `source /home/chenglong/miniconda3/etc/profile.d/conda.sh`，因此可以直接 `bash scripts/run_nero_infer_server.sh` 和 `bash scripts/run_nero_infer_client.sh`，不依赖交互式 shell 里的 `conda init`。

### 9.2 推理录制和 RTC 调试输出

推理录制根目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/
```

每次推理会创建一个按时间命名的子目录，例如：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_110944/
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_110056/
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_101330/
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260610_184240/
```

典型文件：

```text
meta.json             本次推理配置、机器人配置、相机配置、policy 路径
trace.jsonl           policy/action queue/高频执行器/CAN 下发等详细 trace
replay_actions.jsonl  可用于 replay_inference_trace 的最终执行 action
rtc_debug.jsonl       RTC 内部统计；开启 --async_rtc.debug_dump.enabled=true 后生成
```

`rtc_debug.jsonl` 和本次录制数据放在同一个目录，便于把 RTC 内部量与 `trace.jsonl` 对齐分析。当前保存的是轻量统计量，不保存完整 tensor。

2026-06-11 重点分析过的推理目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_101330/
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_110944/
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_114417/
```

其中 `20260611_114417` 结论是：没有明显长期吃空，`aggregate_fn_name=average` 后 queue 大 gap 很少；剩余顿挫更可能来自 RTC `inference_delay` 估计过保守和 target 频繁变化，而不是 action queue 长时间为空。

### 9.3 相机检查输出

相机检查图片目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/camera_checks/
```

已保存的 front 相机检查图示例：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/camera_checks/front/front_324422301659_1280x800_30_20260610_181247.png
```

### 9.4 训练 checkpoint 和推理 policy

当前 fold towel PI05 推理使用的 checkpoint：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model/checkpoints/012000/pretrained_model
```

其中 `config.json` 里当前 RTC 静态参数为：

```text
rtc_config.enabled: true
rtc_config.prefix_attention_schedule: LINEAR
rtc_config.max_guidance_weight: 10.0
rtc_config.execution_horizon: 10
```

### 9.5 近期关键源码

Nero 双臂实机配置和异步推理 client：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot_robot_nero/config_nero.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot_robot_nero/async_client.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot_robot_nero/trace.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot_robot_nero/robot_nero_dual.py
```

异步推理和 RTC 调度相关代码：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/async_inference/configs.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/async_inference/helpers.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/async_inference/policy_server.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/async_inference/robot_client.py
```

2026-06-11 相关更新：

```text
PolicyServerConfig.async_rtc.latency_quantile 新增为可调参数，默认 0.95。
run_nero_infer_server.sh 当前显式使用 --async_rtc.latency_quantile=0.9。
policy_server._rtc_inference_delay() 从历史最大延迟改为 rolling quantile。
这样首包慢推理不会长期把 RTC inference_delay 锁在 26 step。
```

RTC 官方实现相关代码：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/policies/rtc/configuration_rtc.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/policies/rtc/modeling_rtc.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/policies/rtc/debug_tracker.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/src/lerobot/policies/rtc/debug_visualizer.py
```

### 9.6 近期关键测试

异步推理、RTC trim、debug dump、client 安全执行器相关测试：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/tests/async_inference/test_policy_server.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/tests/async_inference/test_robot_client.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/tests/scripts/test_nero_async_client.py
/home/chenglong/workplace/nero_teleop_ws/lerobot/tests/scripts/test_nero_inference_trace.py
```

常用验证命令：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
conda activate lerobot

python -m pytest tests/async_inference/test_policy_server.py tests/scripts/test_nero_async_client.py -q
python -m py_compile src/lerobot/async_inference/configs.py src/lerobot/async_inference/helpers.py src/lerobot/async_inference/policy_server.py src/lerobot/async_inference/robot_client.py src/lerobot_robot_nero/async_client.py
bash -n scripts/run_nero_infer_server.sh
bash -n scripts/run_nero_infer_client.sh
```

### 9.7 日志目录

server/client 日志目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/logs/
```

常见日志文件名：

```text
policy_server_*.log
robot_client_*.log
```

如果推理卡住，优先检查最新的 `policy_server_*.log` 是否卡在 `huggingface.co` 网络连接；server 必须带：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### 9.8 2026-06-15 之后新增 Nero 文档

Bus table 新模型推理专用文档：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/docs/source/nero/bus_table_inference.md
```

该文档记录 `bryce301best/bus_table` 下载后的新数据模型推理命令，包括：

```text
server 命令
client 命令
RTC 参数
录制参数
CAN 映射
task 文本
```

cuRobo IK 与 EE pose 回放相关文档：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/docs/source/nero/curobo_ik_replay_summary.md
/home/chenglong/workplace/nero_teleop_ws/lerobot/docs/source/nero/curobo_ik_work_log.md
```

其中 `curobo_ik_replay_summary.md` 是总结，`curobo_ik_work_log.md` 是更完整的工作日志。

### 9.9 EE / flange pose 数据集

`pickplace_flange_pose_004` 已确认是末端 flange 位姿数据集，不是关节角数据集：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/pickplace_flange_pose_004
```

数据集摘要：

```text
fps: 30
total_episodes: 50
total_frames: 32456
robot_type: nero_dual
task: pickplace
```

`action` 和 `observation.state` 都是 14 维：

```text
right_flange_x
right_flange_y
right_flange_z
right_flange_roll
right_flange_pitch
right_flange_yaw
left_flange_x
left_flange_y
left_flange_z
left_flange_roll
left_flange_pitch
left_flange_yaw
right_gripper_width
left_gripper_width
```

含义：

```text
x/y/z: 米
roll/pitch/yaw: 弧度
gripper_width: 米
```

结论：

```text
pickplace_flange_pose_004 使用 EE/flange pose action space。
不是 right_nero_joint_1 ... right_nero_joint_7 这种 joint action space。
```

### 9.10 EE pose / cuRobo IK 回放脚本

主要 EE pose 回放脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_nero_dual_ee_pose.py
```

该脚本支持：

```text
--ik-backend=sdk|curobo
--curobo-robot=nero_custom.yml
--episode=<idx>|all
--interpolate-first-target
--joint-target-tolerance-rad
--joint-wait-timeout-s
--joint-timeout-error-rad
--profile-csv=<path>
```

Bus table EE pose + cuRobo IK 回放入口：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay.sh
```

Pickplace flange pose + cuRobo IK 回放入口：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay_pickplace_flange_pose_001.sh
```

`pickplace_flange_pose_001` 回放脚本当前使用：

```text
--episode=all
--ik-backend=curobo
--curobo-robot=nero_custom.yml
--right-channel=nero_right
--left-channel=nero_left
--speed-percent=20
--fps=30
```

曾验证 `pickplace_flange_pose_001` 包含：

```text
episode 0: 881 frames
episode 1: 990 frames
episode 2: 1025 frames
episode 3: 990 frames
episode 4: 934 frames
```

Nero joint 数据转 EE pose 相关脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/convert_nero_joint_dataset_to_ee_rotvec.py
```

cuRobo 配置相关路径：

```text
/home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/configs/robot/nero_custom.yml
/home/chenglong/miniconda3/envs/lerobot/lib/python3.12/site-packages/curobo/content/configs/robot/nero_custom.yml
```

注意：运行时通常读取 conda env site-packages 里的 cuRobo content，因此修改 `nero_custom.yml` 时要确认两份配置同步。

cuRobo 适配中确认的关键点：

```text
数据集 EE pose 对应 link7。
不是 gripper_tcp。
gripper_tcp 曾导致约 13.5 cm 的整体偏移。
```

因此 `nero_custom.yml` 中 tool frame 应保持：

```yaml
tool_frames:
- link7
```

离线 profile 曾确认 bus_table EE pose 的 cuRobo IK 计算本身不慢：

```text
combined IK median: about 0.0043 s
IK failures: 0
```

如果实机回放后半段慢或抖，优先看 `profile-csv` 里的 `right_wait_s` / `left_wait_s`，很多情况下瓶颈是等待机械臂反馈到目标附近，不是 IK 计算慢。

### 9.11 fold_towel_ee 模型下载目录

从 Hugging Face repo 下载的 fold towel EE 模型：

```text
repo: bryce301best/fold_towel_ee
local dir: /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best
```

当前实际模型子目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best/pi05_fold_towel_ee_local_se3_rel_8gpu/checkpoints
```

2026-06-22 检查到的完整 checkpoint：

```text
006000
010000
014000
016000
018000
```

每个完整 checkpoint 的 `model.safetensors` 大小：

```text
9354050752 bytes
```

仍缺 `model.safetensors` 或未完全补全的 checkpoint：

```text
008000
012000
020000
```

如果继续下载，使用 `hf`，不要再用已弃用的 `huggingface-cli`。

镜像登录：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_ENDPOINT=https://hf-mirror.com hf auth login
```

登出后重新登录：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_ENDPOINT=https://hf-mirror.com hf auth logout && \
HF_ENDPOINT=https://hf-mirror.com hf auth login
```

检查登录：

```bash
HF_ENDPOINT=https://hf-mirror.com hf auth whoami
```

只补 `020000` 的推荐下载命令：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_ENDPOINT=https://hf-mirror.com hf download bryce301best/fold_towel_ee \
  --repo-type model \
  --include "pi05_fold_towel_ee_local_se3_rel_8gpu/checkpoints/020000/**" \
  --local-dir /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best \
  --max-workers 2
```

补全整个 repo 的命令：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_ENDPOINT=https://hf-mirror.com hf download bryce301best/fold_towel_ee \
  --repo-type model \
  --local-dir /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best \
  --max-workers 2
```

下载注意：

```text
不要加 --force-download。
保持相同 --local-dir 时，会复用已有完整文件，并继续补缺失文件。
如果整库下载，会同时下载多个大 checkpoint，容易占满磁盘。
```

### 9.12 pickplace_ee 快速下载

`bryce301best/pickplace_ee` 中单个 `model.safetensors` 约 9.35 GB。
如果 `hf download` 很慢，优先使用 `aria2c` 从镜像站直接下载大文件。
该方式支持断点续传和多连接下载，现场验证速度明显更快。

当前推荐下载到：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee
```

按顺序下载 `016000` 和 `014000` 两个 checkpoint：

```bash
source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

BASE="https://hf-mirror.com/bryce301best/pickplace_ee/resolve/main"
DEST="/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee"

for CKPT in 016000 014000; do
  DIR="$DEST/checkpoints/$CKPT/pretrained_model"
  mkdir -p "$DIR"

  aria2c -c -x 16 -s 16 -k 1M \
    -d "$DIR" \
    -o model.safetensors \
    "$BASE/checkpoints/$CKPT/pretrained_model/model.safetensors"

  for F in \
    config.json \
    policy_postprocessor.json \
    policy_postprocessor_step_0_unnormalizer_processor.safetensors \
    policy_preprocessor.json \
    policy_preprocessor_step_3_normalizer_processor.safetensors \
    train_config.json
  do
    aria2c -c -x 4 -s 4 \
      -d "$DIR" \
      -o "$F" \
      "$BASE/checkpoints/$CKPT/pretrained_model/$F"
  done
done
```

下载完成后的 checkpoint 路径：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee/checkpoints/016000/pretrained_model
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pickplace_ee/checkpoints/014000/pretrained_model
```

如果模型需要 token，先设置：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

然后给所有 `aria2c` 命令加：

```bash
--header="Authorization: Bearer $HF_TOKEN"
```

注意：

```text
不要整库下载，避免把其它 checkpoint 一起拉下来占满磁盘。
`aria2c -c` 可以从未完成文件继续下载；如果重新执行同一命令，会复用已有部分。
```

### 9.13 PI05 的 task 文本和 tasks.parquet

PI05 checkpoint 的 `policy_preprocessor.json` 中，语言输入字段为：

```text
task_key: "task"
```

已检查过的 checkpoint 包括：

```text
fold towel 006000
bus table 002000
bus table 007000
```

它们都包含：

```text
pi05_prepare_state_tokenizer_processor_step
tokenizer_processor
tokenizer_name: google/paligemma-3b-pt-224
task_key: task
```

推理命令中的：

```bash
--task="..."
```

会进入每帧 observation，然后进入 PI05 prompt。大致流程：

```text
--task
-> raw_observation["task"]
-> server preprocess
-> PI05 task tokenizer
-> language tokens
-> policy
```

LeRobot 数据集中的 `meta/tasks.parquet` 用来把每帧 `task_index` 映射成自然语言 task。

例如：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_02/meta/tasks.parquet
```

内容是：

```text
task        task_index
fold_towel  0
```

PI05 预处理会把下划线替换为空格，因此训练时实际 prompt 更接近：

```text
Task: fold towel, State: ...;
Action:
```

推理时推荐让 `--task` 尽量贴近训练数据：

```bash
--task="fold towel"
```

或：

```bash
--task="fold_towel"
```

二者经过 PI05 清洗后都接近 `fold towel`。不要随意改成差异很大的描述，除非训练数据里就是这种任务文本。

### 9.14 磁盘占用排查记录

2026-06-22 曾因整库下载 `fold_towel_ee` 导致根分区空间不足。排查结论：

```text
/home 是主要占用来源。
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train 是用户当前训练模型主要占用。
/home/chenglong/workplace/nero_teleop_ws/lerobot/logs 曾约 6.9G。
/home/qt/Downloads/lerobot/outputs 曾约 193G，但属于其它用户目录，清理前需要确认。
```

fold_towel_ee 下载目录中 `.cache/huggingface/download` 可能保存未完成的 `.incomplete` 文件。若确认只使用已有完整 checkpoint，可以清理该下载缓存释放空间：

```bash
rm -rf /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best/.cache
```

清理前先检查：

```bash
du -sh /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best
find /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best -type f -name "*.incomplete" -printf "%s %p\n" | sort -n | tail -20
df -h /
```

## 10. 常见问题

### 主从同步位置明显不对

优先检查 SO101 leader 校准文件是否正确：

```text
/home/chenglong/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_8dof_wzh.json
/home/chenglong/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_8dof_pair1.json
```

### 找不到 SO101 串口

检查 `/dev/serial/by-id/` 是否存在两个 SO101 设备，并确认当前命令中的 port 与实际设备一致：

```bash
ls -l /dev/serial/by-id/
```

### 无法打开串口

先执行临时授权脚本。如果希望长期生效，把当前用户加入 `dialout` 并重新登录。

### 找不到相机或相机顺序错误

运行：

```bash
lerobot-find-cameras
```

确认三路 RealSense serial 与 `--robot.cameras` 中的 `serial_number_or_name` 一致。

### 控制循环变慢

如果只是偶发 warning，先通过回放确认轨迹质量。如果频繁出现并影响操作手感，优先降低相机分辨率或采样频率。

### 回放或推理前的安全检查

1. 确认双臂周围没有人员和障碍物。
2. 确认当前 episode 或 ready pose 是安全姿态。
3. 确认急停或断电手段可用。

### CAN 口未 up 或状态异常

常用启动命令，Nero 两条臂 CAN bitrate 使用 `1000000`：

```bash
sudo ip link set nero_right up type can bitrate 1000000
sudo ip link set nero_left up type can bitrate 1000000
```

如果接口已经配置过 bitrate，只是没有起来：

```bash
sudo ip link set nero_right up
sudo ip link set nero_left up
```

如果提示 `busy`，或者 bitrate 参数没有生效，先 down 再重新设置：

```bash
sudo ip link set nero_right down
sudo ip link set nero_right type can bitrate 1000000
sudo ip link set nero_right up

sudo ip link set nero_left down
sudo ip link set nero_left type can bitrate 1000000
sudo ip link set nero_left up
```

检查状态：

```bash
ip link show nero_right
ip link show nero_left
```

### Nero 1.20 固件无法 enable

如果两条 Nero 能读到 firmware 和关节角，但 `enable()` 返回 `False`，先看状态是否类似：

```text
firmware: 1.20
ctrl_mode: ETHERNET_CONTROL_MODE
arm_status: JOINT_BRAKE_NOT_RELEASED
enable_status: [False, False, False, False, False, False, False]
```

这表示 CAN 通信是通的，但机械臂控制权还不在 CAN 模式，刹车也未释放。当前 Nero 固件为 `1.20` 时，应使用 `NeroFW.V120` 驱动，并先切到 follower controlled mode，再 enable：

```text
NeroFW.V120
robot.set_follower_mode()
robot.enable()
```

当前 `lerobot_robot_nero` 的 Nero 入口已经默认使用 `V120`，并在 `enable()` 前执行 CAN follower 接管流程：

```text
_set_mode(ctrl_mode=CAN, move_mode=js, mit_mode=off, enable_can_push=on)
set_follower_mode()
set_motion_mode("js")
clear_joint_error(255)
enable()
```

如果推理日志或 trace 的 `meta.json` 里仍出现 `firmware_version: V112`，说明运行的不是当前代码或命令里显式覆盖了旧固件版本，需要先修正后再排查 IK / policy 问题。

双臂 enable 的确认脚本：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
source /home/chenglong/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=/home/chenglong/workplace/nero_teleop_ws/third_party/pyAgxArm:$PYTHONPATH python - <<'PY'
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


def enable_arm(label: str, channel: str) -> None:
    print(f"=== {label} ===", flush=True)
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    try:
        robot.connect()
        time.sleep(0.3)

        print("firmware:", robot.get_firmware(), flush=True)
        print("before_status:", getattr(robot.get_arm_status(), "msg", robot.get_arm_status()), flush=True)
        print("before_enable:", robot.get_joints_enable_status_list(), flush=True)

        robot.set_follower_mode()
        time.sleep(0.3)

        ok = robot.enable(timeout=2.0)
        time.sleep(0.3)

        status = robot.get_arm_status()
        enabled = robot.get_joints_enable_status_list()
        print("enable_return:", ok, flush=True)
        print("after_status:", getattr(status, "msg", status), flush=True)
        print("after_enable:", enabled, flush=True)
        print("all_enabled:", bool(enabled and all(enabled)), flush=True)
    finally:
        robot.disconnect()


enable_arm("right/nero_right", "nero_right")
enable_arm("left/nero_left", "nero_left")
PY
```

成功状态应为：

```text
ctrl_mode: CAN_CTRL
arm_status: NORMAL
enable_status: [True, True, True, True, True, True, True]
```

如果仍失败，优先检查：

```text
1. nero_right/nero_left 是否 UP，bitrate 是否为 1000000。
2. 是否误用 NeroFW.V112。1.20 固件应使用 NeroFW.V120。
3. 是否有其它 Nero client 或控制程序占用 CAN。
4. 示教器/Web/上位机是否保持 Ethernet 控制权。
```

## 16. 固定 Nero 双臂 USB-CAN 命名

### 16.1 问题原因

两只 Nero 使用的 USB-CAN 适配器是同型号 `candleLight USB to CAN adapter`。Linux 默认会按 USB 枚举顺序分配 `can0`、`can1`，而这个顺序在重启、重插、USB hub 初始化顺序变化时可能改变。

如果代码直接写 `right=can1`、`left=can0`，重启后 `can0/can1` 对应到另一只适配器时，左右臂就会反。

当前已经改为按适配器序列号固定语义接口名：

```text
left Nero USB-CAN:
  stable interface: nero_left
  serial: 003100414148570C20343133

right Nero USB-CAN:
  stable interface: nero_right
  serial: 003400464148570A20343133
```

之后 Nero 相关脚本都应使用：

```text
left arm CAN: nero_left
right arm CAN: nero_right
```

不要在新的 Nero 双臂脚本里重新写死 `can0` / `can1`。

### 16.2 安装固定命名规则

仓库内提供安装脚本：

```bash
cd /home/chenglong/workplace/nero_teleop_ws
sudo bash lerobot/scripts/install_nero_can_names.sh --apply-now
```

脚本会写入：

```text
/etc/systemd/network/10-nero-left-can.link
/etc/systemd/network/10-nero-right-can.link
```

规则按 `ID_SERIAL_SHORT` 匹配 USB-CAN，而不是按当前 `can0/can1` 名字匹配。因此重启后仍然稳定。

`--apply-now` 会在不重启的情况下尝试立即执行：

```text
serial 003100414148570C20343133 -> nero_left
serial 003400464148570A20343133 -> nero_right
bitrate 1000000
interface up
```

如果当时有 Nero client、replay 或其它程序占用 CAN，先停止这些进程再执行安装脚本。

### 16.3 验证命令

检查接口是否已经按语义名出现：

```bash
ip -details link show type can
```

期望看到：

```text
nero_left:  UP, ERROR-ACTIVE, bitrate 1000000
nero_right: UP, ERROR-ACTIVE, bitrate 1000000
```

检查序列号和规则文件是否匹配：

```bash
udevadm info -q property -p /sys/class/net/nero_left | grep -E 'INTERFACE|ID_SERIAL_SHORT|ID_NET_LINK_FILE|ID_NET_NAME'
udevadm info -q property -p /sys/class/net/nero_right | grep -E 'INTERFACE|ID_SERIAL_SHORT|ID_NET_LINK_FILE|ID_NET_NAME'
```

期望结果：

```text
INTERFACE=nero_left
ID_SERIAL_SHORT=003100414148570C20343133
ID_NET_LINK_FILE=/etc/systemd/network/10-nero-left-can.link
ID_NET_NAME=nero_left

INTERFACE=nero_right
ID_SERIAL_SHORT=003400464148570A20343133
ID_NET_LINK_FILE=/etc/systemd/network/10-nero-right-can.link
ID_NET_NAME=nero_right
```

### 16.4 如果重启后没有出现 nero_left / nero_right

先检查两只 USB-CAN 是否被系统识别：

```bash
ip -details link show type can
```

如果只看到 `can0/can1`，检查 `.link` 规则是否存在：

```bash
sudo sed -n '1,80p' /etc/systemd/network/10-nero-left-can.link
sudo sed -n '1,80p' /etc/systemd/network/10-nero-right-can.link
```

重新加载规则并重插 USB-CAN，或直接重启：

```bash
sudo udevadm control --reload
```

如果只是接口没有 up，可手动设置：

```bash
sudo ip link set nero_left down || true
sudo ip link set nero_left type can bitrate 1000000
sudo ip link set nero_left up

sudo ip link set nero_right down || true
sudo ip link set nero_right type can bitrate 1000000
sudo ip link set nero_right up
```

### 16.5 更换 USB-CAN 适配器时

如果更换了任意一只 USB-CAN，序列号会变化，需要重新更新规则。先查看新序列号：

```bash
for iface in $(ls /sys/class/net); do
  udevadm info -q property -p "/sys/class/net/${iface}" 2>/dev/null \
    | grep -E '^(INTERFACE|ID_MODEL|ID_SERIAL_SHORT|ID_PATH)='
done
```

确认哪只物理连接到左臂、哪只物理连接到右臂后，更新：

```text
lerobot/scripts/install_nero_can_names.sh
```

中的：

```text
LEFT_SERIAL=...
RIGHT_SERIAL=...
```

然后重新执行：

```bash
sudo bash lerobot/scripts/install_nero_can_names.sh --apply-now
```

### 16.6 当前代码约定

当前 `nero_dual` 默认配置和 demo 脚本都使用稳定接口名：

```text
robot.right.connection.channel = nero_right
robot.left.connection.channel  = nero_left
```

相关测试：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
conda run -n lerobot pytest \
  tests/robots/test_nero_mapping.py \
  tests/scripts/test_nero_can_bindings.py \
  tests/scripts/test_install_nero_can_names_script.py \
  -q
```
