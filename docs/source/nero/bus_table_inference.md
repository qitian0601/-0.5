# Nero Bus Table 推理命令

本文档记录 `bryce301best/bus_table` 新数据训练模型的 Nero 双臂实机推理命令。

执行推理前，必须确认双臂周围安全、急停或断电手段可用，并确认当前 ready pose 对任务环境是安全的。

## 1. 当前模型

当前使用本地 checkpoint：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_bus_table_new_data_bryce301best/checkpoints/007000/pretrained_model
```

注意：本地目录里目前没有 `006000` checkpoint。已经确认存在的是 `007000`。

## 2. 启动顺序

需要两个终端：

1. 先启动 server。
2. 等 server 完成模型加载并监听 `8080`。
3. 再启动 client。

## 3. Server 端命令

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --obs_similarity_atol=0.09 \
  --async_rtc.enabled=true \
  --async_rtc.latency_quantile=0.9 \
  --async_rtc.debug_dump.enabled=true
```

关键参数：

```text
--fps=30: server 按 30Hz 推理/调度。
--obs_similarity_atol=0.09: 相邻 observation 相似度过滤阈值。
--async_rtc.enabled=true: 开启 RTC。
--async_rtc.latency_quantile=0.9: RTC 延迟估计使用 rolling p90。
--async_rtc.debug_dump.enabled=true: 将 RTC debug 数据保存到本次录制目录。
```

## 4. Client 端命令

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot && \
source /home/chenglong/miniconda3/etc/profile.d/conda.sh && \
conda activate lerobot && \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m lerobot_robot_nero.async_client \
  --robot.type=nero_dual \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --policy_path=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_bus_table_new_data_bryce301best/checkpoints/008000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --task="Use the right arm to place the yellow cube into the black box, then use the left arm to place the green cube into the black box." \
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

关键参数：

```text
--robot.right.connection.channel=nero_right: 当前右臂使用稳定 CAN 名 nero_right。
--robot.left.connection.channel=nero_left: 当前左臂使用稳定 CAN 名 nero_left。
--robot.*.connection.reset_on_connect=false: 连接时不 reset。
--trace.enabled=true: 开启推理录制。
--actions_per_chunk=50: 每个 action chunk 50 step。
--chunk_size_threshold=0.8: action queue 低于阈值时请求新 chunk。
--aggregate_fn_name=average: action chunk 聚合方式使用 average。
```

## 5. 输出目录

推理录制会写入：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records
```

每次运行会生成一个按时间命名的子目录。开启 RTC debug 后，`rtc_debug.jsonl` 会和本次录制数据放在同一个目录里。

## 6. 常见注意点

命令换行时，反斜杠 `\` 必须是该行最后一个字符，后面不能有空格。

如果 server 端卡在 Hugging Face 网络连接，确认命令里带有：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

如果 client 提示连接失败，先确认 server 已经成功启动，并且端口是：

```text
127.0.0.1:8080
```
