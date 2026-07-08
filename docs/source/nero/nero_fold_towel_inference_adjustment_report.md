# Nero fold_towel 推理调试与调整报告

本文记录 fold_towel 任务在 Nero 双臂上的部署推理调试过程。内容来自本次实际排查，重点保留后续复现和继续调参需要的信息。

执行推理、回放或任何会驱动 Nero 的命令前，必须确认双臂周围安全，急停或断电手段可用。

## 1. 当前目标

当前任务是使用训练好的 `pi05_fold_towel_rel_8gpu` checkpoint 在双 Nero 上做异步推理。

当前推荐配置：

```text
policy checkpoint: 012000/pretrained_model
task: fold_towel
policy/client fps: 30
policy_server.obs_similarity_atol: 0.1
async_rtc.enabled: true
async_rtc.latency_quantile: 0.9
async_rtc.debug_dump.enabled: true
actions_per_chunk: 50
chunk_size_threshold: 0.8
aggregate_fn_name: average
high_rate_control: true
high_rate_dt_s: 0.005556   # 约 180 Hz
robot.*.connection.reset_on_connect: false
right Nero CAN: nero_right
left Nero CAN: nero_left
trace.enabled: true
```

2026-06-22 之后另有一条新版 EE-local-SE3 推理路线，用于
`bryce301best/fold_towel_ee` 下载得到的 `pi05_fold_towel_ee_local_se3_rel_8gpu`
checkpoint。该路线和上面的 joint-action 路线并存，不替换旧推理脚本。
EE 路线的默认实机 CAN 映射为：

```text
right Nero CAN: nero_right
left Nero CAN: nero_left
```

## 2. 模型与 checkpoint

### 2.1 模型目录

本机 checkpoint 路径：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model/checkpoints
```

使用过的 checkpoint：

```text
002000
004000
006000
008000
010000
012000
```

推理当前优先使用：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model/checkpoints/012000/pretrained_model
```

### 2.2 Hugging Face 镜像下载

`huggingface-cli` 已废弃，应使用 `hf`。如果从镜像站下载：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download bryce301best/fold_towel \
  --local-dir /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_rel_8gpu/pretrained_model
```

如果只补某个大模型文件，需要注意 `--local-dir` 会影响最终目录结构。不要把 `checkpoints/010000/...` 下载到已经是 `pretrained_model` 的子目录里，否则路径可能多套一层。

### 2.3 完整性校验

每个 checkpoint 的 `pretrained_model` 应有 7 个文件：

```text
config.json
train_config.json
policy_preprocessor.json
policy_postprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor_step_0_unnormalizer_processor.safetensors
model.safetensors
```

本次核对过的关键指纹：

```text
model.safetensors size: 9354050752 bytes
model.safetensors header_len: 139384
model.safetensors tensor_count: 813
```

## 3. RTC、async queue 与 chunk 参数

### 3.1 RTC 配置状态

原始 6 个 checkpoint 的 `config.json` 里：

```text
rtc_config = null
```

之后手动把 `010000` 和 `012000` 改成启用 RTC 默认参数：

```json
"rtc_config": {
  "enabled": true,
  "prefix_attention_schedule": "LINEAR",
  "max_guidance_weight": 10.0,
  "execution_horizon": 10,
  "debug": false,
  "debug_maxlen": 100
}
```

注意：这代表模型配置会初始化 RTC 相关处理器。但当时检查到当前 `policy_server.py` 仍是直接调用 `predict_action_chunk(observation)`，未显式传入 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 等参数。因此：

```text
async queue 一定在发挥作用；
RTC 配置已启用，但真实 leftover-guidance 是否完整发挥，需要继续看 server 侧实现。
```

### 3.2 async queue、chunk 参数和 obs 相似过滤

`policy_server.obs_similarity_atol` 是服务端 observation 去重阈值。服务端会从 observation 中提取 state 向量，计算当前 observation 和上一次已处理 observation 的 state L2 距离：

```text
norm(current_state - previous_state) < obs_similarity_atol
```

如果成立，且该 observation 不是 `must_go`，服务端会跳过这帧，不调用 `predict_action_chunk()`。因此它影响的是“是否生成新的 action chunk”，不是客户端 action queue 的消费速度。

`actions_per_chunk` 是每次服务端生成多少个 policy action。

`chunk_size_threshold` 是客户端队列剩余比例到达阈值时，提前发下一次 observation 给 server。阈值越高，请求越早，越不容易吃空，但会增加 server 压力，也会导致更多 overlap。

本次调参结果：

```text
50 / 0.5: 停顿明显，曾出现约 0.5s 级别 gap。
50 / 0.8: 明显改善。
50 / 0.7: 前段可用，但后段失败 trace 里 gap 变大，不推荐。
40 / 0.8: 当时最干净，最大 queue gap 约 234ms，高频平均约 172.5Hz。
40 / 0.9: 当前准备使用，目标是更早请求下一 chunk，尽量减少等待。
```

`queue gap` 的含义：policy/async 层一段时间没有新的 30Hz target 可消费。高频 executor 仍会继续追/保持最后 target，但机械臂观感上可能出现短暂停顿。

### 3.3 2026-06-09 action chunk 与 `obs_similarity_atol` 更新

今天的核心问题不是单纯 chunk 太短，而是服务端 observation 相似过滤会把大量低速、细微变化的 observation 判为“too similar”，导致很多客户端请求没有触发新的 `predict_action_chunk()`。这会让客户端即使提高 `chunk_size_threshold`，也可能拿不到足够新的 chunk。

本次代码层面做了两处更新：

```text
lerobot/src/lerobot/async_inference/configs.py
- PolicyServerConfig 新增 obs_similarity_atol，默认 1.0。
- 增加非负校验，并写入 to_dict()，方便日志确认实际启动值。

lerobot/src/lerobot/async_inference/policy_server.py
- _obs_sanity_checks() 调用 observations_similar() 时使用 self.config.obs_similarity_atol。
- 旧行为等价于固定 atol=1.0，无法通过 CLI 调整。
```

对应测试：

```text
lerobot/tests/async_inference/test_policy_server.py
- 新增 test_maybe_enqueue_observation_uses_configured_similarity_atol。
- 验证更严格阈值会让小但真实的 state 变化进入 observation queue。
```

2026-06-09 实际跑过的 server 配置记录：

```text
早期默认行为: 等价 obs_similarity_atol=1.0，日志中大量 Observation too similar。
15:19: obs_similarity_atol=0.05。
15:34: obs_similarity_atol=0.1。
16:53: obs_similarity_atol=0.2。
17:27: obs_similarity_atol=0.15。
```

经验结论：

```text
atol 越小，越容易把细微 state 变化送入 policy，更新更积极，但 server 压力更大。
atol 越大，越容易跳过相似 observation，server 压力更小，但 action chunk 更新可能偏慢。
0.15 是 2026-06-09 最后一轮使用值，配合 actions_per_chunk=50、chunk_size_threshold=0.9、reset_on_connect=false。
0.1 仍看到大量 too similar；0.2 更保守；0.05 最激进，适合排查过滤是否过强，不建议直接作为长期默认。
```

2026-06-09 action chunk 侧也从早前 `40 / 0.9`、`40 / 1.0` 继续试到 `50 / 0.9`。当时推荐临时组合是：

```text
policy_server.obs_similarity_atol=0.15
actions_per_chunk=50
chunk_size_threshold=0.9
fps=30
high_rate_dt_s=0.005556
reset_on_connect=false
```

注意：`50 / 0.9` 不代表最终最优，只是当时和 `obs_similarity_atol`、`reset_on_connect=false` 一起收敛后的可复现配置。后续比较时必须同时记录 server 的 `obs_similarity_atol`，否则单看 `actions_per_chunk/chunk_size_threshold` 会误判。

### 3.4 2026-06-11 RTC 调度、debug dump 与延迟估计更新

6 月 11 日围绕 async RTC 做了三类更新：server 侧传入 RTC 需要的 `prev_chunk_left_over` 和 `inference_delay`，推理录制目录里保存 `rtc_debug.jsonl`，以及把 `inference_delay` 的估计方式从历史最大延迟改为 rolling quantile。

当前 async RTC server 行为：

```text
policy_server 会把上一 chunk 还未执行的 leftover 传给 predict_action_chunk()。
policy_server 会根据推理延迟计算 inference_delay，并传给 RTC。
新 chunk 生成后会 trim 掉已经过期的 stale action，再放入 client queue。
开启 --async_rtc.debug_dump.enabled=true 后，同次录制目录写入 rtc_debug.jsonl。
```

`rtc_debug.jsonl` 当前保存轻量统计量，不保存完整 tensor。重点字段：

```text
weights: prefix attention 权重窗口
correction: RTC 给 denoise 的修正量
err: 新旧 chunk 一致性误差
guidance_weight: 实际约束强度
inference_delay: 传给 RTC 的延迟 step
execution_horizon: RTC 平滑窗口
x_t / v_t / x1_t: denoising 中间状态统计
```

对 `/outputs/nero_inference_records/20260611_114417` 的分析结论：

```text
server 推理常态约 0.18-0.23s，即约 6-7 step。
首包曾出现约 0.85s，总延迟换算为 26 step。
旧实现使用 LatencyTracker.max()，导致 inference_delay 长期被锁在 26 step。
实际 trim delay 多数仍是 6-7 step，因此问题是传给 RTC 的延迟估计过保守，不是系统一直慢 26 step。
```

这次把 `PolicyServerConfig.async_rtc.latency_quantile` 做成可调参数：

```text
默认值: 0.95
当前 server 脚本: --async_rtc.latency_quantile=0.9
```

含义是使用最近延迟窗口的 p90 来估计 `inference_delay`。这样首包慢推理不会永久污染整场推理，后续正常情况下 `inference_delay` 应回到约 6-8 step。

当前不建议优先调整 RTC 静态强度参数。`20260611_114417` 中 RTC 修正量相对 denoise 速度并不大：

```text
correction_norm / v_t_norm:
p50 约 2.3%
p90 约 3.4%
p99 约 5.9%
max 约 11%
```

因此当前判断是先修正 `inference_delay` 估计方式，再观察顿挫和目标迟滞是否改善；暂时保持：

```text
rtc_config.execution_horizon: 10
rtc_config.max_guidance_weight: 10.0
rtc_config.prefix_attention_schedule: LINEAR
```

### 3.5 2026-06-11 latest trace 结论

重点分析过的录制目录：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_101330
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_110944
/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records/20260611_114417
```

`20260611_114417` 使用 `aggregate_fn_name=average` 后，严重吃空现象不明显。主要观察：

```text
queue empty / 大 gap 占比很低，约 0.35% 级别。
high-rate executor 多数能维持约 180Hz，偶发 overrun 可以接受。
policy inference 常态约 0.2s，主要异常来自首包慢推理。
机械臂仍有顿挫感时，更像是 target 更新和 RTC 延迟估计偏保守导致的迟滞，而不是 action queue 长时间吃空。
```

调参顺序建议：

```text
1. 先使用 async_rtc.latency_quantile=0.9 复测。
2. 保持 execution_horizon=10、max_guidance_weight=10.0 不动。
3. 如果仍有明显迟滞，再考虑 execution_horizon 从 10 降到 8。
4. 如果修正过强或动作过粘，再考虑 max_guidance_weight 从 10 降到 6-8。
```

## 4. 高频执行器调整

本次对 Nero async/high-rate 链路做过修复：

```text
默认 high_rate_dt_s 调整到 0.005556，目标约 180Hz。
新增 high_rate_interpolation_steps=6，把 30Hz target 插值到 180Hz tick。
高频夹爪发送节流：首帧、每 6 tick 或变化超过阈值才发送。
高频 executor 记录真实 elapsed_s、sleep_s、overrun_s、actual_hz。
底层 send_action 支持 send_gripper=False 和 read_feedback=False。
高频路径不再每 tick 读反馈，也不每 tick 发夹爪。
```

重要结论：

```text
把 high_rate_dt_s 改成 1/180 不等于真实一定能稳定 180Hz。
如果每 tick 的物理发送耗时超过 5.56ms，会出现 overrun。
```

出现如下日志时：

```text
Nero high-rate executor overrun: elapsed=6.6340ms target=5.5560ms overrun=1.0780ms
```

含义是该 tick 的实际工作耗时超过 180Hz 的周期预算。偶发可以接受，频繁出现会降低平滑性。

### 4.1 启动下沉与 reset

启动 Nero client 时，`Nero is at the fixed ready pose` 之前会先连接双臂、接管 CAN、切换模式、使能，然后才同步到 fixed ready pose。

实际排查发现：启动阶段机械臂突然向下沉，主要由连接流程里的 SDK `reset()` 触发。给双臂加上如下参数后，本次实测启动不再出现明显下降：

```bash
--robot.right.connection.reset_on_connect=false \
--robot.left.connection.reset_on_connect=false
```

因此当前推理建议关闭 connect 阶段的 `reset()`。保留 `clear_joint_error(255)`、`enable()` 和 fixed ready pose 同步。若后续确实经历过急停或控制器状态异常，再考虑单独执行 reset，而不是每次推理启动都 reset。

### 4.2 本次实际修改过的脚本/源码

本次不只是调整命令，也改过 Nero 推理链路里的脚本/源码。后续如果同步代码或回滚，需要优先关注这些文件。

#### `lerobot/src/lerobot_robot_nero/async_client.py`

主要改动：

```text
DEFAULT_HIGH_RATE_HZ = 180.0
DEFAULT_HIGH_RATE_DT_S = 1.0 / DEFAULT_HIGH_RATE_HZ
```

`NeroAsyncSafetyConfig` 中新增/调整：

```text
high_rate_dt_s: 默认 1/180
high_rate_interpolation_steps: 默认 6
high_rate_gripper_period: 默认 6
high_rate_gripper_epsilon_m: 默认 1e-4
high_rate_overrun_log_every: 默认 100
```

行为变化：

```text
1. 高频 executor 默认目标从原来的 control_dt_s 回退逻辑，改为明确 180Hz。
2. 30Hz policy target 不再只是被高频线程追最新值，而是在高频线程内按 6 个 tick 做线性插值。
3. 高频夹爪发送被节流，避免每个 180Hz tick 都发夹爪命令。
4. 高频路径调用底层 robot.send_action 时传入 read_feedback=False，避免每 tick 读关节反馈。
5. trace 中增加 executor 真实耗时、sleep、overrun、actual_hz 等字段。
6. overrun 会按 high_rate_overrun_log_every 节流打 warning。
```

相关事件记录：

```text
policy_raw_action
policy_limited_target
high_rate_target
executor_step
executor_timing
```

#### `lerobot/src/lerobot_robot_nero/robot_nero_dual.py`

主要改动是底层发送接口增加两个可选参数：

```python
send_action(
    action,
    *,
    send_gripper: bool = True,
    read_feedback: bool = True,
)
```

行为变化：

```text
send_gripper=False 时，不调用 move_gripper_m。
read_feedback=False 时，不调用 read_joints_once。
双臂 send_action 会把 send_gripper/read_feedback 继续传给左右单臂。
trace 会记录 send_gripper 和 read_feedback，便于确认高频路径是否真的跳过夹爪和反馈读取。
```

这项改动的目的：把高频 180Hz tick 的物理工作量压下来，只保留最关键的关节目标下发。夹爪和反馈读取放低频或按需执行。

另外，连接流程新增 `reset_on_connect` 配置开关：

```text
NeroConnectionConfig.reset_on_connect: 默认 true
```

默认保持旧行为；推理时建议显式传 `false`，跳过每次连接时的 `robot.reset()`，避免启动阶段机械臂下沉。

#### 对应测试

相关测试文件：

```text
lerobot/tests/scripts/test_nero_async_client.py
lerobot/tests/robots/test_nero_dual_robot.py
```

覆盖点：

```text
默认 high_rate_dt_s 是 1/180。
high_rate_interpolation_steps=6 时 target 会被插值。
高频 executor 会按策略跳过部分夹爪发送。
高频路径底层 send_action 使用 read_feedback=False。
trace 中能看到 executor_step、overrun_s、actual_hz。
robot_nero_dual.send_action 支持 send_gripper/read_feedback 参数。
```

当时验证结果：

```text
聚焦测试通过。
相关行为脚本通过。
pytest 在补齐环境后曾跑通过 18 个相关测试。
```

## 5. 硬件绑定

### 5.1 Nero CAN

```text
right Nero: nero_right
left Nero:  nero_left
CAN bitrate: 1000000
```

Nero 机械臂不需要 USB3。USB-CAN 适配器是 full-speed/USB2 即可，因为 CAN 本身是 1Mbps。

拉起 CAN：

```bash
sudo ip link set can0 down || true
sudo ip link set can1 down || true
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
sudo ip link set can1 up
```

检查：

```bash
ip -details link show can0
ip -details link show can1
```

正常应看到：

```text
UP, LOWER_UP
can state ERROR-ACTIVE
bitrate 1000000
```

### 5.2 RealSense 相机

固定相机绑定：

```text
front:       324422301659  D455  1280x800@30
left_wrist:  244222077114  D435I  640x480@30
right_wrist: 244222070153  D435I  640x480@30
```

三台相机正式推理建议全部 USB3：

```text
front:       5000M 或更高
left_wrist:  5000M 或更高
right_wrist: 5000M 或更高
```

## 6. 推理前检查清单

### 6.1 相机

```bash
lerobot-find-cameras
lsusb -t
```

期望：

```text
front       324422301659  D455   5000M
left_wrist  244222077114  D435I  5000M
right_wrist 244222070153  D435I  5000M
```

### 6.2 Nero 臂

```bash
ip -details link show can0
ip -details link show can1
```

期望：

```text
can0 UP ERROR-ACTIVE bitrate 1000000
can1 UP ERROR-ACTIVE bitrate 1000000
```

SDK 状态应为：

```text
right nero_right enable_status = [True, True, True, True, True, True, True]
left  nero_left enable_status = [True, True, True, True, True, True, True]
```

### 6.3 server

如果之前 server 已经在跑，确认端口和进程：

```bash
ps -eo pid,ppid,stat,cmd | grep -E 'policy_server|async_client' | grep -v grep
```

## 7. 当前推荐推理命令

部署推理分两部分：先启动 policy server，再启动 Nero client。

### 7.1 Policy server

当前推荐直接使用脚本：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_infer_server.sh
```

脚本内关键参数：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --obs_similarity_atol=0.1 \
  --async_rtc.enabled=true \
  --async_rtc.latency_quantile=0.9 \
  --async_rtc.debug_dump.enabled=true
```

### 7.2 Nero client

当前推荐直接使用脚本：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_infer_client.sh
```

脚本保留 ready pose、安全限幅、三相机、高频执行器、trace 录制和键盘退出参数；当前使用 `actions_per_chunk=50`、`chunk_size_threshold=0.8`、`aggregate_fn_name=average`，并显式指定 `right=nero_right`、`left=nero_left`。

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
  --safety.fixed_ready_pose='{"right_nero_joint_1":-0.01864011641129944,"right_nero_joint_2":-1.72533,"right_nero_joint_3":0.005689773361501515,"right_nero_joint_4":1.8416539734118966,"right_nero_joint_5":-0.055553830090979514,"right_nero_joint_6":-0.028972465583105872,"right_nero_joint_7":1.550797,"right_gripper_width":0.02853,"left_nero_joint_1":0.03324852225049198,"left_nero_joint_2":-1.72533,"left_nero_joint_3":-3.490658503988659e-05,"left_nero_joint_4":1.796868824805722,"left_nero_joint_5":-0.017174039839624202,"left_nero_joint_6":-0.06607816548050531,"left_nero_joint_7":1.550797,"left_gripper_width":0.023507}' \
  --safety.takeover_time_s=6.0 \
  --safety.max_policy_step_rad=0.05 \
  --safety.max_gripper_step_m=0.05 \
  --safety.dry_run=false \
  --safety.high_rate_control=true \
  --safety.high_rate_dt_s=0.005556 \
  --safety.max_executor_step_rad=0.005 \
  --safety.max_executor_gripper_step_m=0.004 \
  --robot.right.connection.reset_on_connect=false \
  --robot.left.connection.reset_on_connect=false \
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

退出推理时，`--keyboard_stop=true` 会允许在 Nero client 终端输入 `q` / `s` / `stop` / `e` / `exit` 后回车来停止 client。原始推理退出流程不会自动回到额外 pose，也不会主动 disable 夹爪或双臂。

## 8. 2026-06-22 EE-local-SE3 推理链路修改

本节记录今天针对新版 fold_towel_ee 模型做的推理链路改造。目标是让
PI05 EE checkpoint 可以输出：

```text
right_ee_x/y/z
right_ee_rotvec_x/y/z
right_gripper_width
left_ee_x/y/z
left_ee_rotvec_x/y/z
left_gripper_width
base_or_head_x
base_or_head_y
```

然后通过 hand-eye 和 cuRobo IK 转回 Nero 关节目标，最终仍复用原来的
joint safety / high-rate executor / `move_js` 控制链路。

### 8.1 第一步：保留旧 joint 推理，新增 EE action 模式

没有删除旧 joint-action 推理逻辑。`async_client` 默认仍是：

```text
action_mode = joint
```

只有显式传入：

```text
--action_mode=ee_local_se3
```

时才进入 EE 路线。这样旧的 `run_nero_infer_server.sh` /
`run_nero_infer_client.sh` 仍可继续运行原来的 joint checkpoint。

### 8.2 第二步：让 client 暴露 16D EE observation/action features

新增 EE wrapper 后，client 对 policy server 暴露的 observation/action 名字变为：

```text
right_ee_x
right_ee_y
right_ee_z
right_ee_rotvec_x
right_ee_rotvec_y
right_ee_rotvec_z
right_gripper_width
left_ee_x
left_ee_y
left_ee_z
left_ee_rotvec_x
left_ee_rotvec_y
left_ee_rotvec_z
left_gripper_width
base_or_head_x
base_or_head_y
```

这和 `020000/pretrained_model/config.json` 中的
`action_feature_names` 完全一致。三路图像仍保持：

```text
front: 1280x800
left_wrist: 640x480
right_wrist: 640x480
```

### 8.3 第三步：Nero 末端 Euler 和 EE rotvec 的转换

Nero SDK 当前读出的末端姿态按 Euler 角处理，不使用 TCP offset，
对应 link7/flange pose。转换逻辑是：

```text
Nero base-frame [x,y,z,roll,pitch,yaw]
-> hand-eye base->camera
-> camera-frame [x,y,z,rotvec_x,rotvec_y,rotvec_z]
```

policy 输出 16D EE action 后反向转换：

```text
camera-frame EE target
-> hand-eye camera->base
-> Nero base-frame [x,y,z,roll,pitch,yaw]
```

hand-eye 文件为 camera -> base：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_right_arm_tsai.yml
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_left_arm_tsai.yml
```

client 内部会在需要 base -> camera 时取逆。

### 8.4 第四步：cuRobo IK 只负责 EE target 到 joint target

没有使用 Nero SDK 的 `move_p`，也没有使用 Nero 底层 IK。
EE target 进入 cuRobo 后，使用当前 7D 关节作为 seed，选择最接近当前姿态的
IK 解。默认参数：

```text
curobo_robot_file: nero_custom.yml
curobo_num_seeds: 32
curobo_position_threshold: 0.01
curobo_rotation_threshold: 0.05
curobo_device: cuda
```

cuRobo 输出左右臂 7D joint target 后组装回原来的 Nero action dict：

```text
right_nero_joint_1..7
left_nero_joint_1..7
right_gripper_width
left_gripper_width
```

### 8.5 第五步：继续复用原 joint safety 和 high-rate executor

IK 输出的 joint target 不直接发给 SDK，而是进入原来的：

```text
SafeNeroRobot.send_action()
-> max_policy_step_rad / max_gripper_step_m
-> high-rate interpolation
-> max_executor_step_rad / max_executor_gripper_step_m
-> NeroDualRobot.send_action()
-> arm.send_action()
-> move_js
```

因此 EE 路线仍受这些原有安全参数约束：

```text
safety.max_policy_step_rad=0.05
safety.max_gripper_step_m=0.05
safety.high_rate_control=true
safety.high_rate_dt_s=0.005556
safety.max_executor_step_rad=0.005
safety.max_executor_gripper_step_m=0.004
```

### 8.6 第六步：新增 EE 版本 bash 启动脚本

新增脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_server.sh
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/run_nero_ee_infer_client.sh
```

启动方式：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_ee_infer_server.sh
```

另一个终端：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
bash scripts/run_nero_ee_infer_client.sh
```

client 默认使用：

```text
policy_path: /home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/pi05_fold_towel_ee_bryce301best/pi05_fold_towel_ee_local_se3_rel_8gpu/checkpoints/020000/pretrained_model
action_mode: ee_local_se3
right CAN: nero_right
left CAN: nero_left
policy_device: cuda
curobo_device: cuda
client_device: cpu
actions_per_chunk: 50
chunk_size_threshold: 0.8
aggregate_fn_name: average
```

可用环境变量临时覆盖：

```bash
NERO_DRY_RUN=true bash scripts/run_nero_ee_infer_client.sh
NERO_RIGHT_CAN=nero_right NERO_LEFT_CAN=nero_left bash scripts/run_nero_ee_infer_client.sh
NERO_EE_POLICY_PATH=/path/to/pretrained_model bash scripts/run_nero_ee_infer_client.sh
```

### 8.7 验证结果

离线仿真验证过完整链路，但没有真实驱动机械臂：

```text
020000 EE checkpoint 可以加载。
server 真实输出 action chunk，shape 为 [1, 3, 16]。
policy action names 和 client EE action_features 完全一致。
client 能把 16D EE action 解包。
EE adapter 能转为左右臂 base-frame 6D EE target。
IK 输入是左右臂 6D pose + 7D 当前关节 seed。
IK 输出 joint dict 后进入原 joint safety / high-rate executor。
最终捕获到的是 move_js 路径，不是 move_p。
```

真实 cuRobo IK 也用 `cuda` 对 ready pose 做过一次离线求解：

```text
right_max_delta_from_seed: about 4.9e-08 rad
left_max_delta_from_seed: about 1.2e-07 rad
```

说明 cuRobo IK adapter 在当前环境中可以调用 CUDA 并返回接近 seed 的解。

### 8.8 发现的问题和当前风险

本次链路检查没有发现会直接打断 EE 推理链路的代码 bug。
已知注意点：

```text
1. 第一次临时仿真脚本把 observation.state.shape 写成 [16]，实际 LeRobot feature 是 (16,)；这是仿真脚本断言问题，不是推理代码问题。
2. CPU 上真实 PI05 推理很慢，3-step chunk 曾耗时约 50.8 s；实际部署应使用 --policy_device=cuda。
3. 当前 020000 EE checkpoint 的 config.json 中 rtc_config = null；server 可保留 async RTC 开关，但实际 RTC guidance 是否生效取决于 checkpoint 是否带启用的 rtc_config。
4. 还没有在真实相机、真实 CAN、真实双臂上跑 EE 版本闭环；实机前建议先 NERO_DRY_RUN=true 检查启动，再低速/有人看护测试。
```

## 9. 如果再次卡住

### 9.1 queue gap 或停顿

优先开启 trace 观察，而不是盲目改参数：

```text
--trace.enabled=true
--trace.dir=/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/nero_inference_records
```

如果动作偶发停顿：

```text
优先看 policy action queue 是否吃空。
再看 high-rate executor 是否频繁 overrun。
```

如果要继续调：

```text
先确认 server 启动值里有 obs_similarity_atol，并记录实际数值。
当前临时组合是 obs_similarity_atol=0.1、50 / 0.8、aggregate_fn_name=average、async_rtc.latency_quantile=0.9。
若动作更新太少或日志大量 too similar，可试 0.1；若 server 压力过大，可试 0.2。
若还要比较 chunk，再回测 40 / 0.9，但必须保持 obs_similarity_atol 相同。
不建议回到 50 / 0.5。
```
