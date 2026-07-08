# cuRobo IK 适配与 Nero 回放工作日志

本文档整理本对话框内围绕 Nero 双臂、cuRobo IK、EE pose 回放、move_js 回放和 pickplace 数据回放所做的工作。

## 1. 背景

最初目标是将原先依赖 Nero SDK `move_p()` 的 EE pose 回放流程，改造成使用 cuRobo 做 IK，再用 Nero SDK `move_js()` 执行关节命令。

涉及的主要路径：

```text
/home/chenglong/workplace/nero_teleop_ws
/home/chenglong/workplace/nero_teleop_ws/lerobot
/home/chenglong/workplace/nero_teleop_ws/curobo
```

涉及的主要数据：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/pickplace_flange_pose_001
```

## 2. cuRobo 获取与环境安装

### 2.1 GitHub 克隆问题

最开始直接克隆 cuRobo 时遇到 TLS 连接异常：

```text
gnutls_handshake() failed: The TLS connection was non-properly terminated
```

后续使用镜像站下载 cuRobo：

```text
https://ghfast.top/?q=https%3A%2F%2Fgithub.com%2FNVlabs%2Fcurobo.git
```

### 2.2 conda 环境选择

用户明确希望使用 conda 环境，不使用 `uv venv`。最终采用方案：

```bash
conda activate lerobot
```

把 cuRobo 安装进 `lerobot` 环境，而不是单独使用 `uv` 创建虚拟环境。

安装过程中遇到：

- conda 环境内没有 `pip`。
- 国内 PyPI 镜像下载 NVIDIA wheel 时出现 403。
- cuRobo 依赖中的 CUDA/PyTorch 相关包需要从可用源安装。

最终在 `lerobot` 环境中验证：

```bash
python -c "import curobo; print(curobo.__version__)"
```

曾确认版本为：

```text
0.8.0.post1.dev36
```

## 3. `nero_custom.yml` 配置修正

用户已有一份 Nero cuRobo 机械臂配置 `nero_custom.yml`。适配过程中发现并修正了以下问题。

### 3.1 顶层结构

cuRobo 当前接口需要：

```yaml
robot_cfg:
  kinematics:
    ...
```

原始文件顶层不匹配，因此进行了修正。

### 3.2 资源路径

原文件中存在旧路径：

```text
/home/zfc/...
```

修正为当前机器路径：

```text
/home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/assets/robot/nero
```

### 3.3 tool frame 修正

最关键的问题是 tool frame。

最初使用 `gripper_tcp` 时，cuRobo FK/IK 与数据集中的 EE pose 存在约 13.5 cm 偏移。通过对比记录关节角的 FK 和数据中的 EE pose，确认数据集的 EE pose 对应 `link7`，不是 `gripper_tcp`。

最终配置为：

```yaml
tool_frames:
- link7
```

当前需要保持同步的两份配置：

```text
/home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/configs/robot/nero_custom.yml
/home/chenglong/miniconda3/envs/lerobot/lib/python3.12/site-packages/curobo/content/configs/robot/nero_custom.yml
```

## 4. EE pose 回放脚本改造

主要脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_nero_dual_ee_pose.py
```

新增或修改的能力：

- 支持 `--ik-backend=sdk|curobo`。
- 支持 `--curobo-robot=nero_custom.yml`。
- 支持 EE pose 欧拉角转 cuRobo quaternion。
- 支持使用当前关节作为 cuRobo IK seed。
- 支持选择离当前关节最近的冗余 IK 解。
- 支持 `move_js()` 关节执行。
- 支持最大关节步长限制。
- 支持首帧平滑接管。
- 支持每帧 feedback 等待和软超时。
- 支持逐帧 profiling CSV。
- 支持从 metadata 中按 action names 拆分 EE pose。
- 支持指定 episode 或 `--episode=all`。

## 5. EE pose action 格式处理

### 5.1 bus_table 数据格式

bus_table 的 EE pose 数据顺序是：

```text
right x y z roll pitch yaw
right_gripper_width
left x y z roll pitch yaw
left_gripper_width
```

即：

```text
[right6, right_gripper, left6, left_gripper]
```

### 5.2 pickplace flange pose 数据格式

后续用户要求回放：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/pickplace_flange_pose_001
```

检查后发现其 action names 为：

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

即：

```text
[right6, left6, right_gripper, left_gripper]
```

这与 bus_table 不同。如果直接用旧拆分逻辑，会把左臂 pose 的一部分当成夹爪或错误目标，存在实机风险。

因此 `replay_nero_dual_ee_pose.py` 改为优先读取：

```text
meta/info.json -> features.action.names
```

并按字段名拆分左右臂 pose 和夹爪宽度；没有 metadata 时才回退旧格式。

## 6. ready pose 处理

启动 ready 阶段保留 Nero SDK `move_p()`，没有改成 cuRobo IK。

原因：

- ready pose 是旧脚本里已验证可用的 EE pose。
- cuRobo 在 7DOF 冗余空间可能选到另一个合法但不符合预期的关节姿态。
- 保留 `move_p()` 可以降低启动阶段突然跳变风险。

回放阶段才使用：

```text
cuRobo IK -> move_js()
```

## 7. 平滑与安全限制

曾遇到机械臂一开始瞬移到奇怪姿态的问题。为降低风险，加入了：

```text
--max-joint-step-rad
--interpolate-first-target
--interpolate-each-frame
--joint-target-tolerance-rad
--joint-wait-timeout-s
--joint-timeout-error-rad
--control-dt-s
```

其中：

- `--interpolate-first-target`：回放前平滑移动到第一帧。
- `--interpolate-each-frame`：每一帧都等待关节反馈到目标附近。
- `--max-joint-step-rad`：限制单次关节命令最大变化量。
- `--joint-target-tolerance-rad`：反馈进入目标附近的容差。
- `--joint-wait-timeout-s`：等待反馈的超时时间。
- `--joint-timeout-error-rad`：软超时和硬错误的分界。

## 8. profile 诊断工作

为了判断“机械臂后半段很慢、抖动明显”是否由 IK 造成，加入了 profiling CSV：

```text
dataset
phase
frame_index
frame_count
fps
frame_total_s
right_ik_s
left_ik_s
right_command_s
left_command_s
right_wait_s
left_wait_s
right_steps
left_steps
right_soft_timeouts
left_soft_timeouts
right_max_feedback_error_rad
left_max_feedback_error_rad
```

### 8.1 离线 IK 计时结论

对 bus_table ep000 做离线 cuRobo IK profile：

```text
frames: 682
right_ik median: about 0.0021 s
left_ik median: about 0.0021 s
combined IK median: about 0.0043 s
IK failures: 0
```

结论：

```text
慢不是 cuRobo IK 算不出来，也不是 IK 计算慢。
```

### 8.2 实机 profile 结论

用户运行实机后输出最慢帧，看到：

```text
frame_total_s: about 9.13 s
right_wait_s: about 9.02 s
right_steps: 3
right_max_feedback_error_rad: 0.050000
```

当时参数：

```text
joint_target_tolerance_rad = 0.03
joint_wait_timeout_s = 3.0
```

解释：

```text
右臂每个小步都没有进入 0.03 rad 容差。
每小步等待 3 秒。
3 个小步叠加约 9 秒。
```

所以慢和抖的直接原因是 feedback wait，而不是 IK。

## 9. 参数调整结论

为了改善顺畅性，最终建议：

- 保留 `--interpolate-first-target`。
- 删除 `--interpolate-each-frame`。
- 放宽等待容差。
- 缩短等待超时。

典型参数：

```bash
--interpolate-first-target \
--joint-target-tolerance-rad=0.06 \
--joint-wait-timeout-s=0.3 \
--joint-timeout-error-rad=0.12 \
--control-dt-s=0.05
```

效果：

- 运动更顺畅。
- 精度略降，因为不再每帧等待机械臂完全到位。

这是精度和流畅性的取舍。

## 10. bus_table cuRobo 回放脚本

当前 bus_table EE pose cuRobo 回放入口：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay.sh
```

运行：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/curobo_replay.sh
```

该脚本当前回放：

```text
bus_table_01_ep000_ee_pose
bus_table_01_ep001_ee_pose
bus_table_01_ep002_ee_pose
bus_table_01_ep003_ee_pose
bus_table_01_ep004_ee_pose
```

注意：曾使用过的名字 `replay_bus_table_first5_ee_pose.sh` 当前不存在，内容对应现在的：

```text
scripts/curobo_replay.sh
```

## 11. move_js 原始关节回放脚本

用户要求为原始关节数据：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table/bus_table_01
```

写一个 `move_js` replay 脚本。

新增脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_bus_table_01_move_js.sh
```

运行：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/replay_bus_table_01_move_js.sh 0
```

也支持：

```bash
./scripts/replay_bus_table_01_move_js.sh 3
./scripts/replay_bus_table_01_move_js.sh all
```

### 11.1 噪音和抖动问题

最初版本的 move_js 脚本存在两个危险差异：

- 通道使用 `right=nero_right,left=nero_left`，与当前 EE pose 脚本一致。
- 开启了 `--high_rate_control=true`，高频持续发命令。

这可能导致一启动机械臂噪音大并抖动。

后续修正为：

```text
right=nero_right
left=nero_left
speed_percent=20
move_method=move_js
max_step_rad=0.05
dataset.fps=20
high_rate_control disabled
```

## 12. pickplace flange pose 回放脚本

用户要求用 cuRobo IK 回放：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/pickplace_flange_pose_001
```

新增脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay_pickplace_flange_pose_001.sh
```

运行：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/curobo_replay_pickplace_flange_pose_001.sh
```

该数据集包含 5 个 episode：

```text
episode 0: 881 frames
episode 1: 990 frames
episode 2: 1025 frames
episode 3: 990 frames
episode 4: 934 frames
```

最初只回放 episode 0 后进程退出。原因是 `load_actions()` 写死 `episode_index == 0`。

后续已改为支持：

```bash
--episode=all
```

现在 pickplace 脚本会依次回放 episode 0 到 4，并在每条开始前提示按 Enter。

## 13. 当前主要交付文件

### 13.1 Python 脚本

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_nero_dual_ee_pose.py
```

职责：

- 读取 EE pose dataset。
- 根据 metadata 拆分 action。
- 用 cuRobo 做 IK。
- 用 Nero `move_js()` 执行。
- 处理平滑、等待、profile、episode。

### 13.2 bus_table cuRobo 回放

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay.sh
```

### 13.3 pickplace cuRobo 回放

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay_pickplace_flange_pose_001.sh
```

### 13.4 原始关节 move_js 回放

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_bus_table_01_move_js.sh
```

### 13.5 项目总结文档

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/docs/source/nero/curobo_ik_replay_summary.md
/home/chenglong/workplace/nero_teleop_ws/lerobot/docs/source/nero/curobo_ik_work_log.md
```

## 14. 常用运行命令

### 14.1 bus_table EE pose + cuRobo IK

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/curobo_replay.sh
```

### 14.2 pickplace flange pose + cuRobo IK

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/curobo_replay_pickplace_flange_pose_001.sh
```

### 14.3 bus_table 原始关节 move_js

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/replay_bus_table_01_move_js.sh 0
```

### 14.4 查看 profile 最慢帧

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
conda activate lerobot
python - <<'PY'
import pandas as pd
p = "artifacts/curobo_replay_profile.csv"
df = pd.read_csv(p)
cols = [
    "phase",
    "frame_index",
    "frame_total_s",
    "right_ik_s",
    "left_ik_s",
    "right_wait_s",
    "left_wait_s",
    "right_steps",
    "left_steps",
    "right_max_feedback_error_rad",
    "left_max_feedback_error_rad",
]
print(df[cols].sort_values("frame_total_s", ascending=False).head(20).to_string(index=False))
PY
```

pickplace profile 路径：

```text
artifacts/curobo_pickplace_flange_pose_001_profile.csv
```

## 15. 测试与验证

本轮新增和维护的测试包括：

```text
tests/scripts/test_replay_nero_dual_ee_pose_curobo.py
tests/scripts/test_replay_bus_table_01_move_js_script.py
```

曾验证过的命令：

```bash
conda run -n lerobot pytest tests/scripts/test_replay_nero_dual_ee_pose_curobo.py -q
conda run -n lerobot pytest tests/scripts/test_replay_bus_table_01_move_js_script.py -q
bash -n scripts/curobo_replay.sh
bash -n scripts/curobo_replay_pickplace_flange_pose_001.sh
bash -n scripts/replay_bus_table_01_move_js.sh
conda run -n lerobot python -m py_compile scripts/replay_nero_dual_ee_pose.py
```

最近一次相关验证曾达到：

```text
test_replay_nero_dual_ee_pose_curobo.py: 14 passed
```

## 16. 当前已知注意事项

1. 实机回放前确认双臂周围安全，脚本会实际移动机械臂。
2. `--interpolate-each-frame` 不建议默认打开；它会提升每帧到位精度，但容易因 feedback wait 导致卡顿。
3. 如果运动顺畅但精度不足，优先尝试降低 `fps` 或 `max-joint-step-rad`，不要直接恢复每帧强等待。
4. 修改 `nero_custom.yml` 后要同步源码 cuRobo 配置和 conda env site-packages 配置。
5. 不同数据集 action 顺序可能不同，必须依赖 metadata action names 拆分，不能手写假设。
6. `scripts/replay_bus_table_first5_ee_pose.sh` 当前不存在；使用 `scripts/curobo_replay.sh`。

## 17. 后续建议

- 如果希望恢复旧文件名，可以创建 `replay_bus_table_first5_ee_pose.sh` 指向 `curobo_replay.sh` 的同等内容。
- 将 profile 分析脚本单独保存为工具，方便每次回放后快速定位慢帧。
- 对右臂长期 `0.05 rad` feedback error 的姿态段做单独分析，检查是否接近关节限位、负载限制或控制器限速。
- 若 pickplace 数据继续增加，建议做一个通用 `curobo_replay_dataset.sh <dataset_path> [episode|all]`，减少每个数据集写一个 wrapper。
