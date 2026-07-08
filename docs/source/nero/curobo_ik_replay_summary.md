# Nero cuRobo IK 回放项目总结

本文档总结本轮将 Nero 双臂 EE pose 数据回放适配到 cuRobo IK 的工作、诊断结论和当前可用脚本。

## 1. 目标

原始回放脚本使用 Nero SDK 的 `move_p()` 对 EE pose 做逆解和执行。本轮目标是：

- 在 `lerobot` conda 环境中使用 cuRobo。
- 将数据集中的 EE pose 转换为 cuRobo IK 目标。
- 使用 cuRobo IK 解出 7 维 Nero 关节角。
- 用 Nero SDK `move_js()` 执行关节命令。
- 尽量保持原始数据集、脚本入口和硬件连接方式不变。

目标数据主要是：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table/*_ee_pose
```

其中 action 格式为：

```text
right: x y z rx ry rz gripper
left:  x y z rx ry rz gripper
```

后三维 `rx, ry, rz` 按 Nero SDK `move_p()` 使用的欧拉角格式处理，当前匹配结果使用 `xyz` 顺序。

## 2. 环境和安装

用户希望使用 conda，而不是 `uv venv` 或单独虚拟环境。因此 cuRobo 安装到了现有 `lerobot` conda 环境中。

当前已验证：

```bash
conda activate lerobot
python -c "import curobo; print(curobo.__version__)"
```

输出曾验证为：

```text
0.8.0.post1.dev36
```

安装过程中遇到过 `pip` 缺失和国内 PyPI 镜像 403 的问题，后续通过在 conda 环境内补齐 pip 并绕开失效镜像完成安装。

## 3. cuRobo 机器人配置修正

用户提供了 `nero_custom.yml`。初始问题包括：

- YAML 顶层不是 cuRobo 当前接口期望的 `robot_cfg:`。
- 资源路径指向旧机器路径 `/home/zfc/...`。
- 初始 tool frame 使用了 `gripper_tcp`，导致 EE pose 对不上数据集。

已修正的关键配置：

```yaml
robot_cfg:
  kinematics:
    asset_root_path: /home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/assets/robot/nero
    tool_frames:
    - link7
```

目前需要注意有两份有效配置：

```text
/home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/configs/robot/nero_custom.yml
/home/chenglong/miniconda3/envs/lerobot/lib/python3.12/site-packages/curobo/content/configs/robot/nero_custom.yml
```

实际 `lerobot` 环境运行时读取的是 conda env site-packages 里的 cuRobo content，因此两份配置需要保持同步。

## 4. tool frame 诊断

最关键的几何问题是 tool frame。

通过对比记录数据中的关节角 FK 和 EE pose，确认：

- 数据集 EE pose 对应的是 `link7`。
- 原来的 `gripper_tcp` 比数据中的 EE pose 多出约 13.5 cm 偏移。

因此将 cuRobo 配置中的：

```yaml
tool_frames:
- gripper_tcp
```

改为：

```yaml
tool_frames:
- link7
```

这一步解决了“IK 算出的位姿整体不对”的主要原因。

## 5. 回放脚本改造

主要脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_nero_dual_ee_pose.py
```

新增或改造的能力：

- `--ik-backend sdk|curobo`
- `--curobo-robot nero_custom.yml`
- `--curobo-num-seeds`
- `--curobo-position-threshold`
- `--curobo-rotation-threshold`
- `--euler-order xyz`
- `--max-joint-step-rad`
- `--interpolate-first-target`
- `--interpolate-each-frame`
- `--joint-target-tolerance-rad`
- `--joint-wait-timeout-s`
- `--joint-timeout-error-rad`
- `--control-dt-s`
- `--profile-csv`

核心流程：

```text
EE pose action
  -> scipy Rotation.from_euler("xyz")
  -> cuRobo Pose(position, quaternion_wxyz)
  -> cuRobo IK
  -> Nero move_js(joint_target)
```

ready 阶段保留 Nero SDK `move_p()`：

```text
ready pose 仍然使用 SDK move_p()
replay 帧使用 cuRobo IK + move_js()
```

这样做是因为 ready pose 是原先脚本中已经验证可用的 EE ready 位姿，避免启动阶段 cuRobo 在 7DOF 冗余空间中选到不符合预期的姿态。

## 6. IK seed 和冗余解处理

Nero 是 7DOF，IK 可能存在多个满足同一 EE pose 的关节解。为了避免同一 pose 对应到跳变较大的另一个解，`CuroboArmIK.solve()` 使用当前关节反馈作为 seed，并在 cuRobo 返回的多个 seeds 中选择离当前关节最近的解。

逻辑效果：

- 当前关节角作为 `current_state` 传给 cuRobo。
- `return_seeds = min(8, num_seeds)`。
- 对成功解计算与当前 seed 的关节空间距离。
- 选择距离最小的解。

这减少了冗余 IK 解切换造成的突然姿态变化。

## 7. 平滑和安全限制

为避免机械臂瞬移，加入了关节步长限制：

```text
--max-joint-step-rad
```

底层逻辑是对每个关节的单次命令增量做 clip：

```text
next = current + clip(target - current, -max_step, max_step)
```

也加入了两种插值模式：

- `--interpolate-first-target`：开始回放前，从当前关节平滑到第一帧。
- `--interpolate-each-frame`：每一帧都插值并等待反馈到位。

后续实测发现，`--interpolate-each-frame` 会显著降低流畅性，原因见下一节。

## 8. 性能诊断和结论

用户观察到：

- 机械臂抖动明显。
- 靠近远端位姿时动作特别慢。
- 有时一秒甚至数秒才动一小步。

为定位原因，加入了 `--profile-csv`，每帧记录：

```text
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

离线 cuRobo IK 计时结论：

```text
frames: 682
right_ik median: about 0.0021 s
left_ik median:  about 0.0021 s
combined IK median: about 0.0043 s
IK failures: 0
```

因此“后面一秒多才动一小步”不是 IK 解不出来，也不是 IK 计算慢。

真实回放 profile 显示慢帧主要集中在右臂等待反馈：

```text
frame 204/206/207/208: frame_total_s about 9.13 s
right_wait_s about 9.02 s
right_steps = 3
right_max_feedback_error_rad = 0.05
```

当时参数：

```text
joint_target_tolerance_rad = 0.03
joint_wait_timeout_s = 3.0
```

解释：

```text
右臂每个小步都未进入 0.03 rad 容差。
每小步等待 3 秒超时。
3 个小步叠加为约 9 秒。
```

所以慢和抖的主要原因是：

```text
每帧强制等待右臂反馈到位，而右臂在部分姿态下只能接近到约 0.05 rad 误差。
```

## 9. 当前推荐参数

为了让运动更连续，当前推荐关闭每帧强等待，只保留第一次到第一帧目标的平滑接管：

```bash
--interpolate-first-target \
--joint-target-tolerance-rad=0.06 \
--joint-wait-timeout-s=0.3 \
--joint-timeout-error-rad=0.12 \
--control-dt-s=0.05 \
```

不要启用：

```bash
--interpolate-each-frame
```

原因：

- 开启后精度更高，但右臂跟踪误差超过容差时会 stop-and-go。
- 关闭后按 fps 连续发送目标，运动顺畅很多，但不会等待每一帧完全到位，轨迹精度会略降。

这是流畅性和跟踪精度之间的取舍。

## 10. 当前脚本

当前 EE pose + cuRobo IK 回放脚本入口是：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/curobo_replay.sh
```

运行：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/curobo_replay.sh
```

该脚本会：

- 激活 `lerobot` conda 环境。
- 使用 `replay_nero_dual_ee_pose.py`。
- 连接 `right=nero_right`、`left=nero_left`。
- 设置 `speed_percent=20`。
- 使用 `--ik-backend=curobo`。
- 使用 `nero_custom.yml`。
- 记录 profile 到：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/artifacts/curobo_replay_profile.csv
```

注意：此前使用过的文件名 `replay_bus_table_first5_ee_pose.sh` 当前目录中不存在，内容现在对应 `curobo_replay.sh`。

## 11. profile 查看命令

回放后查看最慢帧：

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

如果 `right_wait_s` 或 `left_wait_s` 很大，说明瓶颈在等待机械臂实际反馈到目标附近，不在 IK。

## 12. move_js 原始关节回放脚本

另外新增了一个直接回放原始 joint action 数据的脚本：

```text
/home/chenglong/workplace/nero_teleop_ws/lerobot/scripts/replay_bus_table_01_move_js.sh
```

它读取：

```text
/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table/bus_table_01
```

并使用：

```text
nero-replay-dual-joint
move_js
```

运行示例：

```bash
cd /home/chenglong/workplace/nero_teleop_ws/lerobot
./scripts/replay_bus_table_01_move_js.sh 0
```

安全相关修正：

- 通道改为与 EE pose 脚本一致：`right=nero_right`、`left=nero_left`。
- `speed_percent=20`。
- 关闭 `high_rate_control=true`。
- 设置 `max_step_rad=0.05`。
- 设置 `dataset.fps=20`。

此前高频控制和通道不一致可能导致启动时噪音和抖动。

## 13. 已知问题和风险

### 13.1 精度和顺畅性的取舍

关闭 `--interpolate-each-frame` 后，机械臂不再每帧等待完全到位，因此运动更顺，但轨迹精度会下降。

如果需要更高精度，可以尝试折中：

```bash
--fps=15
--max-joint-step-rad=0.04
```

保持不启用 `--interpolate-each-frame`。

### 13.2 右臂反馈误差

profile 显示右臂部分姿态下长期停在约 `0.05 rad` 误差附近。可能原因包括：

- 机械臂控制器内部限速或负载限制。
- 远端姿态接近关节限制或奇异区域。
- `move_js()` 是异步命令，反馈尚未完全追上时下一帧已到来。
- 某些关节机械阻力或控制增益导致跟踪误差。

### 13.3 cuRobo 配置同步

修改 `nero_custom.yml` 时要同时确认源码 cuRobo 配置和 conda env site-packages 配置。否则脚本运行时可能使用旧配置。

## 14. 验证过的测试

本轮为相关脚本增加或运行过测试：

```bash
conda run -n lerobot pytest tests/scripts/test_replay_nero_dual_ee_pose_curobo.py -q
conda run -n lerobot pytest tests/scripts/test_replay_bus_table_01_move_js_script.py -q
bash -n scripts/curobo_replay.sh
bash -n scripts/replay_bus_table_01_move_js.sh
```

其中 `test_replay_nero_dual_ee_pose_curobo.py` 覆盖：

- Nero EE pose 到 cuRobo quaternion 的转换。
- cuRobo IK 模式下发送 `move_js`。
- ready 阶段仍使用 SDK `move_p()`。
- 关节步长限制。
- first target 插值。
- feedback 等待和软超时。
- profile CSV 输出。

`test_replay_bus_table_01_move_js_script.py` 覆盖：

- 原始 `bus_table_01` 数据集路径。
- `nero-replay-dual-joint` 入口。
- `move_js` 方法。
- `right=nero_right`、`left=nero_left`。
- `speed_percent=20`。
- 禁用高频控制。

## 15. 后续建议

1. 将 `curobo_replay.sh` 重新复制或重命名为用户习惯的 `replay_bus_table_first5_ee_pose.sh`，避免命令找不到。
2. 保留 `--profile-csv`，每次实机测试后先看 `right_wait_s`、`left_wait_s`。
3. 若需要提升精度，优先调低 `fps` 和 `max_joint_step_rad`，不要立刻恢复 `--interpolate-each-frame`。
4. 若右臂仍在特定帧附近抖动，应导出这些 frame 的目标关节和反馈关节，进一步检查是否接近关节限位或机械控制瓶颈。
