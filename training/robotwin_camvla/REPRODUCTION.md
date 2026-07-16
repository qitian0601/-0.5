# CamVLA Reproduction Plan

参考工作：From Fixed to Free Cameras: Calibration-Free View-Robust
Vision-Language-Action Model, arXiv:2607.05396。

## Goal

模型从单个第三人称 RGB、当前机器人状态和语言指令预测：

1. 相机坐标系下的双臂末端增量动作。
2. 相机到机器人公共基坐标系的 6DoF hand-eye pose。

预测结果通过确定性旋转变换组合成基坐标系动作，再交给 IK 或底层控制器执行。

## Stage 0: Data and Geometry Tests

- 读取同步 RGB、`ee_pose`、夹爪宽度和相机外参。
- 构造连续帧 SO(3) 增量。
- 对所有训练视角执行坐标往返测试。
- 检查同一 simulator state 在不同视角下转换回 base frame 后动作一致。

Exit criteria：位置和旋转往返误差接近浮点数值精度，且没有视角相关符号翻转。

## Stage 1: Base-Frame Pi0.5 Baseline

- 只输入 C0。
- 输出双臂 base-frame 6DoF delta + gripper。
- 保持 10 Hz，先验证单任务闭环成功率。

该阶段用于区分任务学习失败和视角泛化失败。

## Stage 2: Oracle CamVLA

- Action Head 输出 camera-frame delta action。
- 推理时使用数据中的真实 `R_bc` 转换回 base frame。
- 暂不使用预测的 Geometric Head。

该阶段验证 camera-centric action representation 是否有效，是最重要的中间消融。

## Stage 3: Geometric Head

- 从第三人称 RGB 特征预测 hand-eye translation 和 axis-angle rotation。
- 可先单独预训练用于调试，但最终训练不能永久 detach 视觉特征。
- 记录 translation error 和 rotation error；旋转误差优先级更高。

## Stage 4: Joint Training

最终目标：

```text
loss = action_loss + 0.1 * extrinsic_mse
```

Action Head 与 Geometric Head 共享视觉编码器并端到端优化。至少比较：

```text
Pi0.5 base-frame baseline
Oracle CamVLA with ground-truth extrinsics
CamVLA with predicted extrinsics
```

## Stage 5: Scale and View Coverage

当前 40 条训练轨迹乘 3 个训练视角，共 120 个 view-trajectory 样本，只适合管线验证。

建议规模：

| Level | Base trajectories | Training views | Purpose |
| --- | ---: | ---: | --- |
| Smoke test | 40 | 3 | 验证数据和 loss |
| Single-task reproduction | 100 | 5-7 | 验证未见视角提升 |
| Paper-scale per task | 100 | 13 | 1300 view-trajectory samples |

论文仿真训练视角为 0 deg 和 +/-15 deg 到 +/-90 deg，测试使用未参与训练的 5 deg 网格。
如果使用同一 simulator state 重渲染全部视角，应在实验报告中标记为 paired re-rendering，
不要声称与每个视角独立采集 100 条完全相同。

## Evaluation

至少报告：

- C0 seen-view success rate。
- C1/C2 training-view success rate。
- C3/C4 held-out-view success rate。
- Geometric Head 平移误差和旋转角误差。
- Oracle 与 predicted extrinsics 的性能差距。
- 每种视角下不少于 20 次闭环 rollout。

腕部视频不作为第一版模型输入，以免模型绕过第三人称视角变化。

