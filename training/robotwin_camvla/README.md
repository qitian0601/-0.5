# RoboTwin CamVLA Training Preparation

本目录用于管理 `place_two_cubes_box` 的多视角数据准备和 CamVLA 复现实验。

## Current Status

- 已完成 50 条成功轨迹。
- 数据划分：train 40、validation 5、test 5。
- 每条轨迹以同一个 simulator state 同步记录多个视角。
- 训练和验证轨迹包含 C0/C1/C2；测试轨迹额外包含 C3/C4。
- 左右腕部视频已经保存，但 CamVLA 第一版不使用腕部相机。
- 夹爪宽度以米保存，范围为 0 到 0.1 m。

历史生成分为两批：

| Batch | Episodes | Train views | Held-out test views |
| --- | ---: | --- | --- |
| Initial | 19 | C0, pitch +/-15 deg | pitch +/-25 deg |
| Continuation | 31 | C0, pitch +/-10 deg | pitch +/-15 deg |

两批数据只能通过带 `_SUCCESS` 的 episode 目录组成完整的 50 条集合。不要把暂停时遗留的
`.part.mp4` 或无 `_SUCCESS` 目录加入训练。

## Local Data Locations

数据不会提交到本仓库。默认 RoboTwin 工作区中的位置为：

```text
${ROBOTWIN_ROOT}/data/place_two_cubes_box_multiview_50
${ROBOTWIN_ROOT}/data/place_two_cubes_box_multiview_50_pitch10_continuation
```

完整字段见 [DATASET.md](DATASET.md)，训练和评估阶段见 [REPRODUCTION.md](REPRODUCTION.md)。

## Recommended Next Step

先用现有 40 条训练轨迹完成最小闭环：

1. 验证相机外参与末端位姿的坐标约定。
2. 派生 base-frame 和 camera-frame 双臂 6DoF 增量动作。
3. 训练 C0 单视角 Pi0.5 基线。
4. 使用真实外参完成 Oracle CamVLA。
5. 加入 Geometric Head，最后进行端到端联合训练。

当前视角变化主要是俯仰变化，只能验证局部视角鲁棒性。论文级复现还需要增加方位角、
高度和距离变化，并保留训练集之外的视角作为测试集。

## Rendering

当前数据采用高质量路径追踪：1280x800 第三视角、256 samples/pixel、path depth 8。
这更接近离线渲染，而不是大规模训练数据生成。新数据建议先使用
[render_profiles.example.yaml](render_profiles.example.yaml) 中的 `balanced_training` 档位，
并用一条轨迹比较速度和图像质量后再批量生成。

