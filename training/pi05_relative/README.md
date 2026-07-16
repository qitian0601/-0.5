# PI0.5 Relative Training Snapshot

这是一份从实际训练工作区直接导出的代码快照，覆盖三条流程：

1. 16D 双臂关节角 joint-relative 训练。
2. 14D fixed-base EE/SO(3) robot-human mixed 训练。
3. 16D EE-local SE(3) 数据转换、stats 重算和训练。

数据集、视频、模型和 checkpoint 不在 GitHub 中。

## 快照基线

快照来源于 LeRobot commit：

```text
81948979 fix(deps): cap placo below 0.9.16 and harden kinematics import (#3647)
```

`repo_overlay/` 保留了原工作区中的仓库相对路径，包括训练代码、转换脚本、
stats 脚本、配置、Slurm 作业和测试。目标仓库根目录源码没有被本次提交修改。

## 使用 overlay

在另一台机器上，先准备同版本 LeRobot checkout，然后在仓库根目录执行：

```bash
cp -a training/pi05_relative/repo_overlay/. .
git diff --check
```

这会把快照文件复制到其原始位置，例如 `src/lerobot/`、`scripts/`、
`fold_towel/` 和 `tests/`。应用前应保证工作区干净，并先检查将被覆盖的文件。

安装依赖：

```bash
uv sync --locked --extra test --extra dev
```

## 必须修改的本机路径

历史 Slurm 和 env 文件保留了原训练路径。迁移后先检查：

```bash
rg -n '/public/home/chenglongyan|server[0-9]+' \
  scripts fold_towel
```

至少修改：

- `DATASET_ROOT`
- `OUTPUT_DIR`
- `POLICY_PRETRAINED_PATH`
- conda/uv Python 路径
- Slurm partition、GPU、CPU 和 wall time

不要把 Hugging Face token 写进这些文件。

## 统一原则

三种训练都遵循同一原则：parquet 中保存绝对 `observation.state` 和绝对
`action`，训练 processor 针对 action chunk 在线转换为相对 action。

不要把逐行增量提前写入 parquet。每个 future action 都相对于 chunk
第一个 timestep 的 state，而不是相对于前一个 future action。

stats 重算和训练必须使用相同的：

```text
relative_action_type
chunk_size
relative_exclude_joints
action/state 维度和顺序
```

当前训练统一使用：

```text
chunk_size=50
n_action_steps=50
relative_exclude_joints=["gripper"]
```

## 1. 16D Joint-relative

数据布局：

```text
[right_joint_1..7, left_joint_1..7,
 right_gripper_width, left_gripper_width]
```

对 chunk 起点 `t` 和未来步 `k`：

```text
delta_joint[t+k] = absolute_action[t+k] - observation.state[t]
gripper[t+k] = absolute_gripper_action[t+k]
```

官方 relative stats 命令：

```bash
uv run lerobot-edit-dataset \
  --repo_id local/place_two_cubes_box_lerobot_v3 \
  --root /path/to/place_two_cubes_box_lerobot_v3 \
  --new_repo_id local/place_two_cubes_box_lerobot_v3 \
  --new_root /path/to/place_two_cubes_box_lerobot_v3 \
  --operation.type recompute_stats \
  --operation.overwrite true \
  --operation.skip_image_video true \
  --operation.relative_action true \
  --operation.chunk_size 50 \
  --operation.relative_exclude_joints '["gripper"]'
```

对应历史脚本：

```text
scripts/recompute_place_two_cubes_box_v3_official_stats.sbatch
scripts/train_pi05_place_two_cubes_box_v3_joint_rel_2gpu.sbatch
```

实际验证参数：

```text
policy.type=pi05
relative_action_type=joint
2 GPU / 64 CPU / 1 day
batch_size=32 per GPU, effective batch=64
steps=12000
save_freq=2000
dtype=bfloat16
gradient_checkpointing=true
image augmentation=false
```

数据验证结果：53 episodes、19170 frames，relative action stats count 为
`828650`。成功作业运行约 8 小时 11 分钟。

## 2. 14D EE/SO(3) Robot-human Mixed

完整数据格式见：

[`fold_towel/DATA_FORMAT.md`](repo_overlay/fold_towel/DATA_FORMAT.md)

布局：

```text
[right_xyz, right_rotvec, right_gripper,
 left_xyz, left_rotvec, left_gripper]
```

相对旋转不是 rotvec 直接相减：

```text
delta_position = target_position - current_position
delta_rotation = log(R_target @ R_current.T)
gripper = absolute_target_gripper
```

人类样本：

```text
valid_action_mask = [1,1,1,1,1,1,0, 1,1,1,1,1,1,0]
valid_image_mask  = [1,0,0]  # front, left_wrist, right_wrist
```

机器人样本：

```text
valid_action_mask = [1,1,1,1,1,1,1, 1,1,1,1,1,1,1]
valid_image_mask  = [1,1,1]
```

重算 stats：

```bash
uv run python scripts/recompute_ee_so3_relative_stats.py \
  --dataset-root /path/to/ee14_mixed \
  --repo-id local/ee14_mixed \
  --chunk-size 50 \
  --num-workers 16
```

对应文件：

```text
scripts/recompute_ee_so3_relative_stats.py
scripts/recompute_pickplace_human_56_mixed_ee_so3_stats.sbatch
fold_towel/train_pi05_relative.sh
fold_towel/config.pickplace_human_56_mixed_ee_so3_2gpu_any.env
fold_towel/train_pickplace_human_56_mixed_ee_so3_2gpu_any.sbatch
```

实际验证参数：

```text
relative_action_type=ee_so3
2 GPU / 64 CPU / 2 days
batch_size=16 per GPU, effective batch=32
steps=20000
save_freq=2000
image augmentation=false
```

Human56 mixed 数据为 136 episodes / 67146 frames，包含 80 robot episodes
和 56 human episodes；relative action stats count 为 `3024100`。

## 3. 16D EE-local SE(3)

布局：

```text
[right_xyz, right_rotvec, right_gripper,
 left_xyz, left_rotvec, left_gripper,
 base_or_head_x, base_or_head_y]
```

每个 EE 使用局部 SE(3) 变换：

```text
T_delta = inv(T_current) @ T_target
base_delta = base_target_xy - base_current_xy
gripper = absolute_target_gripper
```

### Base-frame Euler 转 camera-frame 16D

必须提供左右手眼标定 YAML：

```bash
uv run python scripts/convert_ee_euler_base_to_camera_16d.py \
  --src-root /path/to/source_ee14_euler \
  --dst-root /path/to/output_camera_ee16 \
  --right-handeye /path/to/right_handeye.yaml \
  --left-handeye /path/to/left_handeye.yaml \
  --input-layout right7_left7 \
  --base-head-x 0.0 \
  --base-head-y 0.0
```

如果 Euler 源数据使用角度制，加 `--degrees`。不要猜测 Euler 顺序、输入布局
或手眼标定；无法从 metadata 明确确定时直接停止转换。

### Camera-frame EE14 rotvec 扩展为 EE16

```bash
uv run python scripts/add_base_head_xy_to_ee_camera_rotvec_dataset.py \
  --src-root /path/to/source_camera_ee14 \
  --dst-root /path/to/output_camera_ee16 \
  --base-head-x 0.0 \
  --base-head-y 0.0
```

重算 EE-local stats：

```bash
uv run python scripts/recompute_ee_local_se3_relative_stats.py \
  --dataset-root /path/to/output_camera_ee16 \
  --repo-id local/output_camera_ee16 \
  --chunk-size 50 \
  --num-workers 16
```

训练配置：

```text
policy.use_relative_actions=true
policy.relative_action_type=ee_local_se3
policy.chunk_size=50
policy.n_action_steps=50
```

对应历史文件：

```text
fold_towel/config.pickplace_001_003_ee_local_se3_8gpu.env
fold_towel/train_pickplace_001_003_ee_local_se3_8gpu.sbatch
```

## 推理

relative-action checkpoint 必须同时保留并加载：

```text
policy_preprocessor.json
policy_postprocessor.json
normalizer/unnormalizer safetensors
config.json
model.safetensors
```

推理 processor 会在反归一化后执行：

```text
joint:    absolute = relative + current_joint_state
ee_so3:   compose relative translation/rotation with current EE pose
ee_local: T_absolute = T_current @ T_delta
```

不要直接把网络原始输出发送给机器人。

## 测试

应用 overlay 后运行：

```bash
uv run pytest \
  tests/datasets/test_compute_stats.py \
  tests/datasets/test_ee_so3_relative_stats.py \
  tests/processor/test_ee_so3_relative_action_processor.py \
  tests/processor/test_batch_processor.py \
  tests/policies/pi0_pi05/test_pi05.py \
  tests/scripts/test_pi05_training_sampler.py \
  tests/scripts/test_convert_ee_euler_base_to_camera_16d.py \
  tests/scripts/test_recompute_ee_local_se3_relative_stats.py \
  tests/scripts/test_recompute_ee_so3_relative_stats.py -q
```

## 不包含的内容

- 数据集、视频和 parquet
- PI0.5 base model
- 训练 checkpoint 和 optimizer state
- Slurm 输出、缓存和 Hugging Face token
- 没有标定参数时的通用 FK/坐标系猜测
