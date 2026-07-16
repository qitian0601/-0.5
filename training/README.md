# Training Workspace

本目录保存 NERO 双臂、RoboTwin 和 Pi0.5/CamVLA 相关的训练入口与实验说明。
生成数据、模型权重和日志不提交到 Git；这些大文件应保存在本地数据盘或对象存储中。

## Contents

- `train_pi05_baseline.sh`：参数化的 Pi0.5 基线训练脚本。
- `robotwin_camvla/README.md`：RoboTwin 多视角数据现状和使用入口。
- `robotwin_camvla/DATASET.md`：数据结构、视角划分与已知限制。
- `robotwin_camvla/REPRODUCTION.md`：CamVLA 分步复现方案。
- `robotwin_camvla/render_profiles.example.yaml`：当前、推荐和快速渲染档位。
- `robotwin_camvla/scripts/`：RoboTwin 多视角回放、启动和 LeRobot v3 转换脚本。

## Pi0.5 Baseline

训练前需要准备：

1. 已转换为 LeRobot v3 的数据集。
2. Pi0.5 基础预训练模型。
3. CUDA GPU 和本仓库要求的 Python 环境。

示例：

```bash
DATASET_REPO_ID=/path/to/lerobot_dataset \
PRETRAINED_PATH=/path/to/pi05_base_pretrained \
OUTPUT_DIR=outputs/pi05_nero_baseline \
bash training/train_pi05_baseline.sh
```

可通过环境变量覆盖 `STEPS`、`BATCH_SIZE`、`NUM_WORKERS`、`CHUNK_SIZE`、
`N_ACTION_STEPS`、`LEARNING_RATE` 和保存频率。脚本默认不上传模型到 Hugging Face Hub。

## Important Boundary

`robotwin_camvla` 下记录的同步多视角 HDF5 不是现成的 CamVLA 训练集。当前文件中的
`action/commanded` 是双臂关节目标；CamVLA 需要从连续 `ee_pose` 派生双臂末端位姿增量，
并利用每个视角的外参将增量转换到相机坐标系。完成该转换和 LeRobot 数据适配后，才能进入联合训练。
