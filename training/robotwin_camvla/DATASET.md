# Multiview Dataset Contract

## Episode Layout

```text
<split>/episode_NNN/
├── _SUCCESS
├── data.hdf5
├── episode.json
├── instruction.json
├── c0.mp4
├── c1.mp4
├── c2.mp4
├── c3.mp4              # test only
├── c4.mp4              # test only
├── left_wrist.mp4
└── right_wrist.mp4
```

只有 `_SUCCESS` 存在时，该 episode 才被视为完整数据。

## HDF5 Fields

主要字段如下：

```text
timestamp
sim_step
frame_index
success
observation/state
action/commanded
robot/{left,right}/qpos
robot/{left,right}/qvel
robot/{left,right}/commanded_qpos
robot/{left,right}/ee_pose
robot/{left,right}/gripper_width_m
robot/{left,right}/commanded_gripper_width_m
robot/{left,right}/gripper_velocity_m_s
objects/{yellow_cube,green_cube}/pose
objects/{yellow_cube,green_cube}/linear_velocity
objects/{yellow_cube,green_cube}/angular_velocity
cameras/<view>/intrinsic_cv
cameras/<view>/extrinsic_cv
cameras/<view>/cam2world_gl
```

视频、状态、动作、相机矩阵和时间戳均以 10 Hz 同步记录。物理仿真频率为 250 Hz。

## Current Action Representation

`action/commanded` 的顺序为：

```text
right_arm_7, left_arm_7, right_gripper_width_m, left_gripper_width_m
```

它是 16 维关节目标，不是 CamVLA 所需的末端位姿增量。

## CamVLA Target Representation

对每个机械臂，应从相邻帧的 `ee_pose` 构造：

```text
[delta_position_xyz, delta_rotation_axis_angle_xyz, gripper]
```

双臂输出共 14 维。设 `R_bc` 为 camera-to-base 旋转：

```text
delta_p_camera = transpose(R_bc) @ delta_p_base
delta_r_camera = transpose(R_bc) @ delta_r_base

delta_p_base = R_bc @ delta_p_camera
delta_r_base = R_bc @ delta_r_camera
```

四元数差必须先通过 SO(3) 对数映射转换成轴角增量，不能直接对四元数分量做减法。

## Coordinate Checks

转换器至少需要验证：

1. `base -> camera -> base` 往返误差。
2. OpenCV 和 OpenGL 相机坐标轴约定。
3. `extrinsic_cv` 是 world-to-camera 还是 camera-to-world。
4. RoboTwin `ee_pose` 所在坐标系是否等同于控制公共基坐标系。
5. 四元数顺序和乘法方向。
6. 左右臂使用同一个公共基坐标系。

这些检查通过之前，不应启动大规模训练。

