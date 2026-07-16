# fold_towel Fixed-Base Mixed Training Data Format

This document is the target format for converting fold-towel robot data and human-video data for
PI0.5 mixed training on the fixed-base Nero dual-arm system.

## 1. Training Assumption

Use fixed-base 14-D EE-pose training:

```text
policy.type=pi05
policy.use_relative_actions=true
policy.relative_action_type=ee_so3
policy.chunk_size=50
policy.n_action_steps=50
```

Do not use the old 16-D `ee_local_se3` format for this fixed-base run. The Nero base is fixed, so
executable base/head motion must not remain as an action dimension.

## 2. Per-Frame Required Columns

Every frame in every episode must have the same schema.

```text
observation.state                 float32[14]
action                            float32[14]
observation.images.front          video/image, RGB, same shape for all samples
observation.images.left_wrist     video/image, RGB, same shape for all samples
observation.images.right_wrist    video/image, RGB, same shape for all samples
valid_action_mask                 bool[14]
valid_image_mask                  bool[3]
timestamp                         float32 scalar
frame_index                       int64 scalar
episode_index                     int64 scalar
index                             int64 scalar
task_index                        int64 scalar
```

If you use the official LeRobot writer, `timestamp`, `frame_index`, `episode_index`, `index`, and
`task_index` are filled automatically. Your frame dict still needs `task` as a string.

`meta/tasks.parquet` must use task strings as the pandas index named `task`, with only one column:

```text
index name: task
columns:    task_index
```

Do not store the task text as a normal `task` column in `meta/tasks.parquet`. `LeRobotDataset`
reads `dataset.meta.tasks.iloc[task_index].name` as the task string, so the index value must be a
string.

For a merged LeRobotDataset, do not omit wrist camera keys from human-video samples. Give them
placeholder wrist videos/images and set `valid_image_mask=[1,0,0]`.

## 3. State and Action Layout

Both `observation.state` and `action` use the same 14-D order:

```text
index  name
0      right_x
1      right_y
2      right_z
3      right_rx
4      right_ry
5      right_rz
6      right_gripper
7      left_x
8      left_y
9      left_z
10     left_rx
11     left_ry
12     left_rz
13     left_gripper
```

Units:

```text
xyz      meters
rotvec   radians, SO(3) rotation vector
gripper  same scale as the robot data
```

Keep parquet values absolute. PI0.5 will convert chunks to relative `ee_so3` actions during training.
Do not pre-store per-row relative deltas in `action`.

## 4. Coordinate Frame

All 14 EE dimensions must be expressed in one fixed frame `F`.

Recommended choices:

```text
Robot data: F = Nero calibrated robot base frame, or the fixed camera frame used by deployment.
Human data: F = episode first head/base camera frame, then align this convention to the robot frame.
```

For source data with a moving head/base camera, absorb the moving base/head transform into the EE pose
before writing `observation.state` and `action`:

```text
T_F_ee_t = inv(T_world_F) @ T_world_head_or_base_t @ T_head_or_base_ee_t
```

Meaning:

```text
T_world_F               fixed frame pose in world
T_world_head_or_base_t  moving head/base pose at timestep t
T_head_or_base_ee_t     EE pose measured in the moving head/base frame
T_F_ee_t                final absolute EE pose stored in the dataset
```

After this conversion, remove base/head motion from the main `action`. If you want to keep the full
head/base 6D trajectory for debugging, store it under a metadata key that does not start with
`observation.` and does not start with `action`.

## 5. Robot Samples

Robot samples have real state/action, real front camera, and real wrist cameras.

```text
valid_action_mask = [1,1,1,1,1,1,1, 1,1,1,1,1,1,1]
valid_image_mask  = [1,1,1]
```

The action gripper values at indices 6 and 13 are supervised.

## 6. Human-Video Samples

Human-video samples usually have front/head video only and no gripper width.

Write:

```text
observation.images.front       real human video frame
observation.images.left_wrist  placeholder image/video frame
observation.images.right_wrist placeholder image/video frame
valid_image_mask               [1,0,0]
```

The placeholder wrist images should have the same shape, fps, codec, and pixel format as the robot
wrist videos/images. A black image is fine because `valid_image_mask=0` makes PI0.5 replace it with
SigLIP padding and image mask 0.

For gripper:

```text
observation.state[6]  = 0
observation.state[13] = 0
action[6]             = 0
action[13]            = 0
valid_action_mask     = [1,1,1,1,1,1,0, 1,1,1,1,1,1,0]
```

The current training code masks action loss only. It does not mask state tokens. So human gripper
state values should be a consistent constant, normally 0.

## 7. `meta/info.json` Features

`valid_action_mask` and `valid_image_mask` should be included in `meta/info.json` `features`, because
LeRobot uses `features` as the authoritative parquet schema. Their names are intentionally top-level
and do not start with `observation.` or `action`, so policy feature inference will ignore them as
model inputs/outputs.

Example feature block:

```json
{
  "observation.state": {
    "dtype": "float32",
    "shape": [14],
    "names": [
      "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
      "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper"
    ]
  },
  "action": {
    "dtype": "float32",
    "shape": [14],
    "names": [
      "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
      "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper"
    ]
  },
  "observation.images.front": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "observation.images.left_wrist": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "observation.images.right_wrist": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "valid_action_mask": {
    "dtype": "bool",
    "shape": [14],
    "names": [
      "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
      "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper"
    ]
  },
  "valid_image_mask": {
    "dtype": "bool",
    "shape": [3],
    "names": ["front", "left_wrist", "right_wrist"]
  }
}
```

LeRobot will also add the default metadata features:

```text
timestamp, frame_index, episode_index, index, task_index
```

If your image size is not `480x640`, replace all three image shapes with your final stored resolution.
All robot and human episodes in the merged dataset must use the same feature shapes.

## 8. Video Requirements

For mixed training, use one unified visual schema:

```text
front, left_wrist, right_wrist

```

All episodes must agree on:

```text
camera keys
height/width/channels in meta/features
fps
video codec and pixel format when merging datasets
timestamp convention
```

If human and robot videos have different resolutions, resize during conversion. If codecs or pixel
formats differ, re-encode before merging.

## 9. Stats Requirements

`meta/stats.json` must match the final fixed-base 14-D representation.

Keep these parquet columns absolute:

```text
observation.state
action
```

Then recompute stats for `ee_so3` training:

```bash
python scripts/recompute_ee_so3_relative_stats.py \
  --dataset-root /path/to/fold_towel_fixed_base_ee14 \
  --repo-id local/fold_towel_fixed_base_ee14 \
  --chunk-size 50 \
  --num-workers 0
```

Run this on a Slurm compute node, not on `mgmtserver02`.

Expected result:

```text
stats["observation.state"] is absolute 14-D state stats
stats["action"] is chunk-relative EE/SO(3) 14-D action stats
```

The stats script may also write stats for `valid_action_mask` and `valid_image_mask`. That is harmless
because PI0.5 does not normalize these mask fields.

## 10. Minimal Frame Examples

Robot frame before writer auto-fills metadata:

```python
frame = {
    "observation.state": state_14.astype("float32"),
    "action": action_14.astype("float32"),
    "observation.images.front": front_rgb,
    "observation.images.left_wrist": left_wrist_rgb,
    "observation.images.right_wrist": right_wrist_rgb,
    "valid_action_mask": np.ones(14, dtype=np.bool_),
    "valid_image_mask": np.ones(3, dtype=np.bool_),
    "task": "Fold the towel.",
}
```

Human-video frame before writer auto-fills metadata:

```python
frame = {
    "observation.state": human_state_14.astype("float32"),
    "action": human_action_14.astype("float32"),
    "observation.images.front": front_rgb,
    "observation.images.left_wrist": black_left_wrist_rgb,
    "observation.images.right_wrist": black_right_wrist_rgb,
    "valid_action_mask": np.array([1,1,1,1,1,1,0, 1,1,1,1,1,1,0], dtype=np.bool_),
    "valid_image_mask": np.array([1,0,0], dtype=np.bool_),
    "task": "Fold the towel.",
}
```

## 11. Conversion Checklist

Before training, check:

```text
[ ] observation.state shape is [14] for every frame.
[ ] action shape is [14] for every frame.
[ ] right arm indices are 0..6, left arm indices are 7..13.
[ ] xyz is meters, rotvec is radians.
[ ] all EE poses are already in fixed frame F.
[ ] no base/head motion remains in action.
[ ] robot valid_action_mask is all ones.
[ ] human valid_action_mask has dims 6 and 13 set to zero.
[ ] robot valid_image_mask is [1,1,1].
[ ] human valid_image_mask is [1,0,0].
[ ] human samples still contain placeholder left_wrist/right_wrist images or videos.
[ ] meta/info.json contains valid_action_mask and valid_image_mask features.
[ ] all image/video feature shapes match across robot and human samples.
[ ] meta/tasks.parquet uses task strings as index `task` and only has the `task_index` column.
[ ] meta/stats.json was recomputed with recompute_ee_so3_relative_stats.py and chunk_size=50.
```

## 12. Things Not To Do

Do not:

```text
store 16-D [EE14 + base_xy] for this fixed-base run
use relative_action_type=ee_local_se3 for this fixed-base run
store base/head action in the main action field
pre-store relative EE deltas in action
omit wrist camera keys from human samples
name masks action_mask or observation.images.mask
put debug-only head/base metadata under observation.*
globally mask gripper loss for robot samples
run conversion, stats recompute, or training on mgmtserver02
```
