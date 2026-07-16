# fold_towel

PI0.5 fixed-base EE relative-action training template for the fold-towel project.

This folder only contains the project scaffold. It does not contain a dataset, trained checkpoint, or Slurm job output.

## Files

- `config.template.env`: project-specific values to fill in before training.
- `train_pi05_relative.sh`: training launcher based on the current PI0.5 relative-action run.
- `DATA_FORMAT.md`: exact fixed-base 14-D robot/human mixed-training dataset format.

## Required Values

Fill these before running:

```bash
DATASET_REPO_ID=
DATASET_ROOT=
OUTPUT_DIR=
JOB_NAME=
```

Meaning:

- `DATASET_REPO_ID`: Hugging Face dataset id, for example `bryce301best/fold_towel_ee`.
- `DATASET_ROOT`: local LeRobot dataset path.
- `OUTPUT_DIR`: output directory for checkpoints and logs.
- `JOB_NAME`: training job name.

## Current Template Defaults

The script keeps the current training choices:

```bash
POLICY_TYPE=pi05
POLICY_PRETRAINED_PATH=hf_downloads/models/pi05_base
USE_RELATIVE_ACTIONS=true
RELATIVE_ACTION_TYPE=ee_so3
RELATIVE_EXCLUDE_JOINTS='["gripper"]'
DTYPE=bfloat16
GRADIENT_CHECKPOINTING=true
COMPILE_MODEL=false
CHUNK_SIZE=50
N_ACTION_STEPS=50
BATCH_SIZE=32
STEPS=16000
SAVE_FREQ=2000
NUM_WORKERS=8
SEED=1000
WANDB_ENABLE=false
NUM_PROCESSES=8
```

## Dataset Requirement

See `DATA_FORMAT.md` for the full conversion specification.

For Nero fixed-base PI0.5 training, the main policy state/action should be absolute 14-D
dual-arm EE poses in the fixed robot frame:

```text
observation.state = [right_xyz, right_rotvec, right_gripper,
                     left_xyz,  left_rotvec,  left_gripper]
action            = [right_xyz, right_rotvec, right_gripper,
                     left_xyz,  left_rotvec,  left_gripper]
```

Do not keep executable base/head motion in the main `action` for Nero fixed-base runs.
If a source dataset contains human head/base motion, absorb it into the EE target pose before
training:

```text
T_fixed_ee_t = inv(T_world_fixed) @ T_world_head_or_base_t @ T_head_or_base_ee_t
```

Usually `fixed` is the episode's first head/base camera frame or the calibrated Nero robot
base frame. Complete head/base 6D motion can be stored as non-observation metadata, but it is
not part of the main policy action.

## Human/Robot Mixed Training Masks

Training supports two optional batch fields for mixed robot and human-video samples:

```text
valid_action_mask  shape: [B, 14] or [B, chunk_size, 14]
valid_image_mask   shape: [B, num_image_features]
```

`valid_action_mask` marks which action dimensions are supervised. Robot samples should supervise all
14 dims. Human-video samples without gripper width should not supervise the two gripper dims:

```text
robot valid_action_mask = [1,1,1,1,1,1,1, 1,1,1,1,1,1,1]
human valid_action_mask = [1,1,1,1,1,1,0, 1,1,1,1,1,1,0]
```

`valid_image_mask` follows the configured image feature order. For a three-camera policy
using `[front, left_wrist, right_wrist]`:

```text
robot valid_image_mask = [1, 1, 1]
human valid_image_mask = [1, 0, 0]
```

Masked camera inputs are replaced with SigLIP padding value `-1` and passed to the model with
image mask `0`; they are not treated as real wrist-camera observations.

The dataset `meta/stats.json` must match the final 14-D fixed-base action/state representation.
For `ee_so3`, keep the parquet action/state absolute and compute normalization stats over the
same transformed representation used by training. Do not use the old EE-local SE(3) 16-D stats
script for fixed-base 14-D training.

Human-video samples may store missing gripper values as `0` in `observation.state` and `action`,
but those gripper positions must have `valid_action_mask=0`. Prefer also marking missing gripper state
in a separate metadata field if a future tokenizer/state-mask path will use it.

On this Slurm cluster, do not download data, recompute stats, or train on the management node. Request a compute node first.

## Run

After filling the required values:

```bash
bash fold_towel/train_pi05_relative.sh
```

If `NUM_PROCESSES` is greater than 1, the script uses `accelerate launch`.
