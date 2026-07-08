import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import lerobot_teleoperator_so101_8dof  # noqa: F401
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.robots.config import RobotConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from .config_nero import NeroDualRobotConfig
from .curobo_ik_adapter import NeroCuroboArmIK
from .ee_local_se3_adapter import FLANGE_POSE_COMPONENTS
from .robot_nero_dual import NeroDualRobot

logger = logging.getLogger(__name__)

CALIB_ROOT = Path("/home/chenglong/workplace/nero_teleop_ws/lerobot/相机_机械臂标定")
DATA_ROOT = Path("/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace_new")

READY_RIGHT_POSE = [
    0.2732181621364244,
    -0.030025818406302345,
    0.35750673546543393,
    -1.6118205859733734,
    0.09584266857003589,
    3.066317491332911,
]
READY_LEFT_POSE = [
    0.2873361223957065,
    -0.01389206349906485,
    0.35802865521534843,
    -1.6379877598473152,
    0.05157470481268328,
    -3.124208233763824,
]

CALIBRATION_READY_RIGHT_JOINTS = [
    -0.015,
    -0.331,
    -0.003,
    1.571,
    -0.094,
    0.053,
    0.655,
]
CALIBRATION_READY_LEFT_JOINTS = [
    -0.023,
    -0.311,
    0.029,
    1.58,
    0.032,
    -0.054,
    0.768,
]

ORIGINAL_READY_RIGHT_JOINTS = [
    -0.01864011641129944,
    -1.72533,
    0.005689773361501515,
    1.8416539734118966,
    -0.055553830090979514,
    -0.028972465583105872,
    1.550797,
]
ORIGINAL_READY_LEFT_JOINTS = [
    0.03324852225049198,
    -1.72533,
    -3.490658503988659e-05,
    1.796868824805722,
    -0.017174039839624202,
    -0.06607816548050531,
    1.550797,
]

EE_FLANGE_EULER_QUATERNION_NAMES = [
    "right_flange_x",
    "right_flange_y",
    "right_flange_z",
    "right_flange_roll",
    "right_flange_pitch",
    "right_flange_yaw",
    "right_flange_qx",
    "right_flange_qy",
    "right_flange_qz",
    "right_flange_qw",
    "left_flange_x",
    "left_flange_y",
    "left_flange_z",
    "left_flange_roll",
    "left_flange_pitch",
    "left_flange_yaw",
    "left_flange_qx",
    "left_flange_qy",
    "left_flange_qz",
    "left_flange_qw",
]


@dataclass(frozen=True)
class CalibrationPoseTarget:
    arm: str
    sequence_index: int
    frame_index: int
    source_dir: Path
    pose: np.ndarray
    timestamp_s: float | None = None
    pause_frame_range: dict[str, int] | None = None


@dataclass
class ReplayCalibrationPosesConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig = field(
        default_factory=lambda: DatasetRecordConfig(
            repo_id="chenglong/pickplace_handeye_replay_actual_flange_quat",
            root=DATA_ROOT / "pickplace_handeye_replay_actual_flange_quat",
            single_task="pickplace",
            fps=30,
            video=True,
            push_to_hub=False,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            video_encoding_batch_size=1,
        )
    )
    right_poses_dir: Path = CALIB_ROOT / "right_arm_stable_front_frames_new_002"
    left_poses_dir: Path = CALIB_ROOT / "left_arm_stable_front_frames_new_003"
    targets_summary_json: Path | None = CALIB_ROOT / "stable_front_frames_new_002_003_summary.json"
    front_camera_key: str = "front"
    euler_order: str = "xyz"
    motion_backend: str = "curobo_move_js"
    curobo_robot: str = "nero_custom.yml"
    curobo_num_seeds: int = 32
    curobo_position_threshold: float = 0.01
    curobo_rotation_threshold: float = 0.05
    curobo_device: str = "cuda"
    ready_right_pose: list[float] = field(default_factory=lambda: list(READY_RIGHT_POSE))
    ready_left_pose: list[float] = field(default_factory=lambda: list(READY_LEFT_POSE))
    ready_right_joints: list[float] = field(default_factory=lambda: list(ORIGINAL_READY_RIGHT_JOINTS))
    ready_left_joints: list[float] = field(default_factory=lambda: list(ORIGINAL_READY_LEFT_JOINTS))
    replay_ready_right_joints: list[float] = field(default_factory=lambda: list(CALIBRATION_READY_RIGHT_JOINTS))
    replay_ready_left_joints: list[float] = field(default_factory=lambda: list(CALIBRATION_READY_LEFT_JOINTS))
    ready_wait_s: float = 2.0
    move_settle_s: float = 1.0
    stable_timeout_s: float = 20.0
    stable_poll_s: float = 0.1
    stable_samples: int = 5
    stable_position_threshold_m: float = 0.001
    stable_rotation_threshold_rad: float = 0.005
    smooth_move: bool = True
    smooth_move_time_s: float = 3.0
    smooth_move_dt_s: float = 0.05
    smooth_min_steps: int = 10
    max_joint_step_rad: float = 0.03
    joint_command_dt_s: float = 0.05
    expected_right_targets: int = 27
    expected_left_targets: int = 27
    reset_to_ready_between_arms: bool = True
    prepare_gripper: bool = True
    gripper_open_width: float = 0.1
    gripper_close_width: float = 0.0
    gripper_force: float = 1.0
    gripper_settle_s: float = 1.0
    prompt_before_capture: bool = True
    prompt_between_targets: bool = True
    overwrite: bool = True
    dry_run: bool = False


def _sort_key(path: Path) -> tuple[int, str]:
    prefix = path.name.split("_", 1)[0]
    try:
        return int(prefix), path.name
    except ValueError:
        return 10**9, path.name


def _pose_from_json(data: dict[str, Any], arm: str) -> np.ndarray:
    key = f"{arm}_observation_pose"
    if key not in data:
        raise KeyError(f"{key!r} not found in pose.json")
    pose_dict = data[key]
    return np.asarray([float(pose_dict[name]) for name in FLANGE_POSE_COMPONENTS], dtype=float)


def load_pose_targets(root: str | Path, *, arm: str) -> list[CalibrationPoseTarget]:
    root = Path(root)
    if arm not in {"right", "left"}:
        raise ValueError(f"arm must be 'right' or 'left', got {arm!r}.")
    if not root.exists():
        raise FileNotFoundError(root)

    targets: list[CalibrationPoseTarget] = []
    for pose_path in sorted(root.glob("*/pose.json"), key=lambda path: _sort_key(path.parent)):
        with pose_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sequence_index = _sort_key(pose_path.parent)[0]
        if sequence_index == 10**9:
            sequence_index = len(targets) + 1
        targets.append(
            CalibrationPoseTarget(
                arm=arm,
                sequence_index=sequence_index,
                frame_index=int(data.get("frame_index", -1)),
                source_dir=pose_path.parent,
                pose=_pose_from_json(data, arm),
                timestamp_s=None if data.get("timestamp_s") is None else float(data["timestamp_s"]),
                pause_frame_range=data.get("pause_frame_range"),
            )
        )
    if not targets:
        raise FileNotFoundError(f"No pose.json files found under {root}")
    return targets


def load_pose_targets_from_summary(
    summary_json: str | Path,
    *,
    dataset_key: str,
    arm: str,
    source_root: str | Path,
) -> list[CalibrationPoseTarget]:
    summary_json = Path(summary_json)
    source_root = Path(source_root)
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if dataset_key not in summary:
        raise KeyError(f"{dataset_key!r} not found in {summary_json}")

    targets: list[CalibrationPoseTarget] = []
    for item in summary[dataset_key].get("items", []):
        pose_dict = item[f"{arm}_observation_pose"]
        pose = np.asarray([float(pose_dict[name]) for name in FLANGE_POSE_COMPONENTS], dtype=float)
        sequence_index = int(item["index"])
        frame_index = int(item["frame_index"])
        targets.append(
            CalibrationPoseTarget(
                arm=arm,
                sequence_index=sequence_index,
                frame_index=frame_index,
                source_dir=source_root / f"{sequence_index:02d}_frame_{frame_index:04d}",
                pose=pose,
                timestamp_s=None if item.get("timestamp_s") is None else float(item["timestamp_s"]),
                pause_frame_range=item.get("pause_frame_range"),
            )
        )
    if not targets:
        raise FileNotFoundError(f"No targets found for {dataset_key!r} in {summary_json}")
    return targets


def load_replay_targets(
    *,
    right_dir: str | Path,
    left_dir: str | Path,
    summary_json: str | Path | None = None,
) -> list[CalibrationPoseTarget]:
    if summary_json is not None and Path(summary_json).exists():
        return [
            *load_pose_targets_from_summary(
                summary_json,
                dataset_key=Path(right_dir).name,
                arm="right",
                source_root=right_dir,
            ),
            *load_pose_targets_from_summary(
                summary_json,
                dataset_key=Path(left_dir).name,
                arm="left",
                source_root=left_dir,
            ),
        ]
    return [
        *load_pose_targets(right_dir, arm="right"),
        *load_pose_targets(left_dir, arm="left"),
    ]


def validate_target_counts(
    targets: list[CalibrationPoseTarget],
    *,
    expected_right: int,
    expected_left: int,
) -> None:
    counts = {
        "right": sum(1 for target in targets if target.arm == "right"),
        "left": sum(1 for target in targets if target.arm == "left"),
    }
    expected = {"right": expected_right, "left": expected_left}
    mismatches = [
        f"{arm}: got {counts[arm]}, expected {expected[arm]}"
        for arm in ("right", "left")
        if expected[arm] > 0 and counts[arm] != expected[arm]
    ]
    if mismatches:
        raise ValueError("Unexpected calibration target count: " + "; ".join(mismatches))


def make_base_flange_features(*, front_shape: tuple[int, int, int], use_videos: bool) -> dict[str, dict]:
    image_dtype = "video" if use_videos else "image"
    state_feature = {
        "dtype": "float32",
        "shape": (len(EE_FLANGE_EULER_QUATERNION_NAMES),),
        "names": list(EE_FLANGE_EULER_QUATERNION_NAMES),
    }
    return {
        "action": dict(state_feature),
        "observation.state": dict(state_feature),
        "observation.images.front": {
            "dtype": image_dtype,
            "shape": front_shape,
            "names": ["height", "width", "channels"],
        },
    }


def flange_dict_from_poses(
    *,
    right_pose: list[float] | np.ndarray,
    left_pose: list[float] | np.ndarray,
    right_gripper_width: float,
    left_gripper_width: float,
) -> dict[str, float]:
    observation: dict[str, float] = {}
    for arm, pose in (("right", right_pose), ("left", left_pose)):
        pose_array = np.asarray(pose, dtype=float)
        if pose_array.shape != (6,):
            raise ValueError(f"{arm}_pose must have shape (6,), got {pose_array.shape}.")
        for component, value in zip(FLANGE_POSE_COMPONENTS, pose_array, strict=True):
            observation[f"{arm}_flange_{component}"] = float(value)
    observation["right_gripper_width"] = float(right_gripper_width)
    observation["left_gripper_width"] = float(left_gripper_width)
    return observation


def pose_from_flange_observation(observation: dict[str, Any], arm: str) -> np.ndarray:
    return np.asarray([float(observation[f"{arm}_flange_{component}"]) for component in FLANGE_POSE_COMPONENTS])


def flange_observation_to_euler_quaternion_state(
    observation: dict[str, Any],
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    values: list[float] = []
    for arm in ("right", "left"):
        pose = pose_from_flange_observation(observation, arm)
        quaternion_xyzw = Rotation.from_euler(euler_order, pose[3:6]).as_quat()
        values.extend(float(value) for value in pose)
        values.extend(float(value) for value in quaternion_xyzw)
    return np.asarray(values, dtype=np.float32)


def make_policy_frame(
    *,
    actual_observation: dict[str, Any],
    front_key: str,
    task: str,
    euler_order: str = "xyz",
) -> dict[str, Any]:
    if front_key not in actual_observation:
        raise KeyError(f"Front camera key {front_key!r} not found in robot observation.")
    state = flange_observation_to_euler_quaternion_state(actual_observation, euler_order=euler_order)
    return {
        "action": state.copy(),
        "observation.state": state,
        "observation.images.front": actual_observation[front_key],
        "task": task,
    }


def prepare_template_output_root(output_root: str | Path, *, overwrite: bool) -> Path:
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Set --overwrite=true if you want to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _template_pose_dict(pose: np.ndarray, *, euler_order: str) -> tuple[dict[str, float], dict[str, float]]:
    pose = np.asarray(pose, dtype=float)
    quaternion_xyzw = Rotation.from_euler(euler_order, pose[3:6]).as_quat()
    euler_pose = {component: float(value) for component, value in zip(FLANGE_POSE_COMPONENTS, pose, strict=True)}
    quaternion_pose = {
        "x": float(pose[0]),
        "y": float(pose[1]),
        "z": float(pose[2]),
        "qx": float(quaternion_xyzw[0]),
        "qy": float(quaternion_xyzw[1]),
        "qz": float(quaternion_xyzw[2]),
        "qw": float(quaternion_xyzw[3]),
    }
    return euler_pose, quaternion_pose


def _save_front_png(image: Any, output_path: Path) -> None:
    image_array = np.asarray(image)
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    Image.fromarray(image_array).save(output_path)


def write_template_capture(
    *,
    output_root: str | Path,
    target: CalibrationPoseTarget,
    actual_observation: dict[str, Any],
    front_key: str,
    euler_order: str,
) -> Path:
    if front_key not in actual_observation:
        raise KeyError(f"Front camera key {front_key!r} not found in robot observation.")

    output_root = Path(output_root)
    arm_root = output_root / f"{target.arm}_arm_stable_front_frames_actual"
    capture_dir = arm_root / f"{target.sequence_index:02d}_frame_{target.frame_index:04d}"
    capture_dir.mkdir(parents=True, exist_ok=False)

    _save_front_png(actual_observation[front_key], capture_dir / "front.png")

    actual_pose = pose_from_flange_observation(actual_observation, target.arm)
    euler_pose, quaternion_pose = _template_pose_dict(actual_pose, euler_order=euler_order)
    pose_json: dict[str, Any] = {
        "frame_index": int(target.frame_index),
        f"{target.arm}_observation_pose": euler_pose,
        f"{target.arm}_observation_pose_quaternion": quaternion_pose,
    }
    if target.timestamp_s is not None:
        pose_json["timestamp_s"] = float(target.timestamp_s)
    if target.pause_frame_range is not None:
        pose_json["pause_frame_range"] = dict(target.pause_frame_range)

    with (capture_dir / "pose.json").open("w", encoding="utf-8") as f:
        json.dump(pose_json, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return capture_dir


def set_move_p_mode(arm_runtime: Any) -> None:
    if arm_runtime.robot is None:
        raise RuntimeError(f"{arm_runtime.arm} arm is not connected.")
    arm_runtime.robot.set_motion_mode("p")


def move_p(arm_runtime: Any, pose: np.ndarray | list[float]) -> None:
    set_move_p_mode(arm_runtime)
    arm_runtime.robot.move_p(np.asarray(pose, dtype=float).tolist())


def set_move_js_mode(arm_runtime: Any) -> None:
    if arm_runtime.robot is None:
        raise RuntimeError(f"{arm_runtime.arm} arm is not connected.")
    arm_runtime.robot.set_motion_mode("js")


def limit_joint_step(current: np.ndarray, target: np.ndarray, max_step_rad: float | None) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError(f"current and target joint shapes differ: {current.shape} vs {target.shape}.")
    if max_step_rad is None or max_step_rad <= 0:
        return target.copy()
    delta = np.clip(target - current, -max_step_rad, max_step_rad)
    return current + delta


def send_limited_joint_target(
    arm_runtime: Any,
    target_joints: np.ndarray | list[float],
    *,
    max_joint_step_rad: float | None,
    dt_s: float,
    sleep_fn=time.sleep,
) -> dict[str, int]:
    target = np.asarray(target_joints, dtype=float)
    if target.shape != (7,):
        raise ValueError(f"target_joints must have shape (7,), got {target.shape}.")
    current = np.asarray(arm_runtime.read_joints(), dtype=float)
    if current.shape != (7,):
        raise ValueError(f"Current joints must have shape (7,), got {current.shape}.")

    set_move_js_mode(arm_runtime)
    steps = 0
    while True:
        command = limit_joint_step(current, target, max_joint_step_rad)
        arm_runtime.robot.move_js(command.tolist())
        steps += 1
        if np.allclose(command, target, rtol=0.0, atol=1e-9):
            break
        current = command
        if dt_s > 0:
            sleep_fn(dt_s)
    return {"steps": steps}


def move_arm_to_pose(
    arm_runtime: Any,
    target_pose: np.ndarray | list[float],
    *,
    motion_backend: str,
    ik: Any | None,
    smooth_move: bool,
    smooth_move_time_s: float,
    smooth_move_dt_s: float,
    smooth_min_steps: int,
    max_joint_step_rad: float,
    joint_command_dt_s: float,
    euler_order: str,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    if motion_backend == "sdk_move_p":
        smooth_move_p(
            arm_runtime,
            target_pose,
            enabled=smooth_move,
            duration_s=smooth_move_time_s,
            dt_s=smooth_move_dt_s,
            min_steps=smooth_min_steps,
            euler_order=euler_order,
            sleep_fn=sleep_fn,
        )
        return {"backend": motion_backend, "joint_steps": 0}
    if motion_backend != "curobo_move_js":
        raise ValueError(
            f"Unsupported motion_backend={motion_backend!r}. "
            "Use 'curobo_move_js' or 'sdk_move_p'."
        )
    if ik is None:
        raise ValueError("motion_backend='curobo_move_js' requires an IK solver.")

    current_joints = np.asarray(arm_runtime.read_joints(), dtype=float)
    target_joints = np.asarray(ik.solve(target_pose, current_joints=current_joints), dtype=float)
    joint_stats = send_limited_joint_target(
        arm_runtime,
        target_joints,
        max_joint_step_rad=max_joint_step_rad,
        dt_s=joint_command_dt_s,
        sleep_fn=sleep_fn,
    )
    return {"backend": motion_backend, "joint_steps": joint_stats["steps"], "target_joints": target_joints}


def smooth_move_p(
    arm_runtime: Any,
    target_pose: np.ndarray | list[float],
    *,
    enabled: bool,
    duration_s: float,
    dt_s: float,
    min_steps: int,
    euler_order: str,
    sleep_fn=time.sleep,
) -> None:
    target = np.asarray(target_pose, dtype=float)
    if target.shape != (6,):
        raise ValueError(f"target_pose must have shape (6,), got {target.shape}.")
    if not enabled:
        move_p(arm_runtime, target)
        return

    current = np.asarray(arm_runtime.read_flange_pose(), dtype=float)
    if current.shape != (6,):
        raise ValueError(f"Current flange pose must have shape (6,), got {current.shape}.")

    steps_from_time = int(np.ceil(max(duration_s, 0.0) / max(dt_s, 1e-6)))
    steps = max(int(min_steps), steps_from_time, 1)
    position_delta = target[:3] - current[:3]
    rotations = Rotation.from_euler(euler_order, np.vstack([current[3:6], target[3:6]]))
    slerp = Slerp([0.0, 1.0], rotations)

    set_move_p_mode(arm_runtime)
    for step in range(1, steps + 1):
        ratio = step / steps
        position = current[:3] + ratio * position_delta
        euler = slerp([ratio]).as_euler(euler_order)[0]
        waypoint = np.concatenate([position, euler])
        arm_runtime.robot.move_p(waypoint.tolist())
        if step < steps:
            sleep_fn(dt_s)


def wait_until_arm_stable(
    arm_runtime: Any,
    *,
    timeout_s: float,
    poll_s: float,
    stable_samples: int,
    position_threshold_m: float,
    rotation_threshold_rad: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    stable_count = 0
    previous = arm_runtime.read_flange_pose()
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        current = arm_runtime.read_flange_pose()
        delta = np.abs(current - previous)
        if float(np.max(delta[:3])) <= position_threshold_m and float(np.max(delta[3:])) <= rotation_threshold_rad:
            stable_count += 1
            if stable_count >= stable_samples:
                return True
        else:
            stable_count = 0
        previous = current
    return False


def make_curobo_ik_by_arm(cfg: ReplayCalibrationPosesConfig) -> dict[str, NeroCuroboArmIK] | None:
    if cfg.motion_backend == "sdk_move_p":
        return None
    if cfg.motion_backend != "curobo_move_js":
        raise ValueError(
            f"Unsupported motion_backend={cfg.motion_backend!r}. "
            "Use 'curobo_move_js' or 'sdk_move_p'."
        )
    logger.info("Initializing cuRobo IK backend with robot=%s", cfg.curobo_robot)
    return {
        "right": NeroCuroboArmIK(
            robot_file=cfg.curobo_robot,
            euler_order=cfg.euler_order,
            num_seeds=cfg.curobo_num_seeds,
            position_threshold=cfg.curobo_position_threshold,
            rotation_threshold=cfg.curobo_rotation_threshold,
            device=cfg.curobo_device,
        ),
        "left": NeroCuroboArmIK(
            robot_file=cfg.curobo_robot,
            euler_order=cfg.euler_order,
            num_seeds=cfg.curobo_num_seeds,
            position_threshold=cfg.curobo_position_threshold,
            rotation_threshold=cfg.curobo_rotation_threshold,
            device=cfg.curobo_device,
        ),
    }


def _move_arm_gripper(arm_runtime: Any, *, width: float, force: float) -> None:
    if arm_runtime.end_effector is None:
        raise RuntimeError(f"{arm_runtime.arm} gripper is not initialized.")
    arm_runtime.end_effector.move_gripper_m(value=float(width), force=float(force))
    arm_runtime._gripper_width = float(width)


def prepare_grippers_for_board(
    robot: NeroDualRobot,
    *,
    enabled: bool,
    open_width: float,
    close_width: float,
    force: float,
    settle_s: float,
    input_fn=input,
    sleep_fn=time.sleep,
) -> None:
    if not enabled:
        return
    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=open_width, force=force)
    sleep_fn(settle_s)
    input_fn("双夹爪已打开。请放入标定板并扶稳，准备好后按 ENTER 闭合夹爪并开始后续流程。")
    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=close_width, force=force)
    sleep_fn(settle_s)


def prompt_before_capture(
    *,
    enabled: bool,
    target_arm: str,
    target_index: int,
    total_targets: int,
    input_fn=input,
) -> None:
    if not enabled:
        return
    input_fn(
        f"[{target_index}/{total_targets}] {target_arm} 已停稳。"
        "请确认画面和标定板状态，按 ENTER 采集 front 图像并读取实际末端位姿。"
    )


def move_dual_to_ready(
    robot: NeroDualRobot,
    cfg: ReplayCalibrationPosesConfig,
    *,
    ik_by_arm: dict[str, Any] | None,
    sleep_fn=time.sleep,
) -> None:
    del ik_by_arm
    send_limited_joint_target(
        robot.right,
        cfg.ready_right_joints,
        max_joint_step_rad=cfg.max_joint_step_rad,
        dt_s=cfg.joint_command_dt_s,
        sleep_fn=sleep_fn,
    )
    send_limited_joint_target(
        robot.left,
        cfg.ready_left_joints,
        max_joint_step_rad=cfg.max_joint_step_rad,
        dt_s=cfg.joint_command_dt_s,
        sleep_fn=sleep_fn,
    )
    sleep_fn(cfg.ready_wait_s)
    wait_until_arm_stable(
        robot.right,
        timeout_s=cfg.stable_timeout_s,
        poll_s=cfg.stable_poll_s,
        stable_samples=cfg.stable_samples,
        position_threshold_m=cfg.stable_position_threshold_m,
        rotation_threshold_rad=cfg.stable_rotation_threshold_rad,
    )
    wait_until_arm_stable(
        robot.left,
        timeout_s=cfg.stable_timeout_s,
        poll_s=cfg.stable_poll_s,
        stable_samples=cfg.stable_samples,
        position_threshold_m=cfg.stable_position_threshold_m,
        rotation_threshold_rad=cfg.stable_rotation_threshold_rad,
    )


def move_arm_to_replay_ready(
    robot: NeroDualRobot,
    cfg: ReplayCalibrationPosesConfig,
    *,
    arm: str,
    sleep_fn=time.sleep,
) -> None:
    if arm == "right":
        arm_runtime = robot.right
        target_joints = cfg.replay_ready_right_joints
    elif arm == "left":
        arm_runtime = robot.left
        target_joints = cfg.replay_ready_left_joints
    else:
        raise ValueError(f"arm must be 'right' or 'left', got {arm!r}.")

    send_limited_joint_target(
        arm_runtime,
        target_joints,
        max_joint_step_rad=cfg.max_joint_step_rad,
        dt_s=cfg.joint_command_dt_s,
        sleep_fn=sleep_fn,
    )
    sleep_fn(cfg.ready_wait_s)
    wait_until_arm_stable(
        arm_runtime,
        timeout_s=cfg.stable_timeout_s,
        poll_s=cfg.stable_poll_s,
        stable_samples=cfg.stable_samples,
        position_threshold_m=cfg.stable_position_threshold_m,
        rotation_threshold_rad=cfg.stable_rotation_threshold_rad,
    )


def prepare_target_arm_for_replay(
    robot: NeroDualRobot,
    cfg: ReplayCalibrationPosesConfig,
    *,
    arm: str,
    input_fn=input,
    sleep_fn=time.sleep,
) -> None:
    move_arm_to_replay_ready(robot, cfg, arm=arm, sleep_fn=sleep_fn)
    input_fn(f"{arm} 臂已到标定回放 ready 位姿。按 ENTER 开始该臂回放。")


def prepare_ready_and_grip_board(
    robot: NeroDualRobot,
    cfg: ReplayCalibrationPosesConfig,
    *,
    ik_by_arm: dict[str, Any] | None,
    input_fn=input,
    sleep_fn=time.sleep,
) -> None:
    if not cfg.prepare_gripper:
        input_fn("即将移动双臂到默认 ready 位姿。请确认机械臂周围安全，然后按 ENTER 开始。")
        move_dual_to_ready(robot, cfg, ik_by_arm=ik_by_arm, sleep_fn=sleep_fn)
        return

    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=cfg.gripper_open_width, force=cfg.gripper_force)
    sleep_fn(cfg.gripper_settle_s)

    print("双夹爪已打开，将先平滑移动双臂到默认 ready 位姿。")
    move_dual_to_ready(robot, cfg, ik_by_arm=ik_by_arm, sleep_fn=sleep_fn)

    input_fn("双臂已到默认 ready 位姿。请放入标定板并扶稳，准备好后按 ENTER 闭合夹爪。")
    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=cfg.gripper_close_width, force=cfg.gripper_force)
    sleep_fn(cfg.gripper_settle_s)


def prepare_between_arms_board_swap(
    robot: NeroDualRobot,
    cfg: ReplayCalibrationPosesConfig,
    *,
    ik_by_arm: dict[str, Any] | None,
    input_fn=input,
    sleep_fn=time.sleep,
) -> None:
    move_dual_to_ready(robot, cfg, ik_by_arm=ik_by_arm, sleep_fn=sleep_fn)
    if not cfg.prepare_gripper:
        input_fn("双臂已回到默认 ready 位姿。请重新放置标定板，准备好后按 ENTER 开始左臂目标。")
        return

    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=cfg.gripper_open_width, force=cfg.gripper_force)
    sleep_fn(cfg.gripper_settle_s)

    input_fn("双臂已回到默认 ready 位姿且夹爪已打开。请重新放置标定板，准备好后按 ENTER 闭合夹爪并开始左臂目标。")
    for arm in (robot.right, robot.left):
        _move_arm_gripper(arm, width=cfg.gripper_close_width, force=cfg.gripper_force)
    sleep_fn(cfg.gripper_settle_s)


def create_dataset(
    *,
    cfg: ReplayCalibrationPosesConfig,
    robot: NeroDualRobot,
    front_shape: tuple[int, int, int],
) -> LeRobotDataset:
    if cfg.dataset.root is not None and Path(cfg.dataset.root).exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {cfg.dataset.root}. "
                "Set --overwrite=true if you want to replace it."
            )
        shutil.rmtree(cfg.dataset.root)

    features = make_base_flange_features(front_shape=front_shape, use_videos=cfg.dataset.video)
    cfg.dataset.stamp_repo_id()
    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        features=features,
        root=cfg.dataset.root,
        robot_type=robot.name,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        camera_encoder=cfg.dataset.camera_encoder,
        encoder_threads=cfg.dataset.encoder_threads,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
    )


def _print_target(target: CalibrationPoseTarget, index: int, total: int) -> None:
    pose_text = ", ".join(f"{value:.6f}" for value in target.pose)
    print(
        f"[{index}/{total}] {target.arm} arm target "
        f"{target.sequence_index:02d}, source frame {target.frame_index}: [{pose_text}]"
    )


@parser.wrap()
def replay_calibration_poses(cfg: ReplayCalibrationPosesConfig) -> None:
    init_logging()
    register_third_party_plugins()
    logging.info(pformat(asdict(cfg)))

    if not isinstance(cfg.robot, NeroDualRobotConfig):
        raise TypeError(f"nero-replay-calibration-poses requires --robot.type=nero_dual, got {cfg.robot.type!r}.")

    targets = load_replay_targets(
        right_dir=cfg.right_poses_dir,
        left_dir=cfg.left_poses_dir,
        summary_json=cfg.targets_summary_json,
    )
    validate_target_counts(
        targets,
        expected_right=cfg.expected_right_targets,
        expected_left=cfg.expected_left_targets,
    )
    if cfg.dry_run:
        print(f"Dry run: loaded {len(targets)} targets. No robot motion or dataset write will be performed.")
        for index, target in enumerate(targets, start=1):
            _print_target(target, index, len(targets))
        return

    ik_by_arm = make_curobo_ik_by_arm(cfg)
    robot = NeroDualRobot(cfg.robot)

    try:
        robot.connect()
        if cfg.front_camera_key not in robot.cameras:
            raise KeyError(
                f"Robot has no camera named {cfg.front_camera_key!r}. "
                f"Available cameras: {sorted(robot.cameras)}"
            )
        output_root = prepare_template_output_root(cfg.dataset.root, overwrite=cfg.overwrite)

        def replay_target(target: CalibrationPoseTarget, index: int, *, has_next_in_same_arm: bool) -> None:
            _print_target(target, index, len(targets))
            arm_runtime = robot.arms[target.arm]
            move_stats = move_arm_to_pose(
                arm_runtime,
                target.pose,
                motion_backend=cfg.motion_backend,
                ik=None if ik_by_arm is None else ik_by_arm[target.arm],
                smooth_move=cfg.smooth_move,
                smooth_move_time_s=cfg.smooth_move_time_s,
                smooth_move_dt_s=cfg.smooth_move_dt_s,
                smooth_min_steps=cfg.smooth_min_steps,
                max_joint_step_rad=cfg.max_joint_step_rad,
                joint_command_dt_s=cfg.joint_command_dt_s,
                euler_order=cfg.euler_order,
            )
            if cfg.motion_backend == "curobo_move_js":
                logger.info("%s target %s joint steps: %s", target.arm, index, move_stats["joint_steps"])
            time.sleep(cfg.move_settle_s)
            stable = wait_until_arm_stable(
                arm_runtime,
                timeout_s=cfg.stable_timeout_s,
                poll_s=cfg.stable_poll_s,
                stable_samples=cfg.stable_samples,
                position_threshold_m=cfg.stable_position_threshold_m,
                rotation_threshold_rad=cfg.stable_rotation_threshold_rad,
            )
            if not stable:
                logger.warning("%s target %s did not satisfy stability threshold before timeout.", target.arm, index)
                print("警告：稳定性等待超时，将仍然采集当前实际位姿。")

            prompt_before_capture(
                enabled=cfg.prompt_before_capture,
                target_arm=target.arm,
                target_index=index,
                total_targets=len(targets),
            )
            actual_observation = robot.get_flange_observation()
            capture_dir = write_template_capture(
                output_root=output_root,
                target=target,
                actual_observation=actual_observation,
                front_key=cfg.front_camera_key,
                euler_order=cfg.euler_order,
            )
            actual_pose = pose_from_flange_observation(actual_observation, target.arm)
            actual_text = ", ".join(f"{value:.6f}" for value in actual_pose)
            print(f"已采集 front 图像和实际 {target.arm} 末端位姿: [{actual_text}]")
            print(f"已保存模板帧目录: {capture_dir}")

            if cfg.prompt_between_targets and has_next_in_same_arm:
                input("按 ENTER 执行下一个位姿。")

        right_targets = [target for target in targets if target.arm == "right"]
        left_targets = [target for target in targets if target.arm == "left"]

        prepare_ready_and_grip_board(robot, cfg, ik_by_arm=ik_by_arm)
        prepare_target_arm_for_replay(robot, cfg, arm="right")

        for arm_index, target in enumerate(right_targets, start=1):
            replay_target(target, targets.index(target) + 1, has_next_in_same_arm=arm_index < len(right_targets))

        if cfg.reset_to_ready_between_arms:
            print("右臂目标已回放完毕，先将双臂回到原始 ready 位姿并打开夹爪，再开始左臂。")
            prepare_between_arms_board_swap(robot, cfg, ik_by_arm=ik_by_arm)
        prepare_target_arm_for_replay(robot, cfg, arm="left")

        for arm_index, target in enumerate(left_targets, start=1):
            replay_target(target, targets.index(target) + 1, has_next_in_same_arm=arm_index < len(left_targets))

        print("左臂目标已回放完毕，双臂回到原始 ready 位姿。")
        move_dual_to_ready(robot, cfg, ik_by_arm=ik_by_arm)
        print(f"已保存 {len(targets)} 帧模板数据到: {output_root}")
    finally:
        robot.disconnect()


def main() -> None:
    replay_calibration_poses()


if __name__ == "__main__":
    main()
