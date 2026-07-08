import logging
import pickle  # nosec
from queue import Queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import numpy as np
from PIL import Image
import grpc
from scipy.spatial.transform import Rotation
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.transport import services_pb2
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .config_nero import NeroDualRobotConfig
from .curobo_ik_adapter import NeroDualCuroboIKAdapter
from .ee_local_se3_adapter import EE_LOCAL_SE3_ACTION_NAMES, NeroEELocalSE3Adapter
from .mapping import namespaced_gripper_name, namespaced_joint_names
from .prepare_sync import smooth_takeover_commands
from .robot_nero_dual import NeroDualRobot
from .trace import NeroInferenceTraceConfig, NeroInferenceTracer

logger = logging.getLogger(__name__)
DEFAULT_HIGH_RATE_HZ = 180.0
DEFAULT_HIGH_RATE_DT_S = 1.0 / DEFAULT_HIGH_RATE_HZ

DEFAULT_LOCAL_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs/train/pi05_bus_table_01_rel_ckpts/checkpoints/005000/pretrained_model"
)

DEFAULT_DUAL_ACTION_NAMES = (
    *namespaced_joint_names("right"),
    *namespaced_joint_names("left"),
    namespaced_gripper_name("right"),
    namespaced_gripper_name("left"),
)
DEFAULT_RIGHT_HANDEYE_CAMERA_TO_BASE_YAML = (
    "/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_right_arm_tsai.yml"
)
DEFAULT_LEFT_HANDEYE_CAMERA_TO_BASE_YAML = (
    "/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_left_arm_tsai.yml"
)


def default_ready_pose() -> dict[str, float]:
    return {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}


@dataclass
class NeroAsyncSafetyConfig:
    fixed_ready_pose: dict[str, float] = field(default_factory=default_ready_pose)
    takeover_time_s: float = 4.0
    takeover_dt_s: float = 0.02
    ready_tolerance_rad: float = 0.05
    max_policy_step_rad: float = 0.08
    max_gripper_step_m: float = 0.01
    max_ee_position_step_m: float = 0.0
    max_ee_rotation_step_rad: float = 0.0
    high_rate_control: bool = True
    high_rate_dt_s: float = DEFAULT_HIGH_RATE_DT_S
    max_executor_step_rad: float = 0.02
    max_executor_gripper_step_m: float = 0.005
    high_rate_interpolation_steps: int = 6
    high_rate_gripper_period: int = 6
    high_rate_gripper_epsilon_m: float = 1e-4
    high_rate_overrun_log_every: int = 100
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.1
    dry_run: bool = False
    recover_on_ik_failure: bool = False
    ik_failure_recovery_wait_for_enter: bool = True

    def __post_init__(self) -> None:
        if self.high_rate_dt_s <= 0:
            raise ValueError(f"high_rate_dt_s must be positive, got {self.high_rate_dt_s}.")
        if self.max_ee_position_step_m < 0:
            raise ValueError(
                f"max_ee_position_step_m must be non-negative, got {self.max_ee_position_step_m}."
            )
        if self.max_ee_rotation_step_rad < 0:
            raise ValueError(
                f"max_ee_rotation_step_rad must be non-negative, got {self.max_ee_rotation_step_rad}."
            )
        if self.high_rate_interpolation_steps <= 0:
            raise ValueError(
                f"high_rate_interpolation_steps must be positive, got {self.high_rate_interpolation_steps}."
            )
        if self.high_rate_gripper_period <= 0:
            raise ValueError(
                f"high_rate_gripper_period must be positive, got {self.high_rate_gripper_period}."
            )
        if self.high_rate_gripper_epsilon_m < 0:
            raise ValueError(
                f"high_rate_gripper_epsilon_m must be non-negative, got {self.high_rate_gripper_epsilon_m}."
            )
        if self.high_rate_overrun_log_every <= 0:
            raise ValueError(
                f"high_rate_overrun_log_every must be positive, got {self.high_rate_overrun_log_every}."
            )


@dataclass
class NeroDebugImageSaveConfig:
    enabled: bool = False
    dir: str = "/tmp/nero_async_client_images"
    every_n: int = 1
    max_frames: int = 0

    def __post_init__(self) -> None:
        if self.every_n <= 0:
            raise ValueError(f"debug_save_images.every_n must be positive, got {self.every_n}.")
        if self.max_frames < 0:
            raise ValueError(f"debug_save_images.max_frames must be non-negative, got {self.max_frames}.")


@dataclass
class NeroDebugVideoSaveConfig:
    enabled: bool = False
    dir: str = ""
    fps: int = 30
    every_n: int = 1
    max_frames: int = 0
    keep_frames: bool = True
    encoder_threads: int | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"debug_save_videos.fps must be positive, got {self.fps}.")
        if self.every_n <= 0:
            raise ValueError(f"debug_save_videos.every_n must be positive, got {self.every_n}.")
        if self.max_frames < 0:
            raise ValueError(f"debug_save_videos.max_frames must be non-negative, got {self.max_frames}.")
        if self.encoder_threads is not None and self.encoder_threads <= 0:
            raise ValueError(
                f"debug_save_videos.encoder_threads must be positive or None, got {self.encoder_threads}."
            )


@dataclass
class NeroAsyncClientConfig:
    robot: RobotConfig = field(default_factory=NeroDualRobotConfig)
    server_address: str = "127.0.0.1:8080"
    policy_type: str = "pi05"
    policy_path: str = str(DEFAULT_LOCAL_POLICY_PATH)
    task: str = "pick up red block"
    policy_device: str = "cuda"
    client_device: str = "cpu"
    fps: int = 30
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    debug_visualize_queue_size: bool = False
    wait_for_enter: bool = True
    keyboard_stop: bool = True
    action_mode: str = "joint"
    right_handeye_camera_to_base_yaml: str = DEFAULT_RIGHT_HANDEYE_CAMERA_TO_BASE_YAML
    left_handeye_camera_to_base_yaml: str = DEFAULT_LEFT_HANDEYE_CAMERA_TO_BASE_YAML
    ee_base_or_head_x: float = 0.0
    ee_base_or_head_y: float = 0.0
    ee_euler_order: str = "xyz"
    curobo_robot_file: str = "nero_custom.yml"
    curobo_num_seeds: int = 32
    curobo_position_threshold: float = 0.01
    curobo_rotation_threshold: float = 0.05
    curobo_device: str = "cuda"
    safety: NeroAsyncSafetyConfig = field(default_factory=NeroAsyncSafetyConfig)
    debug_save_images: NeroDebugImageSaveConfig = field(default_factory=NeroDebugImageSaveConfig)
    debug_save_videos: NeroDebugVideoSaveConfig = field(default_factory=NeroDebugVideoSaveConfig)
    trace: NeroInferenceTraceConfig = field(default_factory=NeroInferenceTraceConfig)

    def __post_init__(self) -> None:
        if self.action_mode not in {"joint", "ee_local_se3"}:
            raise ValueError(
                f"action_mode must be 'joint' or 'ee_local_se3', got {self.action_mode!r}."
            )


class ObservationImageSaver:
    def __init__(self, config: NeroDebugImageSaveConfig):
        self.config = config
        self._observation_count = 0
        self._saved_frame_count = 0
        self._output_dir = Path(config.dir)

    @staticmethod
    def _image_from_array(value: Any) -> Image.Image:
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {image.shape}.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        return Image.fromarray(image)

    def maybe_save(self, observation: RobotObservation) -> None:
        if not self.config.enabled:
            return

        obs_idx = self._observation_count
        self._observation_count += 1
        if obs_idx % self.config.every_n != 0:
            return
        if self.config.max_frames > 0 and self._saved_frame_count >= self.config.max_frames:
            return

        image_items = [
            (name, value)
            for name, value in observation.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        ]
        if not image_items:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        frame_idx = self._saved_frame_count
        for name, value in image_items:
            image = self._image_from_array(value)
            image.save(self._output_dir / f"{frame_idx:06d}_{name}.png")
        self._saved_frame_count += 1


class ObservationVideoSaver:
    def __init__(self, config: NeroDebugVideoSaveConfig):
        self.config = config
        self._observation_count = 0
        self._saved_frame_count = 0
        self._closed = False
        self._camera_names: set[str] = set()

    @property
    def output_dir(self) -> Path:
        return Path(self.config.dir)

    def _frame_dir(self, camera_name: str) -> Path:
        return self.output_dir / "frames" / camera_name

    def _video_path(self, camera_name: str) -> Path:
        return (
            self.output_dir
            / "videos"
            / f"observation.images.{camera_name}"
            / "chunk-000"
            / "file-000.mp4"
        )

    def maybe_save(self, observation: RobotObservation) -> None:
        if not self.config.enabled:
            return

        obs_idx = self._observation_count
        self._observation_count += 1
        if obs_idx % self.config.every_n != 0:
            return
        if self.config.max_frames > 0 and self._saved_frame_count >= self.config.max_frames:
            return

        image_items = [
            (name, value)
            for name, value in observation.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        ]
        if not image_items:
            return

        frame_idx = self._saved_frame_count
        for name, value in image_items:
            image = ObservationImageSaver._image_from_array(value)
            frame_dir = self._frame_dir(name)
            frame_dir.mkdir(parents=True, exist_ok=True)
            image.save(frame_dir / f"frame-{frame_idx:06d}.png")
            self._camera_names.add(name)
        self._saved_frame_count += 1

    def close(self) -> None:
        if self._closed or not self.config.enabled:
            return
        self._closed = True
        if self._saved_frame_count == 0:
            return

        camera_encoder = VideoEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=12)
        for camera_name in sorted(self._camera_names):
            frame_dir = self._frame_dir(camera_name)
            video_path = self._video_path(camera_name)
            encode_video_frames(
                frame_dir,
                video_path,
                fps=self.config.fps,
                camera_encoder=camera_encoder,
                encoder_threads=self.config.encoder_threads,
                overwrite=True,
            )
            logger.info("Saved Nero inference video %s", video_path)


def _require_pose_keys(pose: dict[str, float]) -> None:
    missing = [name for name in DEFAULT_DUAL_ACTION_NAMES if name not in pose]
    if missing:
        raise KeyError(f"Missing Nero pose keys: {missing}")


def action_vector_from_pose(pose: dict[str, float]) -> np.ndarray:
    _require_pose_keys(pose)
    return np.asarray([float(pose[name]) for name in DEFAULT_DUAL_ACTION_NAMES], dtype=float)


def _arm_joint_vector(pose: dict[str, float], arm: str) -> np.ndarray:
    return np.asarray([float(pose[name]) for name in namespaced_joint_names(arm)], dtype=float)


def _pose_from_arm_vectors(
    right_joints: np.ndarray,
    left_joints: np.ndarray,
    pose: dict[str, float],
) -> dict[str, float]:
    action = {
        name: float(value) for name, value in zip(namespaced_joint_names("right"), right_joints, strict=True)
    }
    action.update(
        {name: float(value) for name, value in zip(namespaced_joint_names("left"), left_joints, strict=True)}
    )
    action[namespaced_gripper_name("right")] = float(pose[namespaced_gripper_name("right")])
    action[namespaced_gripper_name("left")] = float(pose[namespaced_gripper_name("left")])
    return action


def sync_to_fixed_ready_pose(
    robot: NeroDualRobot,
    pose: dict[str, float],
    *,
    takeover_time_s: float,
    takeover_dt_s: float,
    tolerance_rad: float,
) -> float:
    _require_pose_keys(pose)
    right_current = robot.right.read_joints()
    left_current = robot.left.read_joints()
    right_target = _arm_joint_vector(pose, "right")
    left_target = _arm_joint_vector(pose, "left")

    original = {
        "right": (robot.right.config.command.alpha, robot.right.config.command.max_step_rad),
        "left": (robot.left.config.command.alpha, robot.left.config.command.max_step_rad),
    }
    try:
        robot.right.config.command.alpha = 1.0
        robot.left.config.command.alpha = 1.0
        robot.right.config.command.max_step_rad = 0.0
        robot.left.config.command.max_step_rad = 0.0
        steps = max(int(round(takeover_time_s / takeover_dt_s)), 1)
        for right_cmd, left_cmd in zip(
            smooth_takeover_commands(right_current, right_target, steps),
            smooth_takeover_commands(left_current, left_target, steps),
            strict=True,
        ):
            robot.send_action(_pose_from_arm_vectors(right_cmd, left_cmd, pose))
            precise_sleep(takeover_dt_s)
    finally:
        robot.right.config.command.alpha, robot.right.config.command.max_step_rad = original["right"]
        robot.left.config.command.alpha, robot.left.config.command.max_step_rad = original["left"]

    right_error = float(np.max(np.abs(right_target - robot.right.read_joints())))
    left_error = float(np.max(np.abs(left_target - robot.left.read_joints())))
    max_error = max(right_error, left_error)
    if max_error > tolerance_rad:
        raise RuntimeError(
            f"Nero ready pose sync error {max_error:.4f} rad exceeds tolerance {tolerance_rad:.4f} rad."
        )
    return max_error


def limit_action_step(
    target: dict[str, float],
    last: dict[str, float],
    *,
    max_joint_step_rad: float,
    max_gripper_step_m: float,
    gripper_min_m: float,
    gripper_max_m: float,
) -> dict[str, float]:
    _require_pose_keys(target)
    _require_pose_keys(last)
    limited: dict[str, float] = {}
    for arm in ("right", "left"):
        for name in namespaced_joint_names(arm):
            delta = np.clip(
                float(target[name]) - float(last[name]),
                -max_joint_step_rad,
                max_joint_step_rad,
            )
            limited[name] = float(last[name]) + float(delta)

        gripper_name = namespaced_gripper_name(arm)
        gripper_target = float(np.clip(float(target[gripper_name]), gripper_min_m, gripper_max_m))
        delta = np.clip(
            gripper_target - float(last[gripper_name]),
            -max_gripper_step_m,
            max_gripper_step_m,
        )
        limited[gripper_name] = float(np.clip(float(last[gripper_name]) + float(delta), gripper_min_m, gripper_max_m))
    return limited


def _limit_vector_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return np.asarray(target, dtype=float).copy()
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm == 0.0:
        return np.asarray(target, dtype=float).copy()
    return np.asarray(current, dtype=float) + delta * (max_step / norm)


def _limit_rotvec_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return np.asarray(target, dtype=float).copy()

    current_rotation = Rotation.from_rotvec(np.asarray(current, dtype=float))
    target_rotation = Rotation.from_rotvec(np.asarray(target, dtype=float))
    delta_rotation = target_rotation * current_rotation.inv()
    delta_rotvec = delta_rotation.as_rotvec()
    delta_angle = float(np.linalg.norm(delta_rotvec))
    if delta_angle <= max_step or delta_angle == 0.0:
        return np.asarray(target, dtype=float).copy()

    limited_delta = Rotation.from_rotvec(delta_rotvec * (max_step / delta_angle))
    return (limited_delta * current_rotation).as_rotvec()


def limit_ee_policy_action_step(
    target: np.ndarray,
    current: np.ndarray,
    *,
    max_position_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)
    if target.shape != (len(EE_LOCAL_SE3_ACTION_NAMES),):
        raise ValueError(f"target EE action must have shape ({len(EE_LOCAL_SE3_ACTION_NAMES)},), got {target.shape}.")
    if current.shape != (len(EE_LOCAL_SE3_ACTION_NAMES),):
        raise ValueError(f"current EE state must have shape ({len(EE_LOCAL_SE3_ACTION_NAMES)},), got {current.shape}.")

    limited = target.copy()
    for offset in (0, 7):
        limited[offset : offset + 3] = _limit_vector_step(
            target[offset : offset + 3],
            current[offset : offset + 3],
            max_position_step_m,
        )
        limited[offset + 3 : offset + 6] = _limit_rotvec_step(
            target[offset + 3 : offset + 6],
            current[offset + 3 : offset + 6],
            max_rotation_step_rad,
        )
    return limited


def _is_ik_failure_exception(exc: BaseException) -> bool:
    """Detect the IK failures that are safe to route through ready-pose recovery."""
    message = str(exc)
    return isinstance(exc, RuntimeError) and (
        "IK failed" in message or "IK error too large" in message
    )


def _should_drop_action_chunk_for_ik_recovery(client: Any, *, request_generation: int) -> bool:
    with client._ik_recovery_lock:
        current_generation = client._ik_recovery_generation
        recovery_active = client._ik_recovery_active.is_set()
    return recovery_active or request_generation != current_generation


def _clear_async_action_state(client: Any) -> None:
    with client.action_queue_lock:
        client.action_queue = Queue()
    with client.latest_action_lock:
        client.latest_action = -1
    client.action_chunk_size = -1
    client.must_go.set()


def recover_nero_async_client_from_ik_failure(
    client: Any,
    safety: NeroAsyncSafetyConfig,
    *,
    tracer: NeroInferenceTracer | None = None,
    wait_for_resume: Callable[[], None] | None = None,
    error: BaseException | None = None,
) -> None:
    with client._ik_recovery_lock:
        client._ik_recovery_generation += 1
        client._ik_recovery_active.set()

    ready_error: float | None = None
    try:
        if tracer is not None:
            tracer.record(
                "ik_failure_recovery_start",
                {
                    "error_type": type(error).__name__ if error is not None else None,
                    "error": str(error) if error is not None else None,
                    "dry_run": safety.dry_run,
                },
            )

        stop_executor = getattr(client.robot, "stop_high_rate_executor", None)
        if callable(stop_executor):
            stop_executor()

        _clear_async_action_state(client)
        if safety.dry_run:
            client.robot.set_last_action(safety.fixed_ready_pose)
        else:
            inner_robot = getattr(client.robot, "robot", client.robot)
            ready_error = sync_to_fixed_ready_pose(
                inner_robot,
                safety.fixed_ready_pose,
                takeover_time_s=safety.takeover_time_s,
                takeover_dt_s=safety.takeover_dt_s,
                tolerance_rad=safety.ready_tolerance_rad,
            )
            client.robot.set_last_action(safety.fixed_ready_pose)

        _clear_async_action_state(client)
        client.stub.Ready(services_pb2.Empty())

        if tracer is not None:
            tracer.record(
                "ik_failure_recovery_ready",
                {
                    "ready_error": ready_error,
                    "wait_for_resume": wait_for_resume is not None,
                },
            )

        if wait_for_resume is not None:
            wait_for_resume()

        _clear_async_action_state(client)
        start_executor = getattr(client.robot, "start_high_rate_executor", None)
        if callable(start_executor):
            start_executor()

        if tracer is not None:
            tracer.record("ik_failure_recovery_complete", {"ready_error": ready_error})
    except Exception as recovery_exc:
        if tracer is not None:
            tracer.record(
                "ik_failure_recovery_failed",
                {
                    "error_type": type(recovery_exc).__name__,
                    "error": str(recovery_exc),
                },
            )
        raise
    finally:
        with client._ik_recovery_lock:
            client._ik_recovery_generation += 1
            client._ik_recovery_active.clear()


class SafeNeroRobot(Robot):
    config_class = NeroDualRobotConfig
    name = "nero_dual_safe"

    def __init__(
        self,
        robot: NeroDualRobot,
        safety: NeroAsyncSafetyConfig,
        image_saver: ObservationImageSaver | None = None,
        video_saver: ObservationVideoSaver | None = None,
        tracer: NeroInferenceTracer | None = None,
    ):
        self.robot = robot
        self.safety = safety
        self.image_saver = image_saver
        self.video_saver = video_saver
        self.tracer = tracer
        set_tracer = getattr(self.robot, "set_tracer", None)
        if callable(set_tracer):
            set_tracer(tracer)
        self._last_action = dict(safety.fixed_ready_pose)
        self._target_action = dict(safety.fixed_ready_pose)
        self._executed_action = dict(safety.fixed_ready_pose)
        self._target_lock = threading.Lock()
        self._executor_stop = threading.Event()
        self._executor_thread: threading.Thread | None = None
        self._executor_step_count = 0
        self._executor_overrun_count = 0
        self._last_gripper_command: dict[str, float] | None = None
        self._interpolation_start_action = dict(safety.fixed_ready_pose)
        self._interpolation_target_action = dict(safety.fixed_ready_pose)
        self._interpolation_index = safety.high_rate_interpolation_steps

    def set_last_action(self, action: dict[str, float]) -> None:
        _require_pose_keys(action)
        normalized = {name: float(action[name]) for name in DEFAULT_DUAL_ACTION_NAMES}
        with self._target_lock:
            self._last_action = dict(normalized)
            self._target_action = dict(normalized)
            self._executed_action = dict(normalized)
            self._last_gripper_command = None
            self._executor_step_count = 0
            self._executor_overrun_count = 0
            self._interpolation_start_action = dict(normalized)
            self._interpolation_target_action = dict(normalized)
            self._interpolation_index = self.safety.high_rate_interpolation_steps

    @property
    def right(self):
        return self.robot.right

    @property
    def left(self):
        return self.robot.left

    @property
    def observation_features(self) -> dict:
        return self.robot.observation_features

    @property
    def action_features(self) -> dict:
        return self.robot.action_features

    @property
    def is_connected(self) -> bool:
        return self.robot.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.robot.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.robot.connect(calibrate=calibrate)

    def calibrate(self) -> None:
        self.robot.calibrate()

    def configure(self) -> None:
        self.robot.configure()

    def get_observation(self) -> RobotObservation:
        observation = self.robot.get_observation()
        if self.image_saver is not None:
            self.image_saver.maybe_save(observation)
        if self.video_saver is not None:
            self.video_saver.maybe_save(observation)
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        target = {name: float(action[name]) for name in DEFAULT_DUAL_ACTION_NAMES}
        if self.tracer is not None:
            self.tracer.record("policy_raw_action", {"action": target})
        with self._target_lock:
            previous_last_action = dict(self._last_action)
            limited = limit_action_step(
                target,
                previous_last_action,
                max_joint_step_rad=self.safety.max_policy_step_rad,
                max_gripper_step_m=self.safety.max_gripper_step_m,
                gripper_min_m=self.safety.gripper_min_m,
                gripper_max_m=self.safety.gripper_max_m,
            )
            self._last_action = dict(limited)
            self._target_action = dict(limited)
            self._interpolation_start_action = dict(self._executed_action)
            self._interpolation_target_action = dict(limited)
            self._interpolation_index = 0
        if self.tracer is not None:
            self.tracer.record(
                "policy_limited_target",
                {
                    "raw_action": target,
                    "last_action": previous_last_action,
                    "limited_action": limited,
                },
            )
        if self.safety.dry_run:
            logger.info("Dry-run Nero action: %s", limited)
            return limited
        if self.safety.high_rate_control:
            return limited
        sent = self.robot.send_action(limited)
        with self._target_lock:
            previous_executed_action = dict(self._executed_action)
            self._executed_action = {name: float(sent[name]) for name in DEFAULT_DUAL_ACTION_NAMES}
        if self.tracer is not None:
            self.tracer.record(
                "executor_step",
                {
                    "target_action": limited,
                    "previous_executed_action": previous_executed_action,
                    "limited_command": limited,
                    "sent_action": sent,
                    "high_rate_control": False,
                },
            )
            self.tracer.record_replay_action(sent)
        return sent

    def _should_send_high_rate_gripper(self, limited: dict[str, float]) -> bool:
        gripper_names = [namespaced_gripper_name("right"), namespaced_gripper_name("left")]
        if self._last_gripper_command is None:
            return True
        if self._executor_step_count % self.safety.high_rate_gripper_period == 0:
            return True
        return any(
            abs(float(limited[name]) - float(self._last_gripper_command[name]))
            >= self.safety.high_rate_gripper_epsilon_m
            for name in gripper_names
        )

    def _next_interpolated_target(self) -> dict[str, float]:
        steps = self.safety.high_rate_interpolation_steps
        if self._interpolation_index >= steps:
            return dict(self._interpolation_target_action)

        t = (self._interpolation_index + 1) / steps
        interpolated = {
            name: float(self._interpolation_start_action[name])
            + t * (float(self._interpolation_target_action[name]) - float(self._interpolation_start_action[name]))
            for name in DEFAULT_DUAL_ACTION_NAMES
        }
        self._interpolation_index += 1
        return interpolated

    def execute_high_rate_step(self, *, dt_s: float | None = None) -> RobotAction:
        with self._target_lock:
            target = self._next_interpolated_target()
            executed = dict(self._executed_action)

        limited = limit_action_step(
            target,
            executed,
            max_joint_step_rad=self.safety.max_executor_step_rad,
            max_gripper_step_m=self.safety.max_executor_gripper_step_m,
            gripper_min_m=self.safety.gripper_min_m,
            gripper_max_m=self.safety.gripper_max_m,
        )
        if self.safety.dry_run:
            logger.info("Dry-run high-rate Nero action: %s", limited)
            with self._target_lock:
                self._executed_action = dict(limited)
            return limited
        send_gripper = self._should_send_high_rate_gripper(limited)
        sent = self.robot.send_action(
            limited,
            send_gripper=send_gripper,
            read_feedback=False,
        )
        with self._target_lock:
            self._executed_action = {name: float(sent[name]) for name in DEFAULT_DUAL_ACTION_NAMES}
            if send_gripper:
                self._last_gripper_command = {
                    namespaced_gripper_name("right"): self._executed_action[namespaced_gripper_name("right")],
                    namespaced_gripper_name("left"): self._executed_action[namespaced_gripper_name("left")],
                }
            self._executor_step_count += 1
        if self.tracer is not None:
            self.tracer.record(
                "executor_step",
                {
                    "target_action": target,
                    "previous_executed_action": executed,
                    "limited_command": limited,
                    "sent_action": sent,
                    "dt_s": dt_s,
                    "high_rate_control": True,
                    "send_gripper": send_gripper,
                },
            )
            self.tracer.record_replay_action(sent, dt_s=dt_s)
        return sent

    def _run_high_rate_executor_once(self, *, dt_s: float) -> None:
        start_t = time.perf_counter()
        self.execute_high_rate_step(dt_s=dt_s)
        elapsed_s = time.perf_counter() - start_t
        sleep_s = max(dt_s - elapsed_s, 0.0)
        overrun_s = max(elapsed_s - dt_s, 0.0)
        if overrun_s > 0:
            self._executor_overrun_count += 1
            if self._executor_overrun_count % self.safety.high_rate_overrun_log_every == 1:
                logger.warning(
                    "Nero high-rate executor overrun: elapsed=%.4fms target=%.4fms overrun=%.4fms count=%d",
                    elapsed_s * 1000.0,
                    dt_s * 1000.0,
                    overrun_s * 1000.0,
                    self._executor_overrun_count,
                )
        if self.tracer is not None:
            self.tracer.record(
                "executor_timing",
                {
                    "target_dt_s": dt_s,
                    "elapsed_s": elapsed_s,
                    "sleep_s": sleep_s,
                    "overrun_s": overrun_s,
                    "actual_hz": 1.0 / elapsed_s if elapsed_s > 0 else None,
                    "overrun_count": self._executor_overrun_count,
                },
            )
        precise_sleep(sleep_s)

    def _high_rate_executor_loop(self) -> None:
        dt_s = self.safety.high_rate_dt_s
        logger.info("Starting Nero high-rate executor at %.1f Hz.", 1.0 / dt_s)
        while not self._executor_stop.is_set():
            self._run_high_rate_executor_once(dt_s=dt_s)

    def start_high_rate_executor(self) -> None:
        if not self.safety.high_rate_control or self.safety.dry_run:
            return
        if self._executor_thread is not None and self._executor_thread.is_alive():
            return
        self._executor_stop.clear()
        self._executor_thread = threading.Thread(target=self._high_rate_executor_loop, daemon=True)
        self._executor_thread.start()

    def stop_high_rate_executor(self) -> None:
        self._executor_stop.set()
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=1.0)
            self._executor_thread = None

    def disconnect(self) -> None:
        self.stop_high_rate_executor()
        self.robot.disconnect()
        if self.tracer is not None:
            self.tracer.close()


class EESafeNeroRobot(SafeNeroRobot):
    name = "nero_dual_ee_local_se3_safe"

    def __init__(
        self,
        robot: NeroDualRobot,
        safety: NeroAsyncSafetyConfig,
        *,
        ee_adapter: NeroEELocalSE3Adapter,
        ik_adapter: NeroDualCuroboIKAdapter,
        image_saver: ObservationImageSaver | None = None,
        video_saver: ObservationVideoSaver | None = None,
        tracer: NeroInferenceTracer | None = None,
    ):
        super().__init__(robot, safety, image_saver=image_saver, video_saver=video_saver, tracer=tracer)
        self.ee_adapter = ee_adapter
        self.ik_adapter = ik_adapter
        self._last_success_ik_action: dict[str, float] | None = None

    @property
    def observation_features(self) -> dict:
        features = {name: float for name in EE_LOCAL_SE3_ACTION_NAMES}
        flange_features = getattr(self.robot, "flange_observation_features", {})
        features.update(
            {
                name: feature
                for name, feature in flange_features.items()
                if isinstance(feature, tuple)
            }
        )
        return features

    @property
    def action_features(self) -> dict:
        return {name: float for name in EE_LOCAL_SE3_ACTION_NAMES}

    def get_observation(self) -> RobotObservation:
        get_flange_observation = getattr(self.robot, "get_flange_observation", None)
        if not callable(get_flange_observation):
            raise TypeError("EE local SE3 inference requires robot.get_flange_observation().")
        flange_observation = get_flange_observation()
        if self.image_saver is not None:
            self.image_saver.maybe_save(flange_observation)
        if self.video_saver is not None:
            self.video_saver.maybe_save(flange_observation)

        policy_state = np.asarray(
            self.ee_adapter.flange_observation_to_policy_state(flange_observation),
            dtype=float,
        )
        if policy_state.shape != (len(EE_LOCAL_SE3_ACTION_NAMES),):
            raise ValueError(
                f"EE local SE3 observation state must have shape ({len(EE_LOCAL_SE3_ACTION_NAMES)},), "
                f"got {policy_state.shape}."
            )
        observation = {
            name: float(policy_state[idx]) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)
        }
        for name in self.observation_features:
            if name in observation:
                continue
            if name in flange_observation:
                observation[name] = flange_observation[name]
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        policy_action = np.asarray(
            [float(action[name]) for name in EE_LOCAL_SE3_ACTION_NAMES],
            dtype=float,
        )
        if self.tracer is not None:
            self.tracer.record(
                "policy_raw_ee_action",
                {
                    "action": {
                        name: float(policy_action[idx])
                        for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)
                    }
                },
            )
        limited_policy_action = policy_action
        if self.safety.max_ee_position_step_m > 0 or self.safety.max_ee_rotation_step_rad > 0:
            current_policy_state = np.asarray(self.ee_adapter.read_robot_policy_state(self.robot), dtype=float)
            limited_policy_action = limit_ee_policy_action_step(
                policy_action,
                current_policy_state,
                max_position_step_m=self.safety.max_ee_position_step_m,
                max_rotation_step_rad=self.safety.max_ee_rotation_step_rad,
            )
            if self.tracer is not None:
                self.tracer.record(
                    "policy_limited_ee_action",
                    {
                        "raw_action": {
                            name: float(policy_action[idx])
                            for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)
                        },
                        "current_state": {
                            name: float(current_policy_state[idx])
                            for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)
                        },
                        "limited_action": {
                            name: float(limited_policy_action[idx])
                            for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)
                        },
                    },
                )
        ee_targets = self.ee_adapter.policy_action_to_nero_ee_targets(limited_policy_action)
        right_current_joints = np.asarray(self.robot.right.read_joints(), dtype=float)
        left_current_joints = np.asarray(self.robot.left.read_joints(), dtype=float)
        right_seed_candidates = [
            (
                "last_sent_action",
                np.asarray(
                    [self._executed_action[name] for name in namespaced_joint_names("right")],
                    dtype=float,
                ),
            )
        ]
        left_seed_candidates = [
            (
                "last_sent_action",
                np.asarray(
                    [self._executed_action[name] for name in namespaced_joint_names("left")],
                    dtype=float,
                ),
            )
        ]
        if self._last_success_ik_action is not None:
            right_seed_candidates.append(
                (
                    "last_success_ik_solution",
                    np.asarray(
                        [
                            self._last_success_ik_action[name]
                            for name in namespaced_joint_names("right")
                        ],
                        dtype=float,
                    ),
                )
            )
            left_seed_candidates.append(
                (
                    "last_success_ik_solution",
                    np.asarray(
                        [
                            self._last_success_ik_action[name]
                            for name in namespaced_joint_names("left")
                        ],
                        dtype=float,
                    ),
                )
            )
        if self.tracer is not None:
            self.tracer.record(
                "ee_ik_request",
                {
                    "right_pose": ee_targets.right_pose,
                    "left_pose": ee_targets.left_pose,
                    "right_current_joints": right_current_joints,
                    "left_current_joints": left_current_joints,
                    "right_seed_candidates": right_seed_candidates,
                    "left_seed_candidates": left_seed_candidates,
                },
            )
        try:
            solve_with_metadata = getattr(
                self.ik_adapter,
                "ee_targets_to_joint_action_with_metadata",
                None,
            )
            if callable(solve_with_metadata):
                joint_target, ik_metadata = solve_with_metadata(
                    ee_targets,
                    right_current_joints=right_current_joints,
                    left_current_joints=left_current_joints,
                    right_seed_candidates=right_seed_candidates,
                    left_seed_candidates=left_seed_candidates,
                )
            else:
                joint_target = self.ik_adapter.ee_targets_to_joint_action(
                    ee_targets,
                    right_current_joints=right_current_joints,
                    left_current_joints=left_current_joints,
                )
                ik_metadata = {
                    "right": {
                        "seed_source": "real_current_joints",
                        "seed_joints": right_current_joints,
                        "attempts": [],
                    },
                    "left": {
                        "seed_source": "real_current_joints",
                        "seed_joints": left_current_joints,
                        "attempts": [],
                    },
                }
        except Exception as exc:
            if self.tracer is not None:
                self.tracer.record(
                    "ee_ik_failed",
                    {
                        "right_pose": ee_targets.right_pose,
                        "left_pose": ee_targets.left_pose,
                        "right_current_joints": right_current_joints,
                        "left_current_joints": left_current_joints,
                        "right_seed_candidates": right_seed_candidates,
                        "left_seed_candidates": left_seed_candidates,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise
        self._last_success_ik_action = {
            name: float(joint_target[name]) for name in DEFAULT_DUAL_ACTION_NAMES
        }
        if self.tracer is not None:
            self.tracer.record(
                "ee_ik_joint_target",
                {
                    "right_pose": ee_targets.right_pose,
                    "left_pose": ee_targets.left_pose,
                    "right_current_joints": right_current_joints,
                    "left_current_joints": left_current_joints,
                    "joint_target": joint_target,
                    "ik_metadata": ik_metadata,
                },
            )
        return super().send_action(joint_target)


def _make_ee_adapter_from_config(cfg: NeroAsyncClientConfig) -> NeroEELocalSE3Adapter:
    return NeroEELocalSE3Adapter.from_camera_to_base_yamls(
        right_camera_to_base_yaml=cfg.right_handeye_camera_to_base_yaml,
        left_camera_to_base_yaml=cfg.left_handeye_camera_to_base_yaml,
        base_or_head_xy=(cfg.ee_base_or_head_x, cfg.ee_base_or_head_y),
        euler_order=cfg.ee_euler_order,
    )


def _make_curobo_adapter_from_config(cfg: NeroAsyncClientConfig) -> NeroDualCuroboIKAdapter:
    return NeroDualCuroboIKAdapter(
        robot_file=cfg.curobo_robot_file,
        euler_order=cfg.ee_euler_order,
        num_seeds=cfg.curobo_num_seeds,
        position_threshold=cfg.curobo_position_threshold,
        rotation_threshold=cfg.curobo_rotation_threshold,
        device=cfg.curobo_device,
    )


def _make_nero_robot_wrapper(
    robot: NeroDualRobot,
    *,
    cfg: NeroAsyncClientConfig | None,
    safety: NeroAsyncSafetyConfig,
    image_saver: ObservationImageSaver,
    video_saver: ObservationVideoSaver,
    tracer: NeroInferenceTracer | None,
) -> SafeNeroRobot:
    if cfg is not None and cfg.action_mode == "ee_local_se3":
        return EESafeNeroRobot(
            robot,
            safety,
            ee_adapter=_make_ee_adapter_from_config(cfg),
            ik_adapter=_make_curobo_adapter_from_config(cfg),
            image_saver=image_saver,
            video_saver=video_saver,
            tracer=tracer,
        )
    return SafeNeroRobot(
        robot,
        safety,
        image_saver=image_saver,
        video_saver=video_saver,
        tracer=tracer,
    )


def _make_client_config(cfg: NeroAsyncClientConfig, *, record_dir: str = ""):
    from lerobot.async_inference.configs import RobotClientConfig

    return RobotClientConfig(
        robot=cfg.robot,
        server_address=cfg.server_address,
        policy_device=cfg.policy_device,
        client_device=cfg.client_device,
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.policy_path,
        chunk_size_threshold=cfg.chunk_size_threshold,
        actions_per_chunk=cfg.actions_per_chunk,
        fps=cfg.fps,
        task=cfg.task,
        aggregate_fn_name=cfg.aggregate_fn_name,
        debug_visualize_queue_size=cfg.debug_visualize_queue_size,
        record_dir=record_dir,
    )


def _make_safe_robot_client(
    client_config,
    safety: NeroAsyncSafetyConfig,
    debug_save_images: NeroDebugImageSaveConfig,
    debug_save_videos: NeroDebugVideoSaveConfig,
    tracer: NeroInferenceTracer | None = None,
    nero_cfg: NeroAsyncClientConfig | None = None,
):
    from lerobot.async_inference.robot_client import RobotClient

    class NeroSafeRobotClient(RobotClient):
        def __init__(
            self,
            config,
            safety_config: NeroAsyncSafetyConfig,
            image_save_config: NeroDebugImageSaveConfig,
            video_save_config: NeroDebugVideoSaveConfig,
            trace_writer: NeroInferenceTracer | None,
            async_client_config: NeroAsyncClientConfig | None,
        ):
            self._nero_safety = safety_config
            self._nero_image_saver = ObservationImageSaver(image_save_config)
            self._nero_video_saver = ObservationVideoSaver(video_save_config)
            self._nero_tracer = trace_writer
            self._nero_cfg = async_client_config
            self._ik_recovery_active = threading.Event()
            self._ik_recovery_generation = 0
            self._ik_recovery_lock = threading.Lock()
            super().__init__(config)

        def _make_robot(self, robot_config: RobotConfig) -> SafeNeroRobot:
            if not isinstance(robot_config, NeroDualRobotConfig):
                raise TypeError(
                    f"nero-async-client requires --robot.type=nero_dual, got {robot_config.type!r}."
                )
            return _make_nero_robot_wrapper(
                NeroDualRobot(robot_config),
                cfg=self._nero_cfg,
                safety=self._nero_safety,
                image_saver=self._nero_image_saver,
                video_saver=self._nero_video_saver,
                tracer=self._nero_tracer,
            )

        def _action_tensor_to_trace_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
            action_cpu = action_tensor.detach().to("cpu")
            return {name: float(action_cpu[idx].item()) for idx, name in enumerate(self.robot.action_features)}

        def receive_actions(self, verbose: bool = False):
            self.start_barrier.wait()
            self.logger.info("Action receiving thread starting")
            while self.running:
                try:
                    if self._ik_recovery_active.is_set():
                        time.sleep(self.config.environment_dt)
                        continue

                    with self._ik_recovery_lock:
                        request_generation = self._ik_recovery_generation
                    actions_chunk = self.stub.GetActions(services_pb2.Empty())
                    if len(actions_chunk.data) == 0:
                        continue

                    receive_time = time.time()
                    deserialize_start = time.perf_counter()
                    timed_actions = pickle.loads(actions_chunk.data)  # nosec
                    deserialize_time = time.perf_counter() - deserialize_start

                    client_device = self.config.client_device
                    if client_device != "cpu":
                        for timed_action in timed_actions:
                            if timed_action.get_action().device.type != client_device:
                                timed_action.action = timed_action.get_action().to(client_device)

                    with self._ik_recovery_lock:
                        drop_for_recovery = (
                            self._ik_recovery_active.is_set()
                            or request_generation != self._ik_recovery_generation
                        )
                        if not drop_for_recovery:
                            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                            self.must_go.set()

                    if drop_for_recovery:
                        if self._nero_tracer is not None and timed_actions:
                            self._nero_tracer.record(
                                "policy_action_chunk_dropped",
                                {
                                    "reason": "ik_failure_recovery",
                                    "receive_time_s": receive_time,
                                    "deserialize_time_s": deserialize_time,
                                    "timesteps": [int(action.get_timestep()) for action in timed_actions],
                                },
                            )
                        continue

                    if self._nero_tracer is not None and timed_actions:
                        self._nero_tracer.record(
                            "policy_action_chunk",
                            {
                                "receive_time_s": receive_time,
                                "deserialize_time_s": deserialize_time,
                                "timesteps": [int(action.get_timestep()) for action in timed_actions],
                                "actions": [
                                    self._action_tensor_to_trace_dict(action.get_action())
                                    for action in timed_actions
                                ],
                            },
                        )
                except grpc.RpcError as e:
                    self.logger.error(f"Error receiving actions: {e}")

        def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
            get_start = time.perf_counter()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
            get_end = time.perf_counter() - get_start

            action = self._action_tensor_to_action_dict(timed_action.get_action())
            if self._nero_tracer is not None:
                self._nero_tracer.record(
                    "policy_queue_action",
                    {
                        "timestep": int(timed_action.get_timestep()),
                        "timestamp": float(timed_action.get_timestamp()),
                        "queue_get_time_s": get_end,
                        "action": action,
                    },
                )
            try:
                performed_action = self.robot.send_action(action)
            except Exception as exc:
                if (
                    not self._nero_safety.recover_on_ik_failure
                    or not _is_ik_failure_exception(exc)
                ):
                    raise
                self.logger.warning("Nero EE IK failed; recovering to fixed ready pose.", exc_info=True)
                wait_for_resume = (
                    _wait_for_enter
                    if self._nero_safety.ik_failure_recovery_wait_for_enter
                    else None
                )
                recover_nero_async_client_from_ik_failure(
                    self,
                    self._nero_safety,
                    tracer=self._nero_tracer,
                    wait_for_resume=wait_for_resume,
                    error=exc,
                )
                return dict(self._nero_safety.fixed_ready_pose)
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            if verbose:
                with self.action_queue_lock:
                    current_queue_size = self.action_queue.qsize()
                self.logger.debug(
                    f"Ts={timed_action.get_timestamp()} | "
                    f"Action #{timed_action.get_timestep()} performed | "
                    f"Queue size: {current_queue_size}"
                )
                self.logger.debug(
                    f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
                )
            return performed_action

    return NeroSafeRobotClient(client_config, safety, debug_save_images, debug_save_videos, tracer, nero_cfg)


def _keyboard_stop_watcher(client: Any) -> None:
    while client.running:
        try:
            value = input().strip().lower()
        except EOFError:
            return
        if value in {"q", "s", "stop", "e", "exit"}:
            logger.warning("Keyboard stop requested.")
            client.stop()
            return


def _wait_for_enter() -> None:
    input("Nero is at the fixed ready pose. Press ENTER to start policy control, or Ctrl-C to abort.")


def run_nero_async_client(cfg: NeroAsyncClientConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()

    tracer = NeroInferenceTracer(
        cfg.trace,
        meta={
            "server_address": cfg.server_address,
            "policy_type": cfg.policy_type,
            "policy_path": cfg.policy_path,
            "task": cfg.task,
            "fps": cfg.fps,
            "actions_per_chunk": cfg.actions_per_chunk,
            "chunk_size_threshold": cfg.chunk_size_threshold,
            "aggregate_fn_name": cfg.aggregate_fn_name,
            "action_mode": cfg.action_mode,
            "ee": {
                "right_handeye_camera_to_base_yaml": cfg.right_handeye_camera_to_base_yaml,
                "left_handeye_camera_to_base_yaml": cfg.left_handeye_camera_to_base_yaml,
                "base_or_head_xy": [cfg.ee_base_or_head_x, cfg.ee_base_or_head_y],
                "euler_order": cfg.ee_euler_order,
            },
            "curobo": {
                "robot_file": cfg.curobo_robot_file,
                "num_seeds": cfg.curobo_num_seeds,
                "position_threshold": cfg.curobo_position_threshold,
                "rotation_threshold": cfg.curobo_rotation_threshold,
                "device": cfg.curobo_device,
            },
            "safety": asdict(cfg.safety),
            "robot": asdict(cfg.robot),
            "debug_save_videos": asdict(cfg.debug_save_videos),
        },
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    if tracer.enabled and tracer.run_dir is not None:
        logger.info("Recording Nero inference trace to %s", tracer.run_dir)
    client_cfg = _make_client_config(cfg)
    if hasattr(client_cfg, "record_dir"):
        client_cfg.record_dir = str(tracer.run_dir or "")
    if cfg.debug_save_videos.enabled and not cfg.debug_save_videos.dir and tracer.run_dir is not None:
        cfg.debug_save_videos.dir = str(tracer.run_dir)
    client = None
    try:
        client = _make_safe_robot_client(
            client_cfg,
            cfg.safety,
            cfg.debug_save_images,
            cfg.debug_save_videos,
            tracer,
            nero_cfg=cfg,
        )
        if cfg.safety.dry_run:
            client.robot.set_last_action(cfg.safety.fixed_ready_pose)
            logger.warning("Dry-run enabled; skipping physical ready-pose synchronization.")
        else:
            sync_error = sync_to_fixed_ready_pose(
                client.robot.robot,
                cfg.safety.fixed_ready_pose,
                takeover_time_s=cfg.safety.takeover_time_s,
                takeover_dt_s=cfg.safety.takeover_dt_s,
                tolerance_rad=cfg.safety.ready_tolerance_rad,
            )
            client.robot.set_last_action(cfg.safety.fixed_ready_pose)
            logger.info("Nero ready pose synchronized. Max joint error: %.4f rad", sync_error)

        if cfg.wait_for_enter:
            _wait_for_enter()

        client.robot.start_high_rate_executor()

        if not client.start():
            raise RuntimeError(f"Failed to start Nero async client for server {cfg.server_address}.")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        keyboard_thread = None
        if cfg.keyboard_stop:
            keyboard_thread = threading.Thread(target=_keyboard_stop_watcher, args=(client,), daemon=True)
            keyboard_thread.start()

        client.control_loop(task=cfg.task)
        action_receiver_thread.join()
        if keyboard_thread is not None:
            keyboard_thread.join(timeout=0.1)
    finally:
        if client is not None and client.running:
            client.stop()
        if client is not None and cfg.debug_visualize_queue_size:
            from lerobot.async_inference.helpers import visualize_action_queue_size

            visualize_action_queue_size(client.action_queue_size)
        if client is not None:
            video_saver = getattr(client, "_nero_video_saver", None)
            if video_saver is not None:
                video_saver.close()
        tracer.close()


@parser.wrap()
def nero_async_client(cfg: NeroAsyncClientConfig) -> None:
    run_nero_async_client(cfg)


def main() -> None:
    nero_async_client()


if __name__ == "__main__":
    main()
