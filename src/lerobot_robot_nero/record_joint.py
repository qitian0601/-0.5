import logging
import select
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pprint import pformat
from typing import Callable

import lerobot_teleoperator_so101_8dof  # noqa: F401
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager, aggregate_pipeline_dataset_features
from lerobot.datasets import create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .config_nero import NeroRobotConfig, NeroTrimConfig
from .mapping import (
    SO101ToNeroMapping,
    map_so101_action_to_nero,
    namespaced_gripper_name,
    namespaced_joint_names,
)
from .prepare_sync import smooth_takeover_commands
from .robot_nero import NeroRobot
from .trimming import trim_static_head_tail

logger = logging.getLogger(__name__)


class IdleTeleopDecision(Enum):
    START_RECORDING = "start_recording"
    FINISH_AND_EXIT = "finish_and_exit"


@dataclass
class NeroRecordJointConfig:
    leader: TeleoperatorConfig
    robot: RobotConfig
    dataset: DatasetRecordConfig
    trim: NeroTrimConfig = field(default_factory=NeroTrimConfig)
    resume: bool = False
    dry_run: bool = False
    manual_stop: bool = True


def make_dataset_features(robot: NeroRobot, *, use_videos: bool) -> dict[str, dict]:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=use_videos,
    )
    observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=use_videos,
    )
    return combine_feature_dicts(action_features, observation_features)


def flatten_episode_frame(raw_frame: dict, *, features: dict[str, dict], task: str) -> dict:
    observation_frame = build_dataset_frame(features, raw_frame["observation"], prefix=OBS_STR)
    action_frame = build_dataset_frame(features, raw_frame["action"], prefix=ACTION)
    return {**observation_frame, **action_frame, "task": task}


def should_stop_episode(
    *, start_time_s: float, now_s: float, episode_time_s: float, manual_stop_requested: bool
) -> bool:
    return manual_stop_requested or now_s - start_time_s >= episode_time_s


def _stdin_stop_requested() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def _stdin_line_if_ready() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.readline()


def idle_decision_from_text(text: str) -> IdleTeleopDecision:
    value = text.strip().lower()
    if value in ("q", "quit", "exit", "结束", "退出"):
        return IdleTeleopDecision.FINISH_AND_EXIT
    return IdleTeleopDecision.START_RECORDING


def _target_to_action(q_target: np.ndarray, gripper_width: float, *, arm: str) -> dict[str, float]:
    action = {name: float(value) for name, value in zip(namespaced_joint_names(arm), q_target, strict=True)}
    action[namespaced_gripper_name(arm)] = float(gripper_width)
    return action


def run_teleop_step(
    *,
    leader,
    robot: NeroRobot,
    mapping: SO101ToNeroMapping,
    record: bool,
) -> dict | None:
    observation = robot.get_observation()
    leader_action = leader.get_action()
    action_to_send = map_so101_action_to_nero(leader_action, mapping=mapping, arm=robot.arm)
    sent_action = robot.send_action(action_to_send)
    if not record:
        return None
    return {"observation": observation, "action": sent_action}


def sync_to_current_leader(
    *,
    leader,
    robot: NeroRobot,
    mapping: SO101ToNeroMapping,
    takeover_time_s: float = 2.0,
    takeover_dt_s: float = 0.02,
) -> None:
    current = robot._read_joints()
    leader_action = leader.get_action()
    mapped = map_so101_action_to_nero(leader_action, mapping=mapping, arm=robot.arm)
    target = np.asarray([mapped[name] for name in namespaced_joint_names(robot.arm)], dtype=float)
    gripper_width = mapped[namespaced_gripper_name(robot.arm)]

    original_alpha = robot.config.command.alpha
    original_max_step = robot.config.command.max_step_rad
    try:
        robot.config.command.alpha = 1.0
        robot.config.command.max_step_rad = 0.0
        steps = max(int(round(takeover_time_s / takeover_dt_s)), 1)
        for q_cmd in smooth_takeover_commands(current, target, steps):
            robot.send_action(_target_to_action(q_cmd, gripper_width, arm=robot.arm))
            precise_sleep(takeover_dt_s)
    finally:
        robot.config.command.alpha = original_alpha
        robot.config.command.max_step_rad = original_max_step


def run_idle_teleop_until_start(
    *,
    leader,
    robot: NeroRobot,
    mapping: SO101ToNeroMapping,
    fps: int,
    prompt: str,
) -> IdleTeleopDecision:
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; skipping idle teleop wait and starting the episode immediately.")
        return IdleTeleopDecision.START_RECORDING
    print(prompt)
    print("主从同步保持开启；按 ENTER 开始录制，输入 q 后回车结束并保存已有数据。")
    control_interval = 1.0 / fps
    while True:
        start_loop_t = time.perf_counter()
        line = _stdin_line_if_ready()
        if line is not None:
            return idle_decision_from_text(line)
        run_teleop_step(leader=leader, robot=robot, mapping=mapping, record=False)
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))


def run_idle_teleop_until_shutdown(
    *,
    leader,
    robot: NeroRobot,
    mapping: SO101ToNeroMapping,
    fps: int,
    prompt: str,
) -> None:
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; skipping final reset wait.")
        return
    print(prompt)
    print("主从同步保持开启；请复位机械臂，复位完成后按 ENTER 关闭程序。")
    control_interval = 1.0 / fps
    while True:
        start_loop_t = time.perf_counter()
        if _stdin_stop_requested():
            return
        run_teleop_step(leader=leader, robot=robot, mapping=mapping, record=False)
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))


def collect_episode_buffer(
    *,
    leader,
    robot: NeroRobot,
    mapping: SO101ToNeroMapping,
    episode_time_s: float,
    fps: int,
    stop_requested: Callable[[], bool] | None = None,
) -> list[dict]:
    frames: list[dict] = []
    control_interval = 1.0 / fps
    start_episode_t = time.perf_counter()
    stop_requested = stop_requested or (lambda: False)

    while True:
        now_s = time.perf_counter()
        if should_stop_episode(
            start_time_s=start_episode_t,
            now_s=now_s,
            episode_time_s=episode_time_s,
            manual_stop_requested=stop_requested(),
        ):
            break
        start_loop_t = time.perf_counter()
        frame = run_teleop_step(leader=leader, robot=robot, mapping=mapping, record=True)
        if frame is not None:
            frames.append(frame)

        dt_s = time.perf_counter() - start_loop_t
        if dt_s > control_interval:
            logger.warning(
                "Nero record loop is running slower (%.1f Hz) than target fps (%d Hz).",
                1.0 / dt_s,
                fps,
            )
        precise_sleep(max(control_interval - dt_s, 0.0))

    return frames


def _create_or_resume_dataset(cfg: NeroRecordJointConfig, robot: NeroRobot, features: dict[str, dict]):
    if cfg.dry_run:
        return None

    num_cameras = len(robot.cameras)
    if cfg.resume:
        if cfg.dataset.root is None:
            raise ValueError("--resume=true requires --dataset.root for Nero recording.")
        return LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            camera_encoder=cfg.dataset.camera_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
            if num_cameras > 0
            else 0,
        )

    repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
    if repo_name.startswith("eval_"):
        raise ValueError(
            "Dataset names starting with 'eval_' are reserved for policy evaluation. "
            "Use a data-collection dataset name instead."
        )
    cfg.dataset.stamp_repo_id()
    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        features=features,
        root=cfg.dataset.root,
        robot_type=robot.name,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
        if num_cameras > 0
        else 0,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        camera_encoder=cfg.dataset.camera_encoder,
        encoder_threads=cfg.dataset.encoder_threads,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
    )


@parser.wrap()
def record_joint(cfg: NeroRecordJointConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroRobotConfig):
        raise TypeError(f"nero-record-joint requires --robot.type=nero, got {cfg.robot.type!r}.")

    mapping = SO101ToNeroMapping.from_config(cfg.robot.mapping)
    leader = make_teleoperator_from_config(cfg.leader)
    robot = NeroRobot(cfg.robot)
    dataset = None

    try:
        features = make_dataset_features(robot, use_videos=cfg.dataset.video)
        dataset = _create_or_resume_dataset(cfg, robot, features)

        robot.connect()
        leader.connect()
        input("即将根据当前 SO101 位姿同步 Nero。请确认机械臂周围安全，然后按 ENTER 开始同步。")
        sync_to_current_leader(leader=leader, robot=robot, mapping=mapping)
        logger.info("Nero synchronized to the SO101 leader target. Entering idle teleop mode.")

        recorded_episodes = 0
        manager = VideoEncodingManager(dataset) if dataset is not None else None
        if manager is not None:
            manager.__enter__()
        try:
            while recorded_episodes < cfg.dataset.num_episodes:
                decision = run_idle_teleop_until_start(
                    leader=leader,
                    robot=robot,
                    mapping=mapping,
                    fps=cfg.dataset.fps,
                    prompt=(
                        f"准备录制 Nero episode {recorded_episodes + 1}/"
                        f"{cfg.dataset.num_episodes}。"
                    ),
                )
                if decision is IdleTeleopDecision.FINISH_AND_EXIT:
                    logger.info(
                        "User requested early finish after %d/%d episodes.",
                        recorded_episodes,
                        cfg.dataset.num_episodes,
                    )
                    break
                stop_requested = None
                if cfg.manual_stop:
                    if sys.stdin.isatty():
                        print("Recording... press ENTER to finish this episode.")
                        stop_requested = _stdin_stop_requested
                    else:
                        logger.warning(
                            "manual_stop=true but stdin is not interactive; falling back to episode_time_s."
                        )
                frames = collect_episode_buffer(
                    leader=leader,
                    robot=robot,
                    mapping=mapping,
                    episode_time_s=cfg.dataset.episode_time_s,
                    fps=cfg.dataset.fps,
                    stop_requested=stop_requested,
                )
                trimmed = trim_static_head_tail(
                    frames,
                    fps=cfg.dataset.fps,
                    arm=robot.arm,
                    config=cfg.trim,
                )
                if not trimmed:
                    logger.warning("Episode rejected after trimming: no movement or too few frames.")
                    continue

                if dataset is not None:
                    for frame in trimmed:
                        dataset.add_frame(
                            flatten_episode_frame(
                                frame,
                                features=dataset.features,
                                task=cfg.dataset.single_task,
                            )
                        )
                    dataset.save_episode()
                else:
                    logger.info("Dry run episode kept %d/%d frames after trimming.", len(trimmed), len(frames))

                recorded_episodes += 1

            run_idle_teleop_until_shutdown(
                leader=leader,
                robot=robot,
                mapping=mapping,
                fps=cfg.dataset.fps,
                prompt=(
                    f"录制结束：已保存 {recorded_episodes}/"
                    f"{cfg.dataset.num_episodes} 条 episode。"
                ),
            )
        except BaseException as exc:
            if manager is not None:
                manager.__exit__(type(exc), exc, exc.__traceback__)
                manager = None
            raise
        finally:
            if manager is not None:
                manager.__exit__(None, None, None)

        if dataset is not None and cfg.dataset.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    finally:
        if dataset is not None:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if leader.is_connected:
            leader.disconnect()

    return dataset


def main() -> None:
    register_third_party_plugins()
    record_joint()


if __name__ == "__main__":
    main()
