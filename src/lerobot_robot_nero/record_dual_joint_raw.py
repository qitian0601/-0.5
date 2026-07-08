import logging
import select
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
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

from .config_nero import NeroDualRobotConfig
from .mapping import SO101ToNeroMapping, map_so101_action_to_nero, namespaced_gripper_name, namespaced_joint_names
from .prepare_sync import smooth_takeover_commands
from .robot_nero_dual import NeroDualRobot

logger = logging.getLogger(__name__)


class IdleTeleopDecision(Enum):
    START_RECORDING = "start_recording"
    FINISH_AND_EXIT = "finish_and_exit"


class EpisodeReviewDecision(Enum):
    SAVE = "save"
    RERECORD = "rerecord"
    FINISH_AND_EXIT = "finish_and_exit"


@dataclass
class NeroRecordDualJointRawConfig:
    right_leader: TeleoperatorConfig
    left_leader: TeleoperatorConfig
    robot: RobotConfig
    dataset: DatasetRecordConfig
    flange_dataset: DatasetRecordConfig | None = None
    resume: bool = False
    dry_run: bool = False
    manual_stop: bool = True
    record_joint_dataset: bool = True
    takeover_time_s: float = 2.0
    takeover_dt_s: float = 0.02


def make_dataset_features(robot: NeroDualRobot, *, use_videos: bool) -> dict[str, dict]:
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


def make_flange_dataset_features(robot: NeroDualRobot, *, use_videos: bool) -> dict[str, dict]:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.flange_action_features),
        use_videos=use_videos,
    )
    observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.flange_observation_features),
        use_videos=use_videos,
    )
    return combine_feature_dicts(action_features, observation_features)


def flatten_episode_frame(raw_frame: dict, *, features: dict[str, dict], task: str) -> dict:
    observation_frame = build_dataset_frame(features, raw_frame["observation"], prefix=OBS_STR)
    action_frame = build_dataset_frame(features, raw_frame["action"], prefix=ACTION)
    return {**observation_frame, **action_frame, "task": task}


def convert_joint_frame_to_flange_frame(raw_frame: dict, robot: NeroDualRobot) -> dict:
    flange_observation = dict(raw_frame["flange_observation"])
    for key, value in raw_frame["observation"].items():
        if key in robot.cameras:
            flange_observation[key] = value
    return {
        "observation": flange_observation,
        "action": raw_frame["flange_action"],
    }


def save_raw_episode_frames_pair(*, joint_dataset, flange_dataset, frames: list[dict], task: str) -> None:
    for frame in frames:
        joint_frame = {"observation": frame["observation"], "action": frame["action"]}
        flange_frame = frame["flange"]
        joint_dataset.add_frame(
            flatten_episode_frame(joint_frame, features=joint_dataset.features, task=task)
        )
        flange_dataset.add_frame(
            flatten_episode_frame(flange_frame, features=flange_dataset.features, task=task)
        )
    joint_dataset.save_episode()
    flange_dataset.save_episode()


def save_raw_flange_episode_frames(*, flange_dataset, frames: list[dict], task: str) -> None:
    for frame in frames:
        flange_dataset.add_frame(
            flatten_episode_frame(frame["flange"], features=flange_dataset.features, task=task)
        )
    flange_dataset.save_episode()


def attach_flange_frames(frames: list[dict], robot: NeroDualRobot) -> list[dict]:
    enriched = []
    for frame in frames:
        enriched_frame = dict(frame)
        enriched_frame["flange"] = convert_joint_frame_to_flange_frame(frame, robot)
        enriched.append(enriched_frame)
    return enriched


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


def review_decision_from_text(text: str) -> EpisodeReviewDecision:
    value = text.strip().lower()
    if value in ("r", "redo", "rerecord", "重录"):
        return EpisodeReviewDecision.RERECORD
    if value in ("q", "quit", "exit", "结束", "退出"):
        return EpisodeReviewDecision.FINISH_AND_EXIT
    return EpisodeReviewDecision.SAVE


def prompt_raw_episode_review(*, episode_number: int, raw_frames: int) -> EpisodeReviewDecision:
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; saving the raw episode without review.")
        return EpisodeReviewDecision.SAVE
    print(f"episode {episode_number} 录制结束，完整保留 {raw_frames} 帧。")
    line = input("ENTER=保存完整 episode，r=重录当前 episode，q=结束并保存已有数据：")
    return review_decision_from_text(line)


def _target_to_action(q_target: np.ndarray, gripper_width: float, *, arm: str) -> dict[str, float]:
    action = {name: float(value) for name, value in zip(namespaced_joint_names(arm), q_target, strict=True)}
    action[namespaced_gripper_name(arm)] = float(gripper_width)
    return action


def _map_leader_to_arm_action(*, leader, mapping: SO101ToNeroMapping, arm: str) -> dict[str, float]:
    leader_action = leader.get_action()
    return map_so101_action_to_nero(leader_action, mapping=mapping, arm=arm)


def run_teleop_step(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    record: bool,
) -> dict | None:
    observation = robot.get_observation() if record else None
    right_action = _map_leader_to_arm_action(
        leader=right_leader,
        mapping=robot.right.mapping,
        arm="right",
    )
    left_action = _map_leader_to_arm_action(
        leader=left_leader,
        mapping=robot.left.mapping,
        arm="left",
    )
    sent_action = robot.send_action({**right_action, **left_action})
    if not record:
        return None
    return {
        "observation": observation,
        "action": sent_action,
        "flange_observation": robot.get_flange_state_observation(),
        "flange_action": robot.last_flange_action,
    }


def _sync_arm_to_current_leader(
    *,
    arm_runtime,
    leader,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    arm = arm_runtime.arm
    current = arm_runtime.read_joints()
    mapped = _map_leader_to_arm_action(leader=leader, mapping=arm_runtime.mapping, arm=arm)
    target = np.asarray([mapped[name] for name in namespaced_joint_names(arm)], dtype=float)
    gripper_width = mapped[namespaced_gripper_name(arm)]

    original_alpha = arm_runtime.config.command.alpha
    original_max_step = arm_runtime.config.command.max_step_rad
    try:
        arm_runtime.config.command.alpha = 1.0
        arm_runtime.config.command.max_step_rad = 0.0
        steps = max(int(round(takeover_time_s / takeover_dt_s)), 1)
        for q_cmd in smooth_takeover_commands(current, target, steps):
            arm_runtime.send_action(_target_to_action(q_cmd, gripper_width, arm=arm))
            precise_sleep(takeover_dt_s)
    finally:
        arm_runtime.config.command.alpha = original_alpha
        arm_runtime.config.command.max_step_rad = original_max_step


def sync_to_current_leaders(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    _sync_arm_to_current_leader(
        arm_runtime=robot.right,
        leader=right_leader,
        takeover_time_s=takeover_time_s,
        takeover_dt_s=takeover_dt_s,
    )
    _sync_arm_to_current_leader(
        arm_runtime=robot.left,
        leader=left_leader,
        takeover_time_s=takeover_time_s,
        takeover_dt_s=takeover_dt_s,
    )


def sync_to_current_leaders_before_next_episode(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    recorded_episodes: int,
    max_episodes: int,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    if recorded_episodes >= max_episodes:
        return

    logger.info(
        "Synchronizing Dual Nero to current SO101 leaders before the next raw episode (%d/%d saved).",
        recorded_episodes,
        max_episodes,
    )
    sync_to_current_leaders(
        right_leader=right_leader,
        left_leader=left_leader,
        robot=robot,
        takeover_time_s=takeover_time_s,
        takeover_dt_s=takeover_dt_s,
    )


def run_idle_teleop_until_start(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    fps: int,
    prompt: str,
) -> IdleTeleopDecision:
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; skipping idle teleop wait and starting the episode immediately.")
        return IdleTeleopDecision.START_RECORDING
    print(prompt)
    print("双臂主从同步保持开启；按 ENTER 开始录制，输入 q 后回车结束并保存已有数据。")
    del fps
    control_interval = robot.control_dt_s
    while True:
        start_loop_t = time.perf_counter()
        line = _stdin_line_if_ready()
        if line is not None:
            return idle_decision_from_text(line)
        run_teleop_step(right_leader=right_leader, left_leader=left_leader, robot=robot, record=False)
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))


def run_idle_teleop_until_shutdown(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    fps: int,
    prompt: str,
) -> None:
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; skipping final reset wait.")
        return
    print(prompt)
    print("双臂主从同步保持开启；请复位机械臂，复位完成后按 ENTER 关闭程序。")
    del fps
    control_interval = robot.control_dt_s
    while True:
        start_loop_t = time.perf_counter()
        if _stdin_stop_requested():
            return
        run_teleop_step(right_leader=right_leader, left_leader=left_leader, robot=robot, record=False)
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))


def collect_episode_buffer(
    *,
    right_leader,
    left_leader,
    robot: NeroDualRobot,
    episode_time_s: float,
    fps: int,
    stop_requested: Callable[[], bool] | None = None,
) -> list[dict]:
    frames: list[dict] = []
    control_interval = robot.control_dt_s
    sample_interval = 1.0 / fps
    start_episode_t = time.perf_counter()
    next_sample_t = start_episode_t
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
        should_record = now_s >= next_sample_t
        frame = run_teleop_step(
            right_leader=right_leader,
            left_leader=left_leader,
            robot=robot,
            record=should_record,
        )
        if frame is not None:
            frames.append(frame)
            next_sample_t += sample_interval
            while next_sample_t <= now_s:
                next_sample_t += sample_interval

        dt_s = time.perf_counter() - start_loop_t
        if dt_s > control_interval:
            logger.warning(
                "Nero dual control loop is running slower (%.1f Hz) than target control rate (%.1f Hz).",
                1.0 / dt_s,
                1.0 / control_interval,
            )
        precise_sleep(max(control_interval - dt_s, 0.0))

    return frames


def _resolve_flange_dataset_config(cfg: NeroRecordDualJointRawConfig) -> DatasetRecordConfig:
    base = cfg.dataset
    override = cfg.flange_dataset
    root = None
    if override is not None and override.root is not None:
        root = override.root
    elif base.root is not None:
        root_path = Path(base.root)
        name = root_path.name
        flange_name = name.replace("joint", "flange_pose") if "joint" in name else f"{name}_flange_pose"
        root = root_path.with_name(flange_name)
    repo_id = override.repo_id if override is not None and override.repo_id else base.repo_id
    if override is None and repo_id:
        owner, sep, name = repo_id.partition("/")
        flange_name = name.replace("joint", "flange_pose") if "joint" in name else f"{name}_flange_pose"
        repo_id = f"{owner}{sep}{flange_name}" if sep else flange_name
    return DatasetRecordConfig(
        repo_id=repo_id,
        single_task=base.single_task,
        root=root,
        fps=base.fps,
        episode_time_s=base.episode_time_s,
        reset_time_s=base.reset_time_s,
        num_episodes=base.num_episodes,
        video=base.video,
        push_to_hub=override.push_to_hub if override is not None else base.push_to_hub,
        private=override.private if override is not None else base.private,
        tags=override.tags if override is not None and override.tags is not None else base.tags,
        num_image_writer_processes=base.num_image_writer_processes,
        num_image_writer_threads_per_camera=base.num_image_writer_threads_per_camera,
        video_encoding_batch_size=base.video_encoding_batch_size,
        camera_encoder=base.camera_encoder,
        streaming_encoding=base.streaming_encoding,
        encoder_queue_maxsize=base.encoder_queue_maxsize,
        encoder_threads=base.encoder_threads,
    )


def _create_or_resume_dataset_for_config(
    cfg: NeroRecordDualJointRawConfig,
    dataset_cfg: DatasetRecordConfig,
    robot: NeroDualRobot,
    features: dict[str, dict],
):
    if cfg.dry_run:
        return None

    num_cameras = len(robot.cameras)
    if cfg.resume:
        if dataset_cfg.root is None:
            raise ValueError("--resume=true requires --dataset.root for Nero dual recording.")
        return LeRobotDataset.resume(
            dataset_cfg.repo_id,
            root=dataset_cfg.root,
            batch_encoding_size=dataset_cfg.video_encoding_batch_size,
            camera_encoder=dataset_cfg.camera_encoder,
            encoder_threads=dataset_cfg.encoder_threads,
            streaming_encoding=dataset_cfg.streaming_encoding,
            encoder_queue_maxsize=dataset_cfg.encoder_queue_maxsize,
            image_writer_processes=dataset_cfg.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * num_cameras
            if num_cameras > 0
            else 0,
        )

    repo_name = dataset_cfg.repo_id.split("/", 1)[-1]
    if repo_name.startswith("eval_"):
        raise ValueError(
            "Dataset names starting with 'eval_' are reserved for policy evaluation. "
            "Use a data-collection dataset name instead."
        )
    dataset_cfg.stamp_repo_id()
    return LeRobotDataset.create(
        dataset_cfg.repo_id,
        dataset_cfg.fps,
        features=features,
        root=dataset_cfg.root,
        robot_type=robot.name,
        use_videos=dataset_cfg.video,
        image_writer_processes=dataset_cfg.num_image_writer_processes if num_cameras > 0 else 0,
        image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * num_cameras
        if num_cameras > 0
        else 0,
        batch_encoding_size=dataset_cfg.video_encoding_batch_size,
        camera_encoder=dataset_cfg.camera_encoder,
        encoder_threads=dataset_cfg.encoder_threads,
        streaming_encoding=dataset_cfg.streaming_encoding,
        encoder_queue_maxsize=dataset_cfg.encoder_queue_maxsize,
    )


def _create_or_resume_dataset(cfg: NeroRecordDualJointRawConfig, robot: NeroDualRobot, features: dict[str, dict]):
    return _create_or_resume_dataset_for_config(cfg, cfg.dataset, robot, features)


def save_raw_episode_frames(*, dataset, frames: list[dict], task: str) -> None:
    for frame in frames:
        dataset.add_frame(
            flatten_episode_frame(
                frame,
                features=dataset.features,
                task=task,
            )
        )
    dataset.save_episode()


@parser.wrap()
def record_dual_joint_raw(cfg: NeroRecordDualJointRawConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroDualRobotConfig):
        raise TypeError(f"nero-record-dual-joint-raw requires --robot.type=nero_dual, got {cfg.robot.type!r}.")

    right_leader = make_teleoperator_from_config(cfg.right_leader)
    left_leader = make_teleoperator_from_config(cfg.left_leader)
    robot = NeroDualRobot(cfg.robot)
    dataset = None
    flange_dataset = None
    flange_cfg = _resolve_flange_dataset_config(cfg)

    try:
        flange_features = make_flange_dataset_features(robot, use_videos=flange_cfg.video)
        if cfg.record_joint_dataset:
            features = make_dataset_features(robot, use_videos=cfg.dataset.video)
            dataset = _create_or_resume_dataset(cfg, robot, features)
        flange_dataset = _create_or_resume_dataset_for_config(cfg, flange_cfg, robot, flange_features)

        right_leader.connect()
        left_leader.connect()
        robot.connect()
        input("即将根据当前双 SO101 位姿同步双 Nero。请确认机械臂周围安全，然后按 ENTER 开始同步。")
        sync_to_current_leaders(
            right_leader=right_leader,
            left_leader=left_leader,
            robot=robot,
            takeover_time_s=cfg.takeover_time_s,
            takeover_dt_s=cfg.takeover_dt_s,
        )
        logger.info("Dual Nero synchronized to both SO101 leaders. Entering idle teleop mode.")

        recorded_episodes = 0
        manager = VideoEncodingManager(dataset) if dataset is not None else None
        flange_manager = VideoEncodingManager(flange_dataset) if flange_dataset is not None else None
        if manager is not None:
            manager.__enter__()
        if flange_manager is not None:
            flange_manager.__enter__()
        try:
            while recorded_episodes < cfg.dataset.num_episodes:
                decision = run_idle_teleop_until_start(
                    right_leader=right_leader,
                    left_leader=left_leader,
                    robot=robot,
                    fps=cfg.dataset.fps,
                    prompt=(
                        f"准备录制 Nero 双臂 episode {recorded_episodes + 1}/"
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
                    right_leader=right_leader,
                    left_leader=left_leader,
                    robot=robot,
                    episode_time_s=cfg.dataset.episode_time_s,
                    fps=cfg.dataset.fps,
                    stop_requested=stop_requested,
                )
                if not frames:
                    logger.warning("Episode rejected: no frames collected.")
                    print("当前 episode 没有采集到帧，将重录当前 episode。")
                    sync_to_current_leaders_before_next_episode(
                        right_leader=right_leader,
                        left_leader=left_leader,
                        robot=robot,
                        recorded_episodes=recorded_episodes,
                        max_episodes=cfg.dataset.num_episodes,
                        takeover_time_s=cfg.takeover_time_s,
                        takeover_dt_s=cfg.takeover_dt_s,
                    )
                    continue
                frames = attach_flange_frames(frames, robot)

                review_decision = prompt_raw_episode_review(
                    episode_number=recorded_episodes + 1,
                    raw_frames=len(frames),
                )
                if review_decision is EpisodeReviewDecision.RERECORD:
                    logger.info("User requested re-record for episode %d.", recorded_episodes + 1)
                    sync_to_current_leaders_before_next_episode(
                        right_leader=right_leader,
                        left_leader=left_leader,
                        robot=robot,
                        recorded_episodes=recorded_episodes,
                        max_episodes=cfg.dataset.num_episodes,
                        takeover_time_s=cfg.takeover_time_s,
                        takeover_dt_s=cfg.takeover_dt_s,
                    )
                    continue
                if review_decision is EpisodeReviewDecision.FINISH_AND_EXIT:
                    logger.info(
                        "User requested early finish during episode review after %d/%d saved episodes.",
                        recorded_episodes,
                        cfg.dataset.num_episodes,
                    )
                    break

                if cfg.record_joint_dataset and dataset is not None and flange_dataset is not None:
                    save_raw_episode_frames_pair(
                        joint_dataset=dataset,
                        flange_dataset=flange_dataset,
                        frames=frames,
                        task=cfg.dataset.single_task,
                    )
                elif flange_dataset is not None:
                    save_raw_flange_episode_frames(
                        flange_dataset=flange_dataset,
                        frames=frames,
                        task=cfg.dataset.single_task,
                    )
                else:
                    logger.info("Dry run raw episode kept all %d frames.", len(frames))

                recorded_episodes += 1
                sync_to_current_leaders_before_next_episode(
                    right_leader=right_leader,
                    left_leader=left_leader,
                    robot=robot,
                    recorded_episodes=recorded_episodes,
                    max_episodes=cfg.dataset.num_episodes,
                    takeover_time_s=cfg.takeover_time_s,
                    takeover_dt_s=cfg.takeover_dt_s,
                )

            run_idle_teleop_until_shutdown(
                right_leader=right_leader,
                left_leader=left_leader,
                robot=robot,
                fps=cfg.dataset.fps,
                prompt=(
                    f"录制结束：已保存 {recorded_episodes}/"
                    f"{cfg.dataset.num_episodes} 条双臂 episode。"
                ),
            )
        except BaseException as exc:
            if manager is not None:
                manager.__exit__(type(exc), exc, exc.__traceback__)
                manager = None
            if flange_manager is not None:
                flange_manager.__exit__(type(exc), exc, exc.__traceback__)
                flange_manager = None
            raise
        finally:
            if manager is not None:
                manager.__exit__(None, None, None)
            if flange_manager is not None:
                flange_manager.__exit__(None, None, None)

        if dataset is not None and cfg.dataset.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        if flange_dataset is not None and flange_cfg.push_to_hub and flange_dataset.num_episodes > 0:
            flange_dataset.push_to_hub(tags=flange_cfg.tags, private=flange_cfg.private)
    finally:
        if dataset is not None:
            dataset.finalize()
        if flange_dataset is not None:
            flange_dataset.finalize()
        robot.disconnect()
        if right_leader.is_connected:
            right_leader.disconnect()
        if left_leader.is_connected:
            left_leader.disconnect()

    return dataset


def main() -> None:
    register_third_party_plugins()
    record_dual_joint_raw()


if __name__ == "__main__":
    main()
