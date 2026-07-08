import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .async_client import NeroAsyncSafetyConfig, SafeNeroRobot
from .config_nero import NeroDualRobotConfig
from .mapping import namespaced_gripper_name, namespaced_joint_names
from .prepare_sync import smooth_takeover_commands
from .robot_nero_dual import NeroDualRobot

logger = logging.getLogger(__name__)


@dataclass
class NeroReplayDatasetConfig:
    root: Path
    episode: int = 0
    fps: int | None = None


@dataclass
class NeroReplayDualJointConfig:
    robot: RobotConfig
    dataset: NeroReplayDatasetConfig
    takeover_time_s: float = 2.0
    takeover_dt_s: float = 0.02
    high_rate_control: bool = False
    high_rate_dt_s: float = 0.006
    max_replay_step_rad: float = 0.1
    max_replay_gripper_step_m: float = 0.1
    max_executor_step_rad: float = 0.02
    max_executor_gripper_step_m: float = 0.005


def load_action_names(root: Path) -> list[str]:
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return [
            *namespaced_joint_names("right"),
            *namespaced_joint_names("left"),
            namespaced_gripper_name("right"),
            namespaced_gripper_name("left"),
        ]

    import json

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("features", {}).get(ACTION, {}).get("names")
    if not names:
        raise ValueError(f"Dataset metadata does not define action names: {info_path}")
    return list(names)


def load_dataset_fps(root: Path) -> int:
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return 30

    import json

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    return int(info.get("fps", 30))


def load_episode_actions(root: Path, *, episode: int) -> np.ndarray:
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(path, columns=["episode_index", "frame_index", ACTION]) for path in parquet_files])
    episode_df = df[df["episode_index"] == episode].sort_values("frame_index")
    if episode_df.empty:
        available = sorted(int(value) for value in df["episode_index"].unique())
        raise ValueError(f"Episode {episode} not found in {root}. Available episodes: {available}")

    actions = np.stack(episode_df[ACTION].to_numpy()).astype(float)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D action array, got shape {actions.shape}")
    return actions


def action_dict_from_vector(action: np.ndarray, action_names: list[str]) -> dict[str, float]:
    if len(action) != len(action_names):
        raise ValueError(f"Action vector has {len(action)} values, but dataset defines {len(action_names)} names.")
    return {name: float(action[idx]) for idx, name in enumerate(action_names)}


def _sync_to_first_action(
    robot,
    first_action: dict[str, float],
    *,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    right_current = robot.right.read_joints()
    left_current = robot.left.read_joints()
    right_target = np.asarray([first_action[name] for name in namespaced_joint_names("right")], dtype=float)
    left_target = np.asarray([first_action[name] for name in namespaced_joint_names("left")], dtype=float)

    logger.info(
        "Current-to-first max error rad: right=%.4f left=%.4f",
        float(np.max(np.abs(right_target - right_current))),
        float(np.max(np.abs(left_target - left_current))),
    )

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
            command = {
                name: float(value)
                for name, value in zip(namespaced_joint_names("right"), right_cmd, strict=True)
            }
            command.update(
                {
                    name: float(value)
                    for name, value in zip(namespaced_joint_names("left"), left_cmd, strict=True)
                }
            )
            command[namespaced_gripper_name("right")] = first_action[namespaced_gripper_name("right")]
            command[namespaced_gripper_name("left")] = first_action[namespaced_gripper_name("left")]
            robot.send_action(command)
            precise_sleep(takeover_dt_s)
    finally:
        robot.right.config.command.alpha, robot.right.config.command.max_step_rad = original["right"]
        robot.left.config.command.alpha, robot.left.config.command.max_step_rad = original["left"]


def replay_episode_actions(
    robot,
    actions: np.ndarray,
    action_names: list[str],
    *,
    fps: int,
    takeover_time_s: float,
    takeover_dt_s: float,
    high_rate_control: bool = False,
) -> None:
    first_action = action_dict_from_vector(actions[0], action_names)
    _sync_to_first_action(
        robot.robot if high_rate_control and isinstance(robot, SafeNeroRobot) else robot,
        first_action,
        takeover_time_s=takeover_time_s,
        takeover_dt_s=takeover_dt_s,
    )
    if high_rate_control and isinstance(robot, SafeNeroRobot):
        robot.set_last_action(first_action)

    for idx, action_vector in enumerate(actions):
        start_t = time.perf_counter()
        robot.send_action(action_dict_from_vector(action_vector, action_names))
        if idx % 100 == 0 or idx == len(actions) - 1:
            logger.info("Replay frame %d/%d", idx, len(actions))
        precise_sleep(max(1.0 / fps - (time.perf_counter() - start_t), 0.0))


@parser.wrap()
def replay_dual_joint(cfg: NeroReplayDualJointConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroDualRobotConfig):
        raise TypeError(f"nero-replay-dual-joint requires --robot.type=nero_dual, got {cfg.robot.type!r}.")

    actions = load_episode_actions(Path(cfg.dataset.root), episode=cfg.dataset.episode)
    action_names = load_action_names(Path(cfg.dataset.root))
    fps = cfg.dataset.fps or load_dataset_fps(Path(cfg.dataset.root))

    logger.info(
        "Replaying Nero dual dataset root=%s episode=%d frames=%d fps=%d",
        cfg.dataset.root,
        cfg.dataset.episode,
        len(actions),
        fps,
    )

    robot = NeroDualRobot(cfg.robot)
    robot.connect()
    replay_robot = robot
    if cfg.high_rate_control:
        first_action = action_dict_from_vector(actions[0], action_names)
        safety = NeroAsyncSafetyConfig(
            fixed_ready_pose=first_action,
            high_rate_control=True,
            high_rate_dt_s=cfg.high_rate_dt_s,
            max_policy_step_rad=cfg.max_replay_step_rad,
            max_gripper_step_m=cfg.max_replay_gripper_step_m,
            max_executor_step_rad=cfg.max_executor_step_rad,
            max_executor_gripper_step_m=cfg.max_executor_gripper_step_m,
        )
        replay_robot = SafeNeroRobot(robot, safety)
    try:
        if isinstance(replay_robot, SafeNeroRobot):
            replay_robot.start_high_rate_executor()
        replay_episode_actions(
            replay_robot,
            actions,
            action_names,
            fps=fps,
            takeover_time_s=cfg.takeover_time_s,
            takeover_dt_s=cfg.takeover_dt_s,
            high_rate_control=cfg.high_rate_control,
        )
    finally:
        if isinstance(replay_robot, SafeNeroRobot):
            replay_robot.stop_high_rate_executor()
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    replay_dual_joint()


if __name__ == "__main__":
    main()
