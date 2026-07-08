import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .async_client import DEFAULT_DUAL_ACTION_NAMES
from .config_nero import NeroDualRobotConfig
from .mapping import namespaced_gripper_name, namespaced_joint_names
from .prepare_sync import smooth_takeover_commands
from .replay_dual_joint import action_dict_from_vector
from .robot_nero_dual import NeroDualRobot
from .trace import load_replay_actions

logger = logging.getLogger(__name__)


@dataclass
class NeroInferenceTraceReplaySourceConfig:
    path: Path
    fps: int = 180


@dataclass
class NeroReplayInferenceTraceConfig:
    robot: RobotConfig
    trace: NeroInferenceTraceReplaySourceConfig
    takeover_time_s: float = 2.0
    takeover_dt_s: float = 0.02


def load_trace_replay_actions(path: Path) -> np.ndarray:
    return load_replay_actions(path, DEFAULT_DUAL_ACTION_NAMES)


def _sync_to_first_action(
    robot,
    first_action: dict[str, float],
    *,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    if takeover_time_s <= 0:
        return

    right_current = robot.right.read_joints()
    left_current = robot.left.read_joints()
    right_target = np.asarray([first_action[name] for name in namespaced_joint_names("right")], dtype=float)
    left_target = np.asarray([first_action[name] for name in namespaced_joint_names("left")], dtype=float)
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


def replay_trace_actions(
    robot,
    actions: np.ndarray,
    *,
    fps: int,
    takeover_time_s: float,
    takeover_dt_s: float,
) -> None:
    first_action = action_dict_from_vector(actions[0], list(DEFAULT_DUAL_ACTION_NAMES))
    original = {
        "right": (robot.right.config.command.alpha, robot.right.config.command.max_step_rad),
        "left": (robot.left.config.command.alpha, robot.left.config.command.max_step_rad),
    }
    try:
        robot.right.config.command.alpha = 1.0
        robot.left.config.command.alpha = 1.0
        robot.right.config.command.max_step_rad = 0.0
        robot.left.config.command.max_step_rad = 0.0
        _sync_to_first_action(
            robot,
            first_action,
            takeover_time_s=takeover_time_s,
            takeover_dt_s=takeover_dt_s,
        )
        for idx, action_vector in enumerate(actions):
            start_t = time.perf_counter()
            robot.send_action(action_dict_from_vector(action_vector, list(DEFAULT_DUAL_ACTION_NAMES)))
            if idx % 500 == 0 or idx == len(actions) - 1:
                logger.info("Replay trace frame %d/%d", idx, len(actions))
            precise_sleep(max(1.0 / fps - (time.perf_counter() - start_t), 0.0))
    finally:
        robot.right.config.command.alpha, robot.right.config.command.max_step_rad = original["right"]
        robot.left.config.command.alpha, robot.left.config.command.max_step_rad = original["left"]


@parser.wrap()
def replay_inference_trace(cfg: NeroReplayInferenceTraceConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroDualRobotConfig):
        raise TypeError(f"nero-replay-inference-trace requires --robot.type=nero_dual, got {cfg.robot.type!r}.")

    actions = load_trace_replay_actions(Path(cfg.trace.path))
    logger.info("Replaying Nero inference trace path=%s frames=%d fps=%d", cfg.trace.path, len(actions), cfg.trace.fps)
    robot = NeroDualRobot(cfg.robot)
    robot.connect()
    try:
        replay_trace_actions(
            robot,
            actions,
            fps=cfg.trace.fps,
            takeover_time_s=cfg.takeover_time_s,
            takeover_dt_s=cfg.takeover_dt_s,
        )
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    replay_inference_trace()


if __name__ == "__main__":
    main()
