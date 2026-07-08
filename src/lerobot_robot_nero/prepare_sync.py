import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np

import lerobot_teleoperator_so101_8dof  # noqa: F401
from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .config_nero import NeroRobotConfig
from .mapping import SO101ToNeroMapping, map_so101_action_to_nero, namespaced_gripper_name, namespaced_joint_names
from .robot_nero import NeroRobot

logger = logging.getLogger(__name__)


@dataclass
class NeroPrepareSyncConfig:
    leader: TeleoperatorConfig
    robot: RobotConfig
    takeover_time_s: float = 2.0
    takeover_dt_s: float = 0.02
    follow_after_takeover: bool = False


def confirm_nero_enable() -> bool:
    answer = input("是否现在 enable Nero 机械臂？[Y/n]: ").strip().lower()
    return answer in ("", "y", "yes", "是", "好", "确认")


def smooth_takeover_commands(current: np.ndarray, target: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        return [np.asarray(target, dtype=float)]
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    commands = []
    for idx in range(1, steps + 1):
        t = idx / steps
        weight = t * t * (3.0 - 2.0 * t)
        commands.append(current + weight * (target - current))
    return commands


def _target_to_action(q_target: np.ndarray, gripper_width: float, *, arm: str) -> dict[str, float]:
    action = {name: float(value) for name, value in zip(namespaced_joint_names(arm), q_target, strict=True)}
    action[namespaced_gripper_name(arm)] = float(gripper_width)
    return action


@parser.wrap()
def prepare_sync(cfg: NeroPrepareSyncConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroRobotConfig):
        raise TypeError(f"nero-prepare-sync requires --robot.type=nero, got {cfg.robot.type!r}.")

    mapping = SO101ToNeroMapping.from_config(cfg.robot.mapping)
    leader = make_teleoperator_from_config(cfg.leader)
    robot = NeroRobot(cfg.robot)
    original_alpha = robot.config.command.alpha
    original_max_step = robot.config.command.max_step_rad

    try:
        if not confirm_nero_enable():
            logger.info("Nero enable skipped by user; exiting prepare sync.")
            return

        leader.connect()
        robot.connect()
        current = robot._read_joints()
        input(
            "Move the SO101 leader to the intended synchronized start pose, "
            "then press ENTER to smoothly move Nero to that target."
        )

        leader_action = leader.get_action()
        mapped = map_so101_action_to_nero(leader_action, mapping=mapping, arm=cfg.robot.mapping.arm)
        target = np.asarray([mapped[name] for name in namespaced_joint_names(cfg.robot.mapping.arm)], dtype=float)
        gripper_width = mapped[namespaced_gripper_name(cfg.robot.mapping.arm)]

        steps = max(int(round(cfg.takeover_time_s / cfg.takeover_dt_s)), 1)
        robot.config.command.alpha = 1.0
        robot.config.command.max_step_rad = 0.0
        for q_cmd in smooth_takeover_commands(current, target, steps):
            robot.send_action(_target_to_action(q_cmd, gripper_width, arm=cfg.robot.mapping.arm))
            precise_sleep(cfg.takeover_dt_s)
        robot.config.command.alpha = original_alpha
        robot.config.command.max_step_rad = original_max_step

        logger.info("Nero synchronized to the SO101 leader target.")

        while cfg.follow_after_takeover:
            started = time.perf_counter()
            leader_action = leader.get_action()
            mapped = map_so101_action_to_nero(leader_action, mapping=mapping, arm=cfg.robot.mapping.arm)
            robot.send_action(mapped)
            precise_sleep(max(cfg.takeover_dt_s - (time.perf_counter() - started), 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()
        if leader.is_connected:
            leader.disconnect()


def main() -> None:
    register_third_party_plugins()
    prepare_sync()


if __name__ == "__main__":
    main()
