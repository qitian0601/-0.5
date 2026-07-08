import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Callable

import numpy as np

from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .config_nero import NeroDualRobotConfig
from .mapping import namespaced_gripper_name, namespaced_joint_names
from .prepare_sync import smooth_takeover_commands
from .robot_nero_dual import NeroDualRobot

logger = logging.getLogger(__name__)

DEFAULT_RETURN_POSE = {
    "right_nero_joint_1": -0.034487706019407954,
    "right_nero_joint_2": -1.8699457605867247,
    "right_nero_joint_3": 0.007382742735936014,
    "right_nero_joint_4": 2.1975266078935407,
    "right_nero_joint_5": -0.015184364492350668,
    "right_nero_joint_6": -0.03192207201897629,
    "right_nero_joint_7": 1.705814997729178,
    "right_gripper_width": 0.02852,
    "left_nero_joint_1": 0.044226643245536316,
    "left_nero_joint_2": -1.805804910575933,
    "left_nero_joint_3": 0.06731734924942129,
    "left_nero_joint_4": 2.2148228207808045,
    "left_nero_joint_5": 0.08609709200088027,
    "left_nero_joint_6": -0.026127578902355116,
    "left_nero_joint_7": 1.713791152410792,
    "left_gripper_width": 0.023531,
}


@dataclass
class NeroReturnDisableDualConfig:
    robot: RobotConfig = field(default_factory=NeroDualRobotConfig)
    return_pose: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RETURN_POSE))
    takeover_time_s: float = 6.0
    takeover_dt_s: float = 0.02
    tolerance_rad: float = 0.05
    dry_run: bool = False
    print_current_pose: bool = False


def _require_return_pose_keys(pose: dict[str, float]) -> None:
    names = [
        *namespaced_joint_names("right"),
        *namespaced_joint_names("left"),
        namespaced_gripper_name("right"),
        namespaced_gripper_name("left"),
    ]
    missing = [name for name in names if name not in pose]
    if missing:
        raise KeyError(f"Missing Nero return pose keys: {missing}")


def _arm_enabled(arm) -> bool:
    if arm.robot is None:
        raise RuntimeError(f"{arm.arm} Nero SDK robot is not initialized.")
    return all(arm.robot.get_joints_enable_status_list())


def assert_dual_arm_enabled(robot: NeroDualRobot) -> None:
    disabled = [name for name in ("right", "left") if not _arm_enabled(getattr(robot, name))]
    if disabled:
        raise RuntimeError(
            f"Nero arm(s) not fully enabled: {disabled}. Refusing to move before disable."
        )


def _connect_arm_without_enabling(arm) -> None:
    arm.robot = arm._make_sdk_robot()
    arm.end_effector = arm.robot.init_effector(arm.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    arm.robot.connect()


def _take_can_control_without_enabling(arm) -> None:
    if arm.robot is None:
        raise RuntimeError(f"{arm.arm} Nero SDK robot is not initialized.")
    arm.robot._msg_mode.ctrl_mode = 0x01
    arm.robot._msg_mode.move_mode = 0x01
    arm.robot._msg_mode.mit_mode = 0x00
    arm.robot._msg_mode.enable_can_push = 0x01
    arm.robot._set_mode()
    time.sleep(0.2)
    arm.robot.set_follower_mode()
    time.sleep(0.2)
    arm.robot.set_motion_mode("js")
    time.sleep(0.2)


def _current_pose(robot: NeroDualRobot) -> dict[str, float]:
    pose: dict[str, float] = {}
    for arm_name in ("right", "left"):
        arm = getattr(robot, arm_name)
        joints = arm.read_joints()
        for name, value in zip(namespaced_joint_names(arm_name), joints, strict=True):
            pose[name] = float(value)
        gripper_name = namespaced_gripper_name(arm_name)
        gripper_width = arm._gripper_width
        if arm.end_effector is not None:
            status = arm.end_effector.get_gripper_status()
            msg = getattr(status, "msg", None)
            if msg is not None and hasattr(msg, "value"):
                gripper_width = float(msg.value)
        pose[gripper_name] = float(gripper_width)
    return pose


def _target_arm_joints(pose: dict[str, float], arm: str) -> np.ndarray:
    return np.asarray([float(pose[name]) for name in namespaced_joint_names(arm)], dtype=float)


def _clip_pose_to_robot_limits(robot: NeroDualRobot, pose: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    clipped_pose = dict(pose)
    clipped_names: list[str] = []
    for arm_name in ("right", "left"):
        arm = getattr(robot, arm_name)
        joints = _target_arm_joints(pose, arm_name)
        clipped_joints = np.clip(joints, arm.mapping.nero_limit_low, arm.mapping.nero_limit_high)
        for name, original, clipped in zip(namespaced_joint_names(arm_name), joints, clipped_joints, strict=True):
            clipped_pose[name] = float(clipped)
            if not np.isclose(original, clipped):
                clipped_names.append(name)
        gripper_name = namespaced_gripper_name(arm_name)
        gripper_width = float(pose[gripper_name])
        clipped_gripper = float(
            np.clip(gripper_width, arm.mapping.nero_gripper_width_min, arm.mapping.nero_gripper_width_max)
        )
        clipped_pose[gripper_name] = clipped_gripper
        if not np.isclose(gripper_width, clipped_gripper):
            clipped_names.append(gripper_name)
    return clipped_pose, clipped_names


def _action_from_arm_joints(
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


def _ask_disable_after_unreachable_pose(max_error: float, clipped_names: list[str]) -> bool:
    clipped = ", ".join(clipped_names) if clipped_names else "none"
    logger.warning(
        "Nero reached the closest limited pose, but original return pose is still %.4f rad/m away. "
        "Clipped fields: %s",
        max_error,
        clipped,
    )
    answer = input("Return pose reached joint/gripper limits. Disable Nero now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _disable_gripper(arm) -> None:
    if arm.end_effector is None:
        logger.warning("%s Nero gripper effector is not initialized; skipping gripper disable.", arm.arm)
        return
    if not arm.end_effector.disable_gripper():
        logger.warning("%s Nero gripper disable did not report success.", arm.arm)


def return_to_pose_and_disable(
    robot: NeroDualRobot,
    pose: dict[str, float],
    *,
    takeover_time_s: float,
    takeover_dt_s: float,
    tolerance_rad: float,
    dry_run: bool = False,
    confirm_disable_fn: Callable[[float, list[str]], bool] = _ask_disable_after_unreachable_pose,
) -> None:
    _require_return_pose_keys(pose)
    assert_dual_arm_enabled(robot)

    reachable_pose, clipped_names = _clip_pose_to_robot_limits(robot, pose)
    if clipped_names:
        logger.warning("Nero return pose exceeds configured limits; clipped fields: %s", clipped_names)

    right_current = robot.right.read_joints()
    left_current = robot.left.read_joints()
    right_target = _target_arm_joints(reachable_pose, "right")
    left_target = _target_arm_joints(reachable_pose, "left")

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
            action = _action_from_arm_joints(right_cmd, left_cmd, reachable_pose)
            if dry_run:
                logger.info("Dry-run return action: %s", action)
            else:
                robot.send_action(action)
            precise_sleep(takeover_dt_s)
    finally:
        robot.right.config.command.alpha, robot.right.config.command.max_step_rad = original["right"]
        robot.left.config.command.alpha, robot.left.config.command.max_step_rad = original["left"]

    if not dry_run:
        right_actual = robot.right.read_joints()
        left_actual = robot.left.read_joints()
        right_reachable_error = float(np.max(np.abs(right_target - right_actual)))
        left_reachable_error = float(np.max(np.abs(left_target - left_actual)))
        reachable_error = max(right_reachable_error, left_reachable_error)
        if reachable_error > tolerance_rad:
            raise RuntimeError(
                f"Nero reachable return pose error {reachable_error:.4f} rad exceeds tolerance "
                f"{tolerance_rad:.4f} rad."
            )
        right_original_error = float(np.max(np.abs(_target_arm_joints(pose, "right") - right_actual)))
        left_original_error = float(np.max(np.abs(_target_arm_joints(pose, "left") - left_actual)))
        original_error = max(right_original_error, left_original_error)
        if original_error > tolerance_rad and not confirm_disable_fn(original_error, clipped_names):
            logger.warning("Nero remains enabled because disable was not confirmed.")
            return
        _disable_gripper(robot.right)
        _disable_gripper(robot.left)
        robot.right.robot.disable(255)
        robot.left.robot.disable(255)


@parser.wrap()
def return_disable_dual(cfg: NeroReturnDisableDualConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    register_third_party_plugins()
    if not isinstance(cfg.robot, NeroDualRobotConfig):
        raise TypeError(f"nero-return-disable-dual requires --robot.type=nero_dual, got {cfg.robot.type!r}.")

    robot = NeroDualRobot(cfg.robot)
    for arm in (robot.right, robot.left):
        _connect_arm_without_enabling(arm)
        _take_can_control_without_enabling(arm)
    robot._arms_connected = True
    robot._is_connected = True

    try:
        current_pose = _current_pose(robot)
        logger.info("Current Nero pose: %s", current_pose)
        if cfg.print_current_pose:
            print(current_pose)
        if cfg.print_current_pose and cfg.dry_run:
            return

        return_to_pose_and_disable(
            robot,
            cfg.return_pose,
            takeover_time_s=cfg.takeover_time_s,
            takeover_dt_s=cfg.takeover_dt_s,
            tolerance_rad=cfg.tolerance_rad,
            dry_run=cfg.dry_run,
        )
        if cfg.dry_run:
            logger.info(
                "Dry-run complete; Nero was not moved or disabled. right=%s left=%s",
                robot.right.robot.get_joints_enable_status_list(),
                robot.left.robot.get_joints_enable_status_list(),
            )
        else:
            logger.info(
                "Nero disabled. right=%s left=%s",
                robot.right.robot.get_joints_enable_status_list(),
                robot.left.robot.get_joints_enable_status_list(),
            )
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    return_disable_dual()


if __name__ == "__main__":
    main()
