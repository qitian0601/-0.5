from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_nero import NeroArmConfig, NeroDualRobotConfig
from .mapping import SO101ToNeroMapping, namespaced_action_names, namespaced_gripper_name, namespaced_joint_names

logger = logging.getLogger(__name__)

FLANGE_POSE_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def namespaced_flange_pose_names(arm: str) -> list[str]:
    return [f"{arm}_flange_{component}" for component in FLANGE_POSE_COMPONENTS]


class _NeroArmRuntime:
    def __init__(self, config: NeroArmConfig):
        self.config = config
        self.arm = config.mapping.arm
        self.mapping = SO101ToNeroMapping.from_config(config.mapping)
        self.robot: Any | None = None
        self.end_effector: Any | None = None
        self.tracer: Any | None = None
        self._last_command: np.ndarray | None = None
        self._gripper_width = self.mapping.nero_gripper_width_min

    def connect(self) -> None:
        logger.info(
            "Connecting %s Nero arm on %s with move_method=%s, speed_percent=%s.",
            self.arm,
            self.config.connection.channel,
            self.config.command.move_method,
            self.config.connection.speed_percent,
        )
        self.robot = self._make_sdk_robot()
        self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()
        self._take_can_control()
        while not self._enable_all_joints():
            logger.info("Waiting for %s Nero arm enable...", self.arm)
            time.sleep(self.config.connection.enable_retry_s)
        self.robot.set_speed_percent(self.config.connection.speed_percent)
        self._last_command = self.read_joints()

    def _take_can_control(self) -> None:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")

        self.robot._msg_mode.ctrl_mode = 0x01
        self.robot._msg_mode.move_mode = 0x01
        self.robot._msg_mode.mit_mode = 0x00
        self.robot._msg_mode.enable_can_push = 0x01
        self.robot._set_mode()
        time.sleep(0.2)
        self.robot.set_follower_mode()
        time.sleep(0.2)
        self.robot.set_motion_mode("js")
        time.sleep(0.2)
        if self.config.connection.reset_on_connect:
            self.robot.reset()
            time.sleep(0.2)
        self.robot.clear_joint_error(255)
        time.sleep(0.2)

    def _enable_all_joints(self) -> bool:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")

        self.robot.enable()
        time.sleep(1.0)
        for _ in range(3):
            if not all(self.robot.get_joints_enable_status_list()):
                return False
            time.sleep(0.2)
        return True

    def _make_sdk_robot(self):
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        except ImportError as exc:
            raise ImportError(
                "NeroDualRobot requires the Nero SDK package 'pyAgxArm'. Install/configure the Nero SDK "
                "before using --robot.type=nero_dual."
            ) from exc

        firmware = getattr(NeroFW, self.config.connection.firmware_version)
        arm_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=firmware,
            interface=self.config.connection.interface,
            channel=self.config.connection.channel,
        )
        return AgxArmFactory.create_arm(arm_config)

    def read_joints(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")
        joints = self.robot.get_joint_angles()
        while joints is None:
            time.sleep(0.01)
            joints = self.robot.get_joint_angles()
        joints = getattr(joints, "msg", joints)
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,):
            raise ValueError(f"{self.arm} Nero SDK returned joint shape {joints.shape}; expected 7 values.")
        return joints

    def read_joints_once(self) -> np.ndarray | None:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")
        joints = self.robot.get_joint_angles()
        if joints is None:
            return None
        joints = getattr(joints, "msg", joints)
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,):
            raise ValueError(f"{self.arm} Nero SDK returned joint shape {joints.shape}; expected 7 values.")
        return joints

    def observation(self) -> dict[str, float]:
        joints = self.read_joints()
        observation = {
            name: float(value) for name, value in zip(namespaced_joint_names(self.arm), joints, strict=True)
        }
        observation[namespaced_gripper_name(self.arm)] = float(self._gripper_width)
        return observation

    def read_flange_pose(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")
        pose = self.robot.get_flange_pose()
        while pose is None:
            time.sleep(0.01)
            pose = self.robot.get_flange_pose()
        pose = getattr(pose, "msg", pose)
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,):
            raise ValueError(f"{self.arm} Nero SDK returned flange pose shape {pose.shape}; expected 6 values.")
        return pose

    def flange_pose_from_joints(self, joints: np.ndarray) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")
        pose = self.robot.fk(np.asarray(joints, dtype=float).tolist())
        pose = getattr(pose, "msg", pose)
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,):
            raise ValueError(f"{self.arm} Nero SDK FK returned flange pose shape {pose.shape}; expected 6 values.")
        return pose

    def flange_observation(self) -> dict[str, float]:
        pose = self.read_flange_pose()
        observation = {
            name: float(value)
            for name, value in zip(namespaced_flange_pose_names(self.arm), pose, strict=True)
        }
        observation[namespaced_gripper_name(self.arm)] = float(self._gripper_width)
        return observation

    def flange_action_from_joints(self, joints: np.ndarray, gripper_width: float) -> dict[str, float]:
        pose = self.flange_pose_from_joints(joints)
        action = {
            name: float(value)
            for name, value in zip(namespaced_flange_pose_names(self.arm), pose, strict=True)
        }
        action[namespaced_gripper_name(self.arm)] = float(gripper_width)
        return action

    def _command_vector_from_action(self, action: RobotAction) -> tuple[np.ndarray, float]:
        missing = [name for name in namespaced_action_names(self.arm) if name not in action]
        if missing:
            raise KeyError(f"Missing {self.arm} Nero action keys: {missing}")
        joints = np.asarray([float(action[name]) for name in namespaced_joint_names(self.arm)], dtype=float)
        gripper_width = float(action[namespaced_gripper_name(self.arm)])
        joints = np.clip(joints, self.mapping.nero_limit_low, self.mapping.nero_limit_high)
        gripper_width = float(
            np.clip(gripper_width, self.mapping.nero_gripper_width_min, self.mapping.nero_gripper_width_max)
        )
        return joints, gripper_width

    def smooth_command(self, target: np.ndarray) -> np.ndarray:
        if self._last_command is None:
            self._last_command = self.read_joints()

        alpha = float(np.clip(self.config.command.alpha, 0.0, 1.0))
        smoothed = self._last_command + alpha * (target - self._last_command)

        max_step = self.config.command.max_step_rad
        if max_step is not None and max_step > 0:
            delta = np.clip(smoothed - self._last_command, -max_step, max_step)
            smoothed = self._last_command + delta

        smoothed = np.clip(smoothed, self.mapping.nero_limit_low, self.mapping.nero_limit_high)
        return smoothed

    def send_action(
        self,
        action: RobotAction,
        *,
        send_gripper: bool = True,
        read_feedback: bool = True,
    ) -> RobotAction:
        if self.robot is None or self.end_effector is None:
            raise RuntimeError(f"{self.arm} Nero SDK robot is not initialized.")

        target_joints, gripper_width = self._command_vector_from_action(action)
        command_joints = self.smooth_command(target_joints)
        if self.tracer is not None:
            self.tracer.record(
                "arm_command",
                {
                    "arm": self.arm,
                    "move_method": self.config.command.move_method,
                    "target_joints": target_joints,
                    "command_joints": command_joints,
                    "gripper_width": gripper_width,
                },
            )
        if self.config.command.move_method == "move_js":
            self.robot.move_js(command_joints.tolist())
        elif self.config.command.move_method == "move_j":
            self.robot.move_j(command_joints.tolist())
        else:
            raise ValueError(f"Unsupported Nero move_method: {self.config.command.move_method!r}.")
        if send_gripper:
            self.end_effector.move_gripper_m(
                value=gripper_width,
                force=self.config.connection.gripper_force,
            )
        feedback_joints = self.read_joints_once() if read_feedback else None
        if self.tracer is not None:
            self.tracer.record(
                "arm_feedback_after_command",
                {
                    "arm": self.arm,
                    "command_joints": command_joints,
                    "feedback_joints": feedback_joints,
                    "joint_error": None if feedback_joints is None else feedback_joints - command_joints,
                    "gripper_command_width": gripper_width,
                    "gripper_feedback_width": None,
                    "send_gripper": send_gripper,
                    "read_feedback": read_feedback,
                },
            )

        self._last_command = command_joints
        self._gripper_width = gripper_width
        sent = {
            name: float(value)
            for name, value in zip(namespaced_joint_names(self.arm), command_joints, strict=True)
        }
        sent[namespaced_gripper_name(self.arm)] = gripper_width
        sent["_flange_action"] = self.flange_action_from_joints(command_joints, gripper_width)
        return sent

    def disconnect(self) -> None:
        if self.robot is not None:
            disconnect = getattr(self.robot, "disconnect", None)
            if callable(disconnect):
                disconnect()


class NeroDualRobot(Robot):
    config_class = NeroDualRobotConfig
    name = "nero_dual"

    def __init__(self, config: NeroDualRobotConfig):
        super().__init__(config)
        self.config = config
        self.right = _NeroArmRuntime(config.right)
        self.left = _NeroArmRuntime(config.left)
        self.arms = {"right": self.right, "left": self.left}
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        self._arms_connected = False
        self._last_flange_action: dict[str, float] | None = None

    def set_tracer(self, tracer: Any | None) -> None:
        self.right.tracer = tracer
        self.left.tracer = tracer

    @property
    def control_dt_s(self) -> float:
        return min(self.right.config.command.control_dt_s, self.left.config.command.control_dt_s)

    @property
    def _joint_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for arm in ("right", "left"):
            features.update({name: float for name in namespaced_joint_names(arm)})
        return features

    @property
    def _gripper_features(self) -> dict[str, type]:
        return {namespaced_gripper_name(arm): float for arm in ("right", "left")}

    @property
    def _flange_pose_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for arm in ("right", "left"):
            features.update({name: float for name in namespaced_flange_pose_names(arm)})
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_features, **self._gripper_features, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._joint_features, **self._gripper_features}

    @cached_property
    def flange_observation_features(self) -> dict[str, type | tuple]:
        return {**self._flange_pose_features, **self._gripper_features, **self._cameras_ft}

    @cached_property
    def flange_action_features(self) -> dict[str, type]:
        return {**self._flange_pose_features, **self._gripper_features}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def last_flange_action(self) -> dict[str, float]:
        if self._last_flange_action is None:
            raise RuntimeError("No Nero flange action has been sent yet.")
        return dict(self._last_flange_action)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.right.connect()
        self.left.connect()
        self._arms_connected = True
        self._connect_cameras()
        self._is_connected = True
        logger.info("%s connected.", self)

    def _connect_cameras(self, max_attempts: int = 3, retry_s: float = 1.0) -> None:
        for cam_name, cam in self.cameras.items():
            for attempt in range(1, max_attempts + 1):
                try:
                    cam.connect()
                    break
                except Exception:
                    try:
                        if cam.is_connected:
                            cam.disconnect()
                    except Exception:
                        logger.warning("Failed to clean up %s after connect failure.", cam_name, exc_info=True)
                    if attempt >= max_attempts:
                        raise
                    logger.warning(
                        "Failed to connect camera %s; retrying %d/%d after %.1fs.",
                        cam_name,
                        attempt + 1,
                        max_attempts,
                        retry_s,
                        exc_info=True,
                    )
                    time.sleep(retry_s)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        for arm in self.arms.values():
            if arm.robot is not None:
                arm.robot.set_speed_percent(arm.config.connection.speed_percent)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: dict[str, Any] = {}
        for arm_name in ("right", "left"):
            observation.update(self.arms[arm_name].observation())
        for cam_key, cam in self.cameras.items():
            observation[cam_key] = cam.async_read(timeout_ms=1000).copy()
        return observation

    @check_if_not_connected
    def get_flange_observation(self) -> RobotObservation:
        observation = self.get_flange_state_observation()
        for cam_key, cam in self.cameras.items():
            observation[cam_key] = cam.async_read(timeout_ms=1000).copy()
        return observation

    def get_flange_state_observation(self) -> RobotObservation:
        observation: dict[str, Any] = {}
        for arm_name in ("right", "left"):
            observation.update(self.arms[arm_name].flange_observation())
        return observation

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        *,
        send_gripper: bool = True,
        read_feedback: bool = True,
    ) -> RobotAction:
        sent: dict[str, float] = {}
        right_sent = self.right.send_action(action, send_gripper=send_gripper, read_feedback=read_feedback)
        left_sent = self.left.send_action(action, send_gripper=send_gripper, read_feedback=read_feedback)
        right_flange = right_sent.pop("_flange_action")
        left_flange = left_sent.pop("_flange_action")
        sent.update(right_sent)
        sent.update(left_sent)
        self._last_flange_action = {**right_flange, **left_flange}
        return sent

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        for arm in (self.left, self.right):
            arm.disconnect()
        self._arms_connected = False
        self._is_connected = False
        logger.info("%s disconnected.", self)
