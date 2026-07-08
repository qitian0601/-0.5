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

from .config_nero import NeroRobotConfig
from .mapping import (
    SO101ToNeroMapping,
    namespaced_action_names,
    namespaced_gripper_name,
    namespaced_joint_names,
)

logger = logging.getLogger(__name__)


class NeroRobot(Robot):
    config_class = NeroRobotConfig
    name = "nero"

    def __init__(self, config: NeroRobotConfig):
        super().__init__(config)
        self.config = config
        self.mapping = SO101ToNeroMapping.from_config(config.mapping)
        self.arm = config.mapping.arm
        self.cameras = make_cameras_from_configs(config.cameras)
        self.robot: Any | None = None
        self.end_effector: Any | None = None
        self._is_connected = False
        self._last_command: np.ndarray | None = None
        self._gripper_width = self.mapping.nero_gripper_width_min

    @property
    def _joint_features(self) -> dict[str, type]:
        return {name: float for name in namespaced_joint_names(self.arm)}

    @property
    def _gripper_features(self) -> dict[str, type]:
        return {namespaced_gripper_name(self.arm): float}

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

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.robot = self._make_sdk_robot()
        self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()
        self._take_can_control()
        while not self.robot.enable():
            logger.info("Waiting for Nero arm enable...")
            time.sleep(self.config.connection.enable_retry_s)
        self.robot.set_speed_percent(self.config.connection.speed_percent)

        for cam in self.cameras.values():
            cam.connect()

        self._last_command = self._read_joints()
        self._is_connected = True
        logger.info("%s connected.", self)

    def _take_can_control(self) -> None:
        if self.robot is None:
            raise RuntimeError("Nero SDK robot is not initialized.")

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

    def _make_sdk_robot(self):
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        except ImportError as exc:
            raise ImportError(
                "NeroRobot requires the Nero SDK package 'pyAgxArm'. Install/configure the Nero SDK "
                "before using --robot.type=nero."
            ) from exc

        firmware = getattr(NeroFW, self.config.connection.firmware_version)
        arm_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=firmware,
            interface=self.config.connection.interface,
            channel=self.config.connection.channel,
        )
        return AgxArmFactory.create_arm(arm_config)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if self.robot is not None:
            self.robot.set_speed_percent(self.config.connection.speed_percent)

    def _read_joints(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Nero SDK robot is not initialized.")
        joints = self.robot.get_joint_angles()
        while joints is None:
            time.sleep(0.01)
            joints = self.robot.get_joint_angles()
        joints = getattr(joints, "msg", joints)
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,):
            raise ValueError(f"Nero SDK returned joint shape {joints.shape}; expected 7 values.")
        return joints

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints = self._read_joints()
        observation: dict[str, Any] = {
            name: float(value) for name, value in zip(namespaced_joint_names(self.arm), joints, strict=True)
        }
        observation[namespaced_gripper_name(self.arm)] = float(self._gripper_width)

        for cam_key, cam in self.cameras.items():
            observation[cam_key] = cam.async_read(timeout_ms=1000).copy()

        return observation

    def _command_vector_from_action(self, action: RobotAction) -> tuple[np.ndarray, float]:
        missing = [name for name in namespaced_action_names(self.arm) if name not in action]
        if missing:
            raise KeyError(f"Missing Nero action keys: {missing}")
        joints = np.asarray([float(action[name]) for name in namespaced_joint_names(self.arm)], dtype=float)
        gripper_width = float(action[namespaced_gripper_name(self.arm)])
        joints = np.clip(joints, self.mapping.nero_limit_low, self.mapping.nero_limit_high)
        gripper_width = float(
            np.clip(gripper_width, self.mapping.nero_gripper_width_min, self.mapping.nero_gripper_width_max)
        )
        return joints, gripper_width

    def _smooth_command(self, target: np.ndarray) -> np.ndarray:
        if self._last_command is None:
            self._last_command = self._read_joints()

        alpha = float(np.clip(self.config.command.alpha, 0.0, 1.0))
        smoothed = self._last_command + alpha * (target - self._last_command)

        max_step = self.config.command.max_step_rad
        if max_step is not None and max_step > 0:
            delta = np.clip(smoothed - self._last_command, -max_step, max_step)
            smoothed = self._last_command + delta

        smoothed = np.clip(smoothed, self.mapping.nero_limit_low, self.mapping.nero_limit_high)
        return smoothed

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.robot is None or self.end_effector is None:
            raise RuntimeError("Nero SDK robot is not initialized.")

        target_joints, gripper_width = self._command_vector_from_action(action)
        command_joints = self._smooth_command(target_joints)
        if self.config.command.move_method == "move_js":
            self.robot.move_js(command_joints.tolist())
        elif self.config.command.move_method == "move_j":
            self.robot.move_j(command_joints.tolist())
        else:
            raise ValueError(f"Unsupported Nero move_method: {self.config.command.move_method!r}.")
        self.end_effector.move_gripper_m(
            value=gripper_width,
            force=self.config.connection.gripper_force,
        )

        self._last_command = command_joints
        self._gripper_width = gripper_width
        sent = {
            name: float(value)
            for name, value in zip(namespaced_joint_names(self.arm), command_joints, strict=True)
        }
        sent[namespaced_gripper_name(self.arm)] = gripper_width
        return sent

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        if self.robot is not None:
            disconnect = getattr(self.robot, "disconnect", None)
            if callable(disconnect):
                disconnect()
        self._is_connected = False
        logger.info("%s disconnected.", self)
