import numpy as np
from types import SimpleNamespace

from lerobot_robot_nero.config_nero import (
    NeroCommandConfig,
    NeroConnectionConfig,
    NeroMappingConfig,
    NeroRobotConfig,
)
from lerobot_robot_nero.robot_nero import NeroRobot


class FakeEffector:
    def __init__(self):
        self.commands = []

    def move_gripper_m(self, *, value, force):
        self.commands.append((value, force))


class FakeSdkRobot:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper"))

    def __init__(self):
        self._msg_mode = SimpleNamespace(ctrl_mode=3, move_mode=1, mit_mode=0, enable_can_push=0)
        self.calls = []
        self.joints = np.zeros(7)
        self.move_js_commands = []

    def init_effector(self, effector):
        self.calls.append(("init_effector", effector))
        return FakeEffector()

    def connect(self):
        self.calls.append(("connect",))

    def _set_mode(self):
        self.calls.append(
            (
                "_set_mode",
                self._msg_mode.ctrl_mode,
                self._msg_mode.move_mode,
                self._msg_mode.mit_mode,
                self._msg_mode.enable_can_push,
            )
        )

    def set_follower_mode(self):
        self.calls.append(("set_follower_mode",))

    def set_motion_mode(self, motion_mode):
        self.calls.append(("set_motion_mode", motion_mode))

    def reset(self):
        self.calls.append(("reset",))

    def clear_joint_error(self, joint_index=255):
        self.calls.append(("clear_joint_error", joint_index))

    def enable(self):
        self.calls.append(("enable",))
        return True

    def set_speed_percent(self, speed_percent):
        self.calls.append(("set_speed_percent", speed_percent))

    def get_joint_angles(self):
        return self.joints

    def move_js(self, joints):
        self.move_js_commands.append(joints)


class FakeCamera:
    height = 2
    width = 3
    is_connected = True

    def __init__(self):
        self.async_read_calls = []

    def async_read(self, *, timeout_ms):
        self.async_read_calls.append(timeout_ms)
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def disconnect(self):
        pass


def test_nero_robot_features_are_namespaced():
    robot = NeroRobot(NeroRobotConfig(mapping=NeroMappingConfig(arm="right")))

    assert list(robot.action_features) == [
        "right_nero_joint_1",
        "right_nero_joint_2",
        "right_nero_joint_3",
        "right_nero_joint_4",
        "right_nero_joint_5",
        "right_nero_joint_6",
        "right_nero_joint_7",
        "right_gripper_width",
    ]
    assert list(robot.observation_features) == list(robot.action_features)


def test_nero_connection_default_uses_v120_firmware():
    assert NeroConnectionConfig().firmware_version == "V120"


def test_single_arm_connect_takes_follower_can_control_before_enabling(monkeypatch):
    robot = NeroRobot(
        NeroRobotConfig(
            connection=NeroConnectionConfig(speed_percent=42, enable_retry_s=0.0),
        )
    )
    fake_robot = FakeSdkRobot()
    monkeypatch.setattr(robot, "_make_sdk_robot", lambda: fake_robot)
    monkeypatch.setattr("lerobot_robot_nero.robot_nero.time.sleep", lambda seconds: None)

    robot.connect()

    assert fake_robot.calls[:7] == [
        ("init_effector", "agx_gripper"),
        ("connect",),
        ("_set_mode", 1, 1, 0, 1),
        ("set_follower_mode",),
        ("set_motion_mode", "js"),
        ("reset",),
        ("clear_joint_error", 255),
    ]
    assert fake_robot.calls[-2:] == [("enable",), ("set_speed_percent", 42)]


def test_send_action_clips_smooths_and_returns_actual_command():
    robot = NeroRobot(
        NeroRobotConfig(
            command=NeroCommandConfig(alpha=1.0, max_step_rad=0.05),
            mapping=NeroMappingConfig(
                nero_limit_low=[-0.1] * 7,
                nero_limit_high=[0.1] * 7,
                nero_gripper_width_min=0.0,
                nero_gripper_width_max=0.1,
            ),
        )
    )
    fake_robot = FakeSdkRobot()
    fake_effector = FakeEffector()
    robot.robot = fake_robot
    robot.end_effector = fake_effector
    robot._last_command = np.zeros(7)
    robot._is_connected = True

    action = {f"right_nero_joint_{idx}": 1.0 for idx in range(1, 8)}
    action["right_gripper_width"] = 1.0

    sent = robot.send_action(action)

    assert np.allclose(fake_robot.move_js_commands[-1], [0.05] * 7)
    assert sent == {**{f"right_nero_joint_{idx}": 0.05 for idx in range(1, 8)}, "right_gripper_width": 0.1}
    assert fake_effector.commands[-1] == (0.1, 1.0)


def test_read_joints_accepts_sdk_message_objects():
    robot = NeroRobot(NeroRobotConfig())
    fake_robot = FakeSdkRobot()
    fake_robot.joints = SimpleNamespace(msg=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    robot.robot = fake_robot

    assert np.allclose(robot._read_joints(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


def test_get_observation_reads_cameras_with_timeout():
    robot = NeroRobot(NeroRobotConfig())
    fake_robot = FakeSdkRobot()
    fake_camera = FakeCamera()
    robot.robot = fake_robot
    robot.cameras = {"front": fake_camera}
    robot._is_connected = True

    observation = robot.get_observation()

    assert observation["front"].shape == (2, 3, 3)
    assert fake_camera.async_read_calls == [1000]
