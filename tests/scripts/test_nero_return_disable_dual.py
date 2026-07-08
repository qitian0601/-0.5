import numpy as np
import pytest

from lerobot_robot_nero.mapping import namespaced_gripper_name, namespaced_joint_names
from lerobot_robot_nero.return_disable_dual import (
    DEFAULT_RETURN_POSE,
    _take_can_control_without_enabling,
    assert_dual_arm_enabled,
    return_to_pose_and_disable,
)


class FakeSdkArm:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.disabled = []
        self._msg_mode = type("Mode", (), {"ctrl_mode": 3, "move_mode": 1, "mit_mode": 0, "enable_can_push": 0})()
        self.calls = []

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

    def get_joints_enable_status_list(self):
        return [self.enabled] * 7

    def disable(self, joint_index=255):
        self.disabled.append(joint_index)
        self.enabled = False
        return True


class FakeEffector:
    def __init__(self):
        self.disable_gripper_calls = 0

    def disable_gripper(self):
        self.disable_gripper_calls += 1
        return True


class FakeArmRuntime:
    def __init__(self, arm: str, enabled=True):
        self.arm = arm
        self.robot = FakeSdkArm(enabled=enabled)
        self.end_effector = FakeEffector()
        self.joints = np.zeros(7)
        self.mapping = type(
            "Mapping",
            (),
            {
                "nero_limit_low": np.asarray([-1.0] * 7, dtype=float),
                "nero_limit_high": np.asarray([1.0] * 7, dtype=float),
                "nero_gripper_width_min": 0.0,
                "nero_gripper_width_max": 0.1,
            },
        )()
        self.config = type(
            "Config",
            (),
            {"command": type("Command", (), {"alpha": 0.8, "max_step_rad": 0.1})()},
        )()

    def read_joints(self):
        return self.joints


class FakeRobot:
    def __init__(self, enabled=True):
        self.right = FakeArmRuntime("right", enabled=enabled)
        self.left = FakeArmRuntime("left", enabled=enabled)
        self.sent = []

    def send_action(self, action):
        self.sent.append(action)
        self.right.joints = np.asarray([action[name] for name in namespaced_joint_names("right")], dtype=float)
        self.left.joints = np.asarray([action[name] for name in namespaced_joint_names("left")], dtype=float)
        return action


def test_assert_dual_arm_enabled_rejects_unenabled_robot():
    robot = FakeRobot(enabled=False)

    with pytest.raises(RuntimeError, match="not fully enabled"):
        assert_dual_arm_enabled(robot)


def test_take_can_control_without_enabling_uses_follower_mode(monkeypatch):
    monkeypatch.setattr("lerobot_robot_nero.return_disable_dual.time.sleep", lambda seconds: None)
    arm = FakeArmRuntime("right", enabled=True)

    _take_can_control_without_enabling(arm)

    assert arm.robot.calls == [
        ("_set_mode", 1, 1, 0, 1),
        ("set_follower_mode",),
        ("set_motion_mode", "js"),
    ]


def test_return_to_pose_and_disable_moves_to_pose_before_disabling(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.return_disable_dual.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot(enabled=True)
    pose = dict(DEFAULT_RETURN_POSE)
    for idx, name in enumerate(namespaced_joint_names("right"), start=1):
        pose[name] = float(idx) * 0.1
    for idx, name in enumerate(namespaced_joint_names("left"), start=1):
        pose[name] = float(-idx) * 0.1
    pose[namespaced_gripper_name("right")] = 0.03
    pose[namespaced_gripper_name("left")] = 0.04

    return_to_pose_and_disable(
        robot,
        pose,
        takeover_time_s=0.04,
        takeover_dt_s=0.02,
        tolerance_rad=0.001,
    )

    assert robot.sent[-1] == pose
    assert robot.right.end_effector.disable_gripper_calls == 1
    assert robot.left.end_effector.disable_gripper_calls == 1
    assert robot.right.robot.disabled == [255]
    assert robot.left.robot.disabled == [255]
    assert robot.right.robot.get_joints_enable_status_list() == [False] * 7
    assert robot.left.robot.get_joints_enable_status_list() == [False] * 7
    assert sleeps == [0.02, 0.02]


def test_return_to_pose_dry_run_does_not_move_or_disable(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.return_disable_dual.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot(enabled=True)
    pose = dict(DEFAULT_RETURN_POSE)

    return_to_pose_and_disable(
        robot,
        pose,
        takeover_time_s=0.04,
        takeover_dt_s=0.02,
        tolerance_rad=0.001,
        dry_run=True,
    )

    assert robot.sent == []
    assert robot.right.end_effector.disable_gripper_calls == 0
    assert robot.left.end_effector.disable_gripper_calls == 0
    assert robot.right.robot.disabled == []
    assert robot.left.robot.disabled == []
    assert robot.right.robot.get_joints_enable_status_list() == [True] * 7
    assert robot.left.robot.get_joints_enable_status_list() == [True] * 7
    assert sleeps == [0.02, 0.02]


def test_return_to_unreachable_pose_keeps_enabled_when_user_declines_disable(monkeypatch):
    monkeypatch.setattr("lerobot_robot_nero.return_disable_dual.precise_sleep", lambda value: None)
    robot = FakeRobot(enabled=True)
    pose = dict(DEFAULT_RETURN_POSE)
    for name in namespaced_joint_names("right"):
        pose[name] = 2.0
    for name in namespaced_joint_names("left"):
        pose[name] = -2.0
    pose[namespaced_gripper_name("right")] = 0.2
    pose[namespaced_gripper_name("left")] = -0.1

    return_to_pose_and_disable(
        robot,
        pose,
        takeover_time_s=0.02,
        takeover_dt_s=0.02,
        tolerance_rad=0.001,
        confirm_disable_fn=lambda max_error, clipped: False,
    )

    sent = robot.sent[-1]
    for name in namespaced_joint_names("right"):
        assert sent[name] == 1.0
    for name in namespaced_joint_names("left"):
        assert sent[name] == -1.0
    assert sent[namespaced_gripper_name("right")] == 0.1
    assert sent[namespaced_gripper_name("left")] == 0.0
    assert robot.right.end_effector.disable_gripper_calls == 0
    assert robot.left.end_effector.disable_gripper_calls == 0
    assert robot.right.robot.disabled == []
    assert robot.left.robot.disabled == []
    assert robot.right.robot.get_joints_enable_status_list() == [True] * 7
    assert robot.left.robot.get_joints_enable_status_list() == [True] * 7


def test_return_to_unreachable_pose_disables_when_user_confirms(monkeypatch):
    monkeypatch.setattr("lerobot_robot_nero.return_disable_dual.precise_sleep", lambda value: None)
    robot = FakeRobot(enabled=True)
    pose = dict(DEFAULT_RETURN_POSE)
    pose[namespaced_joint_names("right")[0]] = 2.0
    pose[namespaced_joint_names("left")[0]] = -2.0

    return_to_pose_and_disable(
        robot,
        pose,
        takeover_time_s=0.02,
        takeover_dt_s=0.02,
        tolerance_rad=0.001,
        confirm_disable_fn=lambda max_error, clipped: True,
    )

    sent = robot.sent[-1]
    assert sent[namespaced_joint_names("right")[0]] == 1.0
    assert sent[namespaced_joint_names("left")[0]] == -1.0
    assert robot.right.end_effector.disable_gripper_calls == 1
    assert robot.left.end_effector.disable_gripper_calls == 1
    assert robot.right.robot.disabled == [255]
    assert robot.left.robot.disabled == [255]
