from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_nero_dual_ee_pose.py"


def load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_nero_dual_ee_pose", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRobot:
    def __init__(self, joints=None):
        self.move_p_calls: list[list[float]] = []
        self.move_js_calls: list[list[float]] = []
        self.motion_modes: list[str] = []
        self.joints = np.zeros(7, dtype=float) if joints is None else np.asarray(joints, dtype=float)

    def move_p(self, pose):
        self.move_p_calls.append(list(pose))

    def move_js(self, joints):
        self.move_js_calls.append(list(joints))
        self.joints = np.asarray(joints, dtype=float)

    def set_motion_mode(self, mode):
        self.motion_modes.append(mode)

    def get_joint_angles(self):
        return self.joints.copy()


class LaggingRobot(FakeRobot):
    def __init__(self, joints=None, tracking_alpha=1.0):
        super().__init__(joints=joints)
        self.tracking_alpha = tracking_alpha
        self._last_target = self.joints.copy()

    def move_js(self, joints):
        self.move_js_calls.append(list(joints))
        self._last_target = np.asarray(joints, dtype=float)

    def get_joint_angles(self):
        self.joints = self.joints + self.tracking_alpha * (self._last_target - self.joints)
        return self.joints.copy()


class FakeGripper:
    def __init__(self):
        self.calls: list[tuple[float, float]] = []

    def move_gripper_m(self, *, value, force):
        self.calls.append((float(value), float(force)))


class FakeArmIK:
    def __init__(self, offset: float):
        self.offset = offset
        self.calls: list[tuple[list[float], np.ndarray | None]] = []

    def solve(self, pose, seed=None):
        self.calls.append((list(pose), None if seed is None else np.asarray(seed, dtype=float)))
        return np.arange(7, dtype=float) + self.offset


class SequenceArmIK:
    def __init__(self, solutions):
        self.solutions = [np.asarray(solution, dtype=float) for solution in solutions]
        self.calls: list[tuple[list[float], np.ndarray | None]] = []

    def solve(self, pose, seed=None):
        self.calls.append((list(pose), None if seed is None else np.asarray(seed, dtype=float)))
        return self.solutions.pop(0)


def test_nero_euler_pose_to_curobo_quaternion_uses_wxyz_order():
    replay = load_replay_module()

    position, quaternion = replay.nero_pose_to_curobo_pose([0.1, 0.2, 0.3, 0.4, -0.5, 0.6], "xyz")

    expected_xyzw = Rotation.from_euler("xyz", [0.4, -0.5, 0.6]).as_quat()
    expected_wxyz = np.array([expected_xyzw[3], expected_xyzw[0], expected_xyzw[1], expected_xyzw[2]])
    np.testing.assert_allclose(position, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(quaternion, expected_wxyz)


def test_move_dual_ee_with_curobo_ik_sends_joint_commands_and_grippers():
    replay = load_replay_module()
    right = FakeRobot()
    left = FakeRobot()
    right_gripper = FakeGripper()
    left_gripper = FakeGripper()
    ik_backend = replay.DualCuroboIKBackend(
        right=FakeArmIK(10.0),
        left=FakeArmIK(20.0),
    )

    replay.move_dual_ee(
        right,
        left,
        right_gripper,
        left_gripper,
        right_pose=[0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
        left_pose=[0.4, 0.5, 0.6, 0.3, 0.2, 0.1],
        right_width=0.01,
        left_width=0.02,
        gripper_force=1.5,
        ik_backend=ik_backend,
    )

    assert right.move_p_calls == []
    assert left.move_p_calls == []
    np.testing.assert_allclose(right.move_js_calls[0], np.arange(7, dtype=float) + 10.0)
    np.testing.assert_allclose(left.move_js_calls[0], np.arange(7, dtype=float) + 20.0)
    assert right_gripper.calls == [(0.01, 1.5)]
    assert left_gripper.calls == [(0.02, 1.5)]


def test_move_dual_ee_defaults_to_sdk_move_p_without_ik_backend():
    replay = load_replay_module()
    right = FakeRobot()
    left = FakeRobot()

    replay.move_dual_ee(
        right,
        left,
        FakeGripper(),
        FakeGripper(),
        right_pose=[1, 2, 3, 4, 5, 6],
        left_pose=[7, 8, 9, 10, 11, 12],
        right_width=0.01,
        left_width=0.02,
        gripper_force=1.0,
    )

    assert right.move_p_calls == [[1, 2, 3, 4, 5, 6]]
    assert left.move_p_calls == [[7, 8, 9, 10, 11, 12]]
    assert right.move_js_calls == []
    assert left.move_js_calls == []


def test_move_to_ready_uses_sdk_move_p_even_when_replay_uses_curobo():
    replay = load_replay_module()
    right = FakeRobot()
    left = FakeRobot()
    ik_backend = replay.DualCuroboIKBackend(
        right=FakeArmIK(10.0),
        left=FakeArmIK(20.0),
    )
    args = type(
        "Args",
        (),
        {
            "ik_backend": "curobo",
            "gripper_force": 1.0,
            "ready_wait_s": 0.0,
        },
    )()

    replay.move_to_ready(args, right, left, FakeGripper(), FakeGripper())

    assert ik_backend.right.calls == []
    assert ik_backend.left.calls == []
    assert right.motion_modes == ["p", "js"]
    assert left.motion_modes == ["p", "js"]
    assert right.move_p_calls == [replay.READY_RIGHT_POSE]
    assert left.move_p_calls == [replay.READY_LEFT_POSE]
    assert right.move_js_calls == []
    assert left.move_js_calls == []


def test_limit_joint_step_clips_each_joint_delta():
    replay = load_replay_module()

    limited = replay.limit_joint_step(
        current=np.array([0.0, 1.0, -1.0]),
        target=np.array([0.2, 0.5, -1.5]),
        max_step=0.03,
    )

    np.testing.assert_allclose(limited, np.array([0.03, 0.97, -1.03]))


def test_split_dual_ee_action_supports_legacy_bus_table_order():
    replay = load_replay_module()
    action = np.array(
        [1, 2, 3, 4, 5, 6, 0.01, 7, 8, 9, 10, 11, 12, 0.02],
        dtype=float,
    )

    right_pose, right_width, left_pose, left_width = replay.split_dual_ee_action(action)

    assert right_pose == [1, 2, 3, 4, 5, 6]
    assert right_width == 0.01
    assert left_pose == [7, 8, 9, 10, 11, 12]
    assert left_width == 0.02


def test_split_dual_ee_action_supports_pickplace_flange_metadata_order():
    replay = load_replay_module()
    action = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0.01, 0.02],
        dtype=float,
    )
    action_names = [
        "right_flange_x",
        "right_flange_y",
        "right_flange_z",
        "right_flange_roll",
        "right_flange_pitch",
        "right_flange_yaw",
        "left_flange_x",
        "left_flange_y",
        "left_flange_z",
        "left_flange_roll",
        "left_flange_pitch",
        "left_flange_yaw",
        "right_gripper_width",
        "left_gripper_width",
    ]

    right_pose, right_width, left_pose, left_width = replay.split_dual_ee_action(action, action_names)

    assert right_pose == [1, 2, 3, 4, 5, 6]
    assert right_width == 0.01
    assert left_pose == [7, 8, 9, 10, 11, 12]
    assert left_width == 0.02


def test_load_actions_selects_requested_episode(tmp_path):
    replay = load_replay_module()
    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    rows = []
    for episode in range(2):
        for frame in range(2):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "action": np.full(14, episode * 10 + frame, dtype=np.float32),
                }
            )
    import pandas as pd

    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet")
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(json.dumps({"fps": 15}), encoding="utf-8")

    actions, fps, action_names = replay.load_actions(tmp_path, episode=1)

    assert fps == 15
    assert action_names is None
    assert actions.shape == (2, 14)
    np.testing.assert_allclose(actions[0], np.full(14, 10, dtype=float))
    np.testing.assert_allclose(actions[1], np.full(14, 11, dtype=float))


def test_move_dual_ee_with_curobo_limits_joint_step_from_feedback():
    replay = load_replay_module()
    right = FakeRobot(joints=np.zeros(7))
    left = FakeRobot(joints=np.ones(7))
    ik_backend = replay.DualCuroboIKBackend(
        right=FakeArmIK(1.0),
        left=FakeArmIK(2.0),
    )

    replay.move_dual_ee(
        right,
        left,
        FakeGripper(),
        FakeGripper(),
        right_pose=[0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
        left_pose=[0.4, 0.5, 0.6, 0.3, 0.2, 0.1],
        right_width=0.01,
        left_width=0.02,
        gripper_force=1.5,
        ik_backend=ik_backend,
        max_joint_step_rad=0.03,
    )

    np.testing.assert_allclose(right.move_js_calls[0], np.full(7, 0.03))
    np.testing.assert_allclose(left.move_js_calls[0], np.full(7, 1.03))


def test_move_dual_ee_with_curobo_can_interpolate_to_target_before_replay():
    replay = load_replay_module()
    right = FakeRobot(joints=np.zeros(7))
    left = FakeRobot(joints=np.zeros(7))
    ik_backend = replay.DualCuroboIKBackend(
        right=SequenceArmIK([np.full(7, 0.05)]),
        left=SequenceArmIK([np.full(7, -0.05)]),
    )

    replay.move_dual_ee(
        right,
        left,
        FakeGripper(),
        FakeGripper(),
        right_pose=[0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
        left_pose=[0.4, 0.5, 0.6, 0.3, 0.2, 0.1],
        right_width=0.01,
        left_width=0.02,
        gripper_force=1.5,
        ik_backend=ik_backend,
        max_joint_step_rad=0.02,
        interpolate_to_target=True,
        step_sleep_s=0.0,
    )

    assert len(right.move_js_calls) == 3
    assert len(left.move_js_calls) == 3
    np.testing.assert_allclose(right.move_js_calls[-1], np.full(7, 0.05))
    np.testing.assert_allclose(left.move_js_calls[-1], np.full(7, -0.05))


def test_send_limited_joint_command_waits_for_feedback_to_reach_target():
    replay = load_replay_module()
    robot = LaggingRobot(joints=np.zeros(7), tracking_alpha=1.0)

    replay.send_limited_joint_command(
        robot,
        np.full(7, 0.05),
        arm_name="right",
        max_joint_step_rad=0.02,
        interpolate_to_target=True,
        step_sleep_s=0.0,
        wait_for_feedback=True,
        joint_target_tolerance_rad=0.001,
        joint_wait_timeout_s=1.0,
        monotonic_time=lambda: 0.0,
    )

    assert len(robot.move_js_calls) == 3
    np.testing.assert_allclose(robot.joints, np.full(7, 0.05))


def test_send_limited_joint_command_raises_when_feedback_does_not_reach_target():
    replay = load_replay_module()
    robot = LaggingRobot(joints=np.zeros(7), tracking_alpha=0.0)
    times = iter([0.0, 0.0, 0.2, 0.4])

    try:
        replay.send_limited_joint_command(
            robot,
            np.full(7, 0.05),
            arm_name="right",
            max_joint_step_rad=0.02,
            interpolate_to_target=True,
            step_sleep_s=0.0,
            wait_for_feedback=True,
            joint_target_tolerance_rad=0.001,
            joint_wait_timeout_s=0.1,
            joint_timeout_error_rad=0.01,
            monotonic_time=lambda: next(times),
        )
    except TimeoutError as exc:
        assert "Timed out waiting for right Nero joints" in str(exc)
    else:
        raise AssertionError("Expected TimeoutError")


def test_send_limited_joint_command_continues_on_soft_timeout():
    replay = load_replay_module()
    robot = LaggingRobot(joints=np.zeros(7), tracking_alpha=0.0)
    times = iter([0.0, 0.0, 0.2, 0.4, 0.6, 0.8])

    replay.send_limited_joint_command(
        robot,
        np.full(7, 0.02),
        arm_name="right",
        max_joint_step_rad=0.02,
        interpolate_to_target=True,
        step_sleep_s=0.0,
        wait_for_feedback=True,
        joint_target_tolerance_rad=0.001,
        joint_wait_timeout_s=0.1,
        joint_timeout_error_rad=0.03,
        monotonic_time=lambda: next(times),
    )

    assert robot.move_js_calls == [[0.02] * 7]


def test_replay_dataset_writes_per_frame_profile_csv(tmp_path, monkeypatch):
    replay = load_replay_module()
    actions = np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.01, 0.4, 0.5, 0.6, 0.3, 0.2, 0.1, 0.02],
            [0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.03, 0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.04],
        ],
        dtype=float,
    )
    monkeypatch.setattr(replay, "load_actions", lambda dataset_root, *, episode=0: (actions, 20, None))
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    args = type(
        "Args",
        (),
        {
            "fps": 20,
            "ik_backend": "curobo",
            "gripper_force": 1.0,
            "max_joint_step_rad": 0.0,
            "interpolate_first_target": False,
            "interpolate_each_frame": False,
            "control_dt_s": 0.0,
            "joint_target_tolerance_rad": 0.03,
            "joint_wait_timeout_s": 0.1,
            "joint_timeout_error_rad": 0.08,
            "takeover_time_s": 0.0,
        },
    )()
    right = FakeRobot()
    left = FakeRobot()
    ik_backend = replay.DualCuroboIKBackend(
        right=SequenceArmIK([np.full(7, 0.1), np.full(7, 0.2), np.full(7, 0.3)]),
        left=SequenceArmIK([np.full(7, -0.1), np.full(7, -0.2), np.full(7, -0.3)]),
    )
    profile_path = tmp_path / "profile.csv"

    with replay.ReplayProfileCsv(profile_path) as profile_writer:
        replay.replay_dataset(
            args,
            tmp_path / "dataset",
            0,
            right,
            left,
            FakeGripper(),
            FakeGripper(),
            ik_backend=ik_backend,
            profile_writer=profile_writer,
        )

    rows = profile_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("dataset,phase,frame_index,frame_count,fps")
    assert len(rows) == 4
    assert ",episode_0_first_target,-1,2,20," in rows[1]
    assert ",episode_0_replay,0,2,20," in rows[2]
    assert ",episode_0_replay,1,2,20," in rows[3]
