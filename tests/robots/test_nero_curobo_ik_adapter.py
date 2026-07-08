import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_robot_nero.curobo_ik_adapter import (
    NeroCuroboArmIK,
    NeroDualCuroboIKAdapter,
    nero_pose_to_curobo_pose,
    select_ik_solution,
)
from lerobot_robot_nero.ee_local_se3_adapter import NeroEETargets
from lerobot_robot_nero.mapping import namespaced_gripper_name, namespaced_joint_names


class FakeArmIK:
    def __init__(self, solution):
        self.solution = np.asarray(solution, dtype=float)
        self.calls = []

    def solve(self, pose, current_joints=None):
        self.calls.append(
            (
                np.asarray(pose, dtype=float),
                None if current_joints is None else np.asarray(current_joints, dtype=float),
            )
        )
        return self.solution


class FallbackArmIK:
    def __init__(self, *, success_on_seed: str, solution):
        self.success_on_seed = success_on_seed
        self.solution = np.asarray(solution, dtype=float)
        self.calls = []

    def solve(self, pose, current_joints=None):
        seed = np.asarray(current_joints, dtype=float)
        self.calls.append((np.asarray(pose, dtype=float), seed))
        seed_name = {
            0.0: "real_current_joints",
            1.0: "last_sent_action",
            2.0: "last_success_ik_solution",
        }[float(seed[0])]
        if seed_name != self.success_on_seed:
            raise RuntimeError(f"{seed_name} failed")
        return self.solution


def test_nero_pose_to_curobo_pose_converts_euler_to_wxyz_quaternion():
    pose = np.array([0.1, 0.2, 0.3, 0.4, -0.2, 0.7], dtype=float)

    position, quaternion = nero_pose_to_curobo_pose(pose, euler_order="xyz")

    expected_xyzw = Rotation.from_euler("xyz", pose[3:6]).as_quat()
    np.testing.assert_allclose(position, pose[:3])
    np.testing.assert_allclose(quaternion, [expected_xyzw[3], *expected_xyzw[:3]])


def test_select_ik_solution_returns_first_success_without_seed():
    solutions = np.asarray([[10.0] * 7, [0.1] * 7], dtype=float)
    success = np.asarray([True, True], dtype=bool)

    selected = select_ik_solution(solutions, success)

    np.testing.assert_allclose(selected, solutions[0])


def test_select_ik_solution_uses_closest_success_to_seed():
    solutions = np.asarray([[10.0] * 7, [0.1] * 7, [-0.2] * 7], dtype=float)
    success = np.asarray([True, True, False], dtype=bool)

    selected = select_ik_solution(solutions, success, seed=np.zeros(7))

    np.testing.assert_allclose(selected, solutions[1])


def test_select_ik_solution_raises_when_no_solution_succeeds():
    solutions = np.asarray([[0.0] * 7], dtype=float)
    success = np.asarray([False], dtype=bool)

    with pytest.raises(RuntimeError, match="cuRobo IK failed"):
        select_ik_solution(solutions, success)


def test_arm_ik_passes_thresholds_to_curobo_success_tolerances(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeInverseKinematicsCfg:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return object()

    class FakeInverseKinematics:
        def __init__(self, config):
            self.tool_frames = ["tool0"]
            self.action_dim = 9
            self.kinematics = types.SimpleNamespace(joint_names=[f"joint_{idx}" for idx in range(9)])

    fake_inverse_kinematics = types.ModuleType("curobo.inverse_kinematics")
    fake_inverse_kinematics.InverseKinematics = FakeInverseKinematics
    fake_inverse_kinematics.InverseKinematicsCfg = FakeInverseKinematicsCfg

    fake_types = types.ModuleType("curobo.types")
    fake_types.GoalToolPose = object
    fake_types.JointState = object
    fake_types.Pose = object

    monkeypatch.setitem(sys.modules, "curobo.inverse_kinematics", fake_inverse_kinematics)
    monkeypatch.setitem(sys.modules, "curobo.types", fake_types)

    NeroCuroboArmIK(
        robot_file="nero_custom.yml",
        num_seeds=17,
        position_threshold=0.012,
        rotation_threshold=0.034,
        device="cpu",
    )

    assert captured["robot"] == "nero_custom.yml"
    assert captured["num_seeds"] == 17
    assert captured["position_tolerance"] == pytest.approx(0.012)
    assert captured["orientation_tolerance"] == pytest.approx(0.034)


def test_dual_curobo_adapter_maps_ee_targets_to_joint_action_dict():
    right_solution = np.arange(1, 8, dtype=float)
    left_solution = np.arange(11, 18, dtype=float)
    right_ik = FakeArmIK(right_solution)
    left_ik = FakeArmIK(left_solution)
    adapter = NeroDualCuroboIKAdapter(right_ik=right_ik, left_ik=left_ik)
    targets = NeroEETargets(
        right_pose=np.arange(6, dtype=float),
        right_gripper_width=0.031,
        left_pose=np.arange(10, 16, dtype=float),
        left_gripper_width=0.042,
    )
    right_current = np.full(7, 0.5, dtype=float)
    left_current = np.full(7, -0.5, dtype=float)

    action = adapter.ee_targets_to_joint_action(
        targets,
        right_current_joints=right_current,
        left_current_joints=left_current,
    )

    np.testing.assert_allclose(right_ik.calls[0][0], targets.right_pose)
    np.testing.assert_allclose(right_ik.calls[0][1], right_current)
    np.testing.assert_allclose(left_ik.calls[0][0], targets.left_pose)
    np.testing.assert_allclose(left_ik.calls[0][1], left_current)
    for idx, name in enumerate(namespaced_joint_names("right")):
        assert action[name] == pytest.approx(right_solution[idx])
    for idx, name in enumerate(namespaced_joint_names("left")):
        assert action[name] == pytest.approx(left_solution[idx])
    assert action[namespaced_gripper_name("right")] == pytest.approx(0.031)
    assert action[namespaced_gripper_name("left")] == pytest.approx(0.042)


def test_dual_curobo_adapter_retries_seed_candidates_and_reports_sources():
    right_ik = FallbackArmIK(success_on_seed="last_sent_action", solution=np.arange(1, 8, dtype=float))
    left_ik = FallbackArmIK(
        success_on_seed="last_success_ik_solution",
        solution=np.arange(11, 18, dtype=float),
    )
    adapter = NeroDualCuroboIKAdapter(right_ik=right_ik, left_ik=left_ik)
    targets = NeroEETargets(
        right_pose=np.arange(6, dtype=float),
        right_gripper_width=0.031,
        left_pose=np.arange(10, 16, dtype=float),
        left_gripper_width=0.042,
    )

    action, metadata = adapter.ee_targets_to_joint_action_with_metadata(
        targets,
        right_current_joints=np.zeros(7),
        left_current_joints=np.zeros(7),
        right_seed_candidates=[
            ("last_sent_action", np.ones(7)),
            ("last_success_ik_solution", np.full(7, 2.0)),
        ],
        left_seed_candidates=[
            ("last_sent_action", np.ones(7)),
            ("last_success_ik_solution", np.full(7, 2.0)),
        ],
    )

    assert metadata["right"]["seed_source"] == "last_sent_action"
    assert metadata["left"]["seed_source"] == "last_success_ik_solution"
    assert [call[1][0] for call in right_ik.calls] == [0.0, 1.0]
    assert [call[1][0] for call in left_ik.calls] == [0.0, 1.0, 2.0]
    assert action["right_nero_joint_1"] == pytest.approx(1.0)
    assert action["left_nero_joint_1"] == pytest.approx(11.0)


def test_dual_curobo_adapter_reports_all_seed_failures():
    right_ik = FallbackArmIK(success_on_seed="last_success_ik_solution", solution=np.arange(1, 8))
    left_ik = FallbackArmIK(success_on_seed="last_success_ik_solution", solution=np.arange(11, 18))
    adapter = NeroDualCuroboIKAdapter(right_ik=right_ik, left_ik=left_ik)
    targets = NeroEETargets(
        right_pose=np.arange(6, dtype=float),
        right_gripper_width=0.031,
        left_pose=np.arange(10, 16, dtype=float),
        left_gripper_width=0.042,
    )

    with pytest.raises(RuntimeError, match="right IK failed.*real_current_joints.*last_sent_action"):
        adapter.ee_targets_to_joint_action_with_metadata(
            targets,
            right_current_joints=np.zeros(7),
            left_current_joints=np.zeros(7),
            right_seed_candidates=[("last_sent_action", np.ones(7))],
            left_seed_candidates=[("last_sent_action", np.ones(7))],
        )
