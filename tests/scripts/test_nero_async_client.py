import json
from pathlib import Path
from queue import Queue
import threading

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from lerobot_robot_nero.async_client import (
    DEFAULT_DUAL_ACTION_NAMES,
    EESafeNeroRobot,
    NeroAsyncClientConfig,
    NeroDebugImageSaveConfig,
    NeroDebugVideoSaveConfig,
    NeroAsyncSafetyConfig,
    ObservationImageSaver,
    ObservationVideoSaver,
    SafeNeroRobot,
    _is_ik_failure_exception,
    _should_drop_action_chunk_for_ik_recovery,
    _make_client_config,
    action_vector_from_pose,
    limit_ee_policy_action_step,
    limit_action_step,
    recover_nero_async_client_from_ik_failure,
    run_nero_async_client,
    sync_to_fixed_ready_pose,
)
from lerobot_robot_nero.ee_local_se3_adapter import (
    EE_LOCAL_SE3_ACTION_NAMES,
    EE_SO3_ACTION_NAMES,
    NeroEESO3Adapter,
    NeroEETargets,
)
from lerobot_robot_nero.trace import NeroInferenceTraceConfig, NeroInferenceTracer


class FakeArm:
    def __init__(self, arm: str):
        self.arm = arm
        self.joints = np.zeros(7)
        self.config = type(
            "Config",
            (),
            {"command": type("Command", (), {"alpha": 0.8, "max_step_rad": 0.1})()},
        )()

    def read_joints(self):
        return self.joints


class FakeRobot:
    def __init__(self):
        self.right = FakeArm("right")
        self.left = FakeArm("left")
        self.sent = []
        self.send_kwargs = []
        self.control_dt_s = 0.006

    def send_action(self, action, **kwargs):
        self.sent.append(action)
        self.send_kwargs.append(kwargs)
        self.right.joints = np.asarray([action[f"right_nero_joint_{idx}"] for idx in range(1, 8)])
        self.left.joints = np.asarray([action[f"left_nero_joint_{idx}"] for idx in range(1, 8)])
        return action


class FakeEERobot(FakeRobot):
    @property
    def flange_observation_features(self):
        features = {}
        for arm in ("right", "left"):
            for component in ("x", "y", "z", "roll", "pitch", "yaw"):
                features[f"{arm}_flange_{component}"] = float
            features[f"{arm}_gripper_width"] = float
        features["front"] = (2, 2, 3)
        return features

    def get_flange_observation(self):
        observation = {}
        for arm, offset in (("right", 0.0), ("left", 10.0)):
            for idx, component in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
                observation[f"{arm}_flange_{component}"] = offset + idx
            observation[f"{arm}_gripper_width"] = 0.03 + offset * 0.001
        observation["front"] = np.zeros((2, 2, 3), dtype=np.uint8)
        return observation

    def get_flange_state_observation(self):
        observation = {}
        for arm in ("right", "left"):
            for component in ("x", "y", "z", "roll", "pitch", "yaw"):
                observation[f"{arm}_flange_{component}"] = 0.0
            observation[f"{arm}_gripper_width"] = 0.03
        return observation


class FakeEEAdapter:
    def __init__(self):
        self.observations = []
        self.actions = []

    def flange_observation_to_policy_state(self, observation):
        self.observations.append(observation)
        return np.arange(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)

    def policy_action_to_nero_ee_targets(self, action):
        self.actions.append(np.asarray(action, dtype=float))
        return "ee-targets"

    def read_robot_policy_state(self, robot):
        return self.flange_observation_to_policy_state(robot.get_flange_state_observation())


class FakeEESO3Adapter(FakeEEAdapter):
    def flange_observation_to_policy_state(self, observation):
        self.observations.append(observation)
        return np.arange(len(EE_SO3_ACTION_NAMES), dtype=float)


class FakeTraceEEAdapter(FakeEEAdapter):
    def policy_action_to_nero_ee_targets(self, action):
        self.actions.append(np.asarray(action, dtype=float))
        return NeroEETargets(
            right_pose=np.arange(6, dtype=float),
            right_gripper_width=0.03,
            left_pose=np.arange(10, 16, dtype=float),
            left_gripper_width=0.04,
        )


class FakeIKAdapter:
    def __init__(self):
        self.calls = []

    def ee_targets_to_joint_action(self, targets, *, right_current_joints, left_current_joints):
        self.calls.append(
            (
                targets,
                np.asarray(right_current_joints, dtype=float),
                np.asarray(left_current_joints, dtype=float),
            )
        )
        action = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
        action["right_nero_joint_1"] = 0.4
        action["left_gripper_width"] = 0.05
        return action


class FailingIKAdapter:
    def ee_targets_to_joint_action(self, targets, *, right_current_joints, left_current_joints):
        raise RuntimeError("synthetic IK failure")


class MetadataIKAdapter:
    def __init__(self):
        self.calls = []

    def ee_targets_to_joint_action_with_metadata(
        self,
        targets,
        *,
        right_current_joints,
        left_current_joints,
        right_seed_candidates=None,
        left_seed_candidates=None,
    ):
        self.calls.append(
            {
                "targets": targets,
                "right_current_joints": np.asarray(right_current_joints, dtype=float),
                "left_current_joints": np.asarray(left_current_joints, dtype=float),
                "right_seed_candidates": right_seed_candidates,
                "left_seed_candidates": left_seed_candidates,
            }
        )
        action = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
        action["right_nero_joint_1"] = 0.41
        action["left_nero_joint_1"] = 0.51
        return action, {
            "right": {
                "seed_source": "last_sent_action",
                "seed_joints": np.ones(7),
                "attempts": [{"seed_source": "real_current_joints", "error": "failed"}],
            },
            "left": {
                "seed_source": "last_success_ik_solution",
                "seed_joints": np.full(7, 2.0),
                "attempts": [
                    {"seed_source": "real_current_joints", "error": "failed"},
                    {"seed_source": "last_sent_action", "error": "failed"},
                ],
            },
        }


def default_test_pose():
    pose = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    pose["right_gripper_width"] = 0.03
    pose["left_gripper_width"] = 0.04
    return pose


def test_action_vector_from_pose_uses_policy_action_order():
    pose = {name: float(idx) for idx, name in enumerate(DEFAULT_DUAL_ACTION_NAMES)}

    vector = action_vector_from_pose(pose)

    assert vector.shape == (16,)
    assert vector[0] == 0.0
    assert vector[7] == 7.0
    assert vector[14] == 14.0
    assert vector[15] == 15.0


def test_limit_action_step_clamps_joints_and_grippers_from_last_action():
    last = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target = {name: 1.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_gripper_width"] = 0.05
    target["left_gripper_width"] = -0.02

    limited = limit_action_step(
        target,
        last,
        max_joint_step_rad=0.2,
        max_gripper_step_m=0.01,
        gripper_min_m=0.0,
        gripper_max_m=0.1,
    )

    assert limited["right_nero_joint_1"] == 0.2
    assert limited["left_nero_joint_7"] == 0.2
    assert limited["right_gripper_width"] == 0.01
    assert limited["left_gripper_width"] == 0.0


def test_limit_ee_policy_action_step_clamps_position_and_rotation_only():
    current = np.zeros(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    target = np.zeros(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    target[0:3] = np.array([0.03, 0.04, 0.0])
    target[3:6] = np.array([0.0, 0.0, 0.20])
    target[6] = 0.08
    target[7:10] = np.array([-0.06, 0.08, 0.0])
    target[10:13] = np.array([0.0, 0.0, -0.30])
    target[13] = 0.02
    target[14:16] = np.array([0.5, -0.5])

    limited = limit_ee_policy_action_step(
        target,
        current,
        max_position_step_m=0.01,
        max_rotation_step_rad=0.05,
    )

    assert np.linalg.norm(limited[0:3]) == pytest.approx(0.01)
    assert np.linalg.norm(limited[3:6]) == pytest.approx(0.05)
    assert np.linalg.norm(limited[7:10]) == pytest.approx(0.01)
    assert np.linalg.norm(limited[10:13]) == pytest.approx(0.05)
    assert limited[6] == 0.08
    assert limited[13] == 0.02
    np.testing.assert_allclose(limited[14:16], np.array([0.5, -0.5]))


def test_limit_ee_policy_action_step_disabled_returns_target_unchanged():
    current = np.zeros(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    target = np.arange(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)

    limited = limit_ee_policy_action_step(
        target,
        current,
        max_position_step_m=0.0,
        max_rotation_step_rad=0.0,
    )

    np.testing.assert_allclose(limited, target)


def test_limit_ee_policy_action_step_uses_shortest_so3_rotation_near_pi_boundary():
    current = np.zeros(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    target = np.zeros(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    current[3:6] = np.array([0.0, 0.0, np.pi - 0.01])
    target[3:6] = np.array([0.0, 0.0, -np.pi + 0.01])

    limited = limit_ee_policy_action_step(
        target,
        current,
        max_position_step_m=0.0,
        max_rotation_step_rad=0.05,
    )

    delta_angle = (
        Rotation.from_rotvec(limited[3:6]) * Rotation.from_rotvec(current[3:6]).inv()
    ).magnitude()
    np.testing.assert_allclose(limited[3:6], target[3:6], atol=1e-8)
    assert delta_angle == pytest.approx(0.02)


def test_default_high_rate_executor_targets_180hz():
    assert NeroAsyncSafetyConfig().high_rate_dt_s == pytest.approx(1.0 / 180.0)


def test_ik_failure_recovery_is_disabled_by_default():
    assert NeroAsyncSafetyConfig().recover_on_ik_failure is False


def test_ik_failure_exception_detection_is_narrow():
    assert _is_ik_failure_exception(RuntimeError("cuRobo IK failed for pose=[...]"))
    assert _is_ik_failure_exception(RuntimeError("right IK failed: no valid seed"))
    assert _is_ik_failure_exception(RuntimeError("cuRobo IK error too large for pose=[...]"))
    assert not _is_ik_failure_exception(RuntimeError("camera disconnected"))
    assert not _is_ik_failure_exception(ValueError("cuRobo IK failed for pose=[...]"))


def test_ik_recovery_generation_drops_inflight_chunks_after_recovery_completes():
    class Client:
        def __init__(self):
            self._ik_recovery_active = threading.Event()
            self._ik_recovery_generation = 1
            self._ik_recovery_lock = threading.Lock()

    client = Client()

    assert _should_drop_action_chunk_for_ik_recovery(client, request_generation=1) is False

    client._ik_recovery_generation = 2
    assert _should_drop_action_chunk_for_ik_recovery(client, request_generation=1) is True

    client._ik_recovery_active.set()
    assert _should_drop_action_chunk_for_ik_recovery(client, request_generation=2) is True


def test_make_client_config_includes_record_dir():
    cfg = NeroAsyncClientConfig()

    client_cfg = _make_client_config(cfg, record_dir="/tmp/nero-record")

    assert client_cfg.record_dir == "/tmp/nero-record"


def test_sync_to_fixed_ready_pose_smooths_both_arms_and_restores_command_settings(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.async_client.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot()
    pose = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    for idx in range(1, 8):
        pose[f"right_nero_joint_{idx}"] = float(idx)
        pose[f"left_nero_joint_{idx}"] = float(-idx)
    pose["right_gripper_width"] = 0.03
    pose["left_gripper_width"] = 0.04

    max_error = sync_to_fixed_ready_pose(
        robot,
        pose,
        takeover_time_s=0.04,
        takeover_dt_s=0.02,
        tolerance_rad=0.001,
    )

    assert len(robot.sent) == 2
    assert robot.sent[-1]["right_nero_joint_1"] == 1.0
    assert robot.sent[-1]["left_nero_joint_7"] == -7.0
    assert robot.sent[-1]["right_gripper_width"] == 0.03
    assert sleeps == [0.02, 0.02]
    assert robot.right.config.command.alpha == 0.8
    assert robot.right.config.command.max_step_rad == 0.1
    assert max_error <= 0.001


def test_safe_nero_robot_queues_policy_target_for_high_rate_executor():
    robot = FakeRobot()
    safety = NeroAsyncSafetyConfig(
        high_rate_control=True,
        max_policy_step_rad=0.1,
        max_gripper_step_m=0.1,
        max_executor_step_rad=0.02,
        max_executor_gripper_step_m=0.01,
    )
    safe_robot = SafeNeroRobot(robot, safety)
    target = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_nero_joint_1"] = 1.0
    target["left_gripper_width"] = 0.05

    returned = safe_robot.send_action(target)

    assert robot.sent == []
    assert returned["right_nero_joint_1"] == 0.1
    assert returned["left_gripper_width"] == 0.05


def test_safe_nero_robot_directly_sends_policy_action_when_high_rate_disabled():
    robot = FakeRobot()
    safety = NeroAsyncSafetyConfig(
        high_rate_control=False,
        max_policy_step_rad=0.1,
        max_gripper_step_m=0.1,
    )
    safe_robot = SafeNeroRobot(robot, safety)
    target = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_nero_joint_1"] = 1.0
    target["left_gripper_width"] = 0.05

    sent = safe_robot.send_action(target)

    assert robot.sent == [sent]
    assert sent["right_nero_joint_1"] == 0.1
    assert sent["left_gripper_width"] == 0.05


def test_safe_nero_robot_high_rate_step_moves_toward_latest_target():
    robot = FakeRobot()
    safety = NeroAsyncSafetyConfig(
        high_rate_control=True,
        max_policy_step_rad=0.1,
        max_gripper_step_m=0.1,
        max_executor_step_rad=0.02,
        max_executor_gripper_step_m=0.01,
    )
    safe_robot = SafeNeroRobot(robot, safety)
    target = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_nero_joint_1"] = 1.0
    target["left_gripper_width"] = 0.05
    safe_robot.send_action(target)

    sent = safe_robot.execute_high_rate_step()

    assert sent["right_nero_joint_1"] == pytest.approx(0.1 / 6.0)
    assert sent["left_gripper_width"] == pytest.approx(0.05 / 6.0)
    assert robot.sent == [sent]


def test_safe_nero_robot_interpolates_policy_target_at_high_rate():
    robot = FakeRobot()
    safety = NeroAsyncSafetyConfig(
        high_rate_control=True,
        max_policy_step_rad=float("inf"),
        max_gripper_step_m=float("inf"),
        max_executor_step_rad=float("inf"),
        max_executor_gripper_step_m=float("inf"),
        high_rate_interpolation_steps=6,
    )
    safe_robot = SafeNeroRobot(robot, safety)
    target = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_nero_joint_1"] = 0.6
    target["left_gripper_width"] = 0.06

    safe_robot.send_action(target)
    sent = safe_robot.execute_high_rate_step()

    assert sent["right_nero_joint_1"] == pytest.approx(0.1)
    assert sent["left_gripper_width"] == pytest.approx(0.01)


def test_safe_nero_robot_high_rate_step_skips_redundant_gripper_and_feedback():
    robot = FakeRobot()
    safety = NeroAsyncSafetyConfig(high_rate_control=True)
    safe_robot = SafeNeroRobot(robot, safety)

    safe_robot.execute_high_rate_step(dt_s=1.0 / 180.0)
    safe_robot.execute_high_rate_step(dt_s=1.0 / 180.0)

    assert robot.send_kwargs[0] == {"send_gripper": True, "read_feedback": False}
    assert robot.send_kwargs[1] == {"send_gripper": False, "read_feedback": False}


def test_ee_safe_nero_robot_exposes_ee_policy_features_and_camera_observation():
    robot = FakeEERobot()
    ee_adapter = FakeEEAdapter()
    safe_robot = EESafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(),
        ee_adapter=ee_adapter,
        ik_adapter=FakeIKAdapter(),
    )

    observation = safe_robot.get_observation()

    assert list(safe_robot.action_features) == EE_LOCAL_SE3_ACTION_NAMES
    assert list(safe_robot.observation_features) == [*EE_LOCAL_SE3_ACTION_NAMES, "front"]
    assert [observation[name] for name in EE_LOCAL_SE3_ACTION_NAMES] == list(range(16))
    assert observation["front"].shape == (2, 2, 3)
    assert ee_adapter.observations[0]["right_flange_x"] == 0.0


def test_ee_safe_nero_robot_can_expose_ee_so3_14d_features():
    robot = FakeEERobot()
    ee_adapter = FakeEESO3Adapter()
    safe_robot = EESafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(),
        ee_adapter=ee_adapter,
        ik_adapter=FakeIKAdapter(),
        action_names=EE_SO3_ACTION_NAMES,
        action_mode_name="ee_so3",
    )

    observation = safe_robot.get_observation()

    assert list(safe_robot.action_features) == EE_SO3_ACTION_NAMES
    assert list(safe_robot.observation_features) == [*EE_SO3_ACTION_NAMES, "front"]
    assert [observation[name] for name in EE_SO3_ACTION_NAMES] == list(range(14))
    assert "base_or_head_x" not in observation
    assert observation["front"].shape == (2, 2, 3)


def test_ee_so3_adapter_uses_14d_policy_state_and_action_without_base_head():
    adapter = NeroEESO3Adapter()
    observation = {}
    for arm in ("right", "left"):
        for idx, component in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            observation[f"{arm}_flange_{component}"] = float(idx)
        observation[f"{arm}_gripper_width"] = 0.03

    state = adapter.flange_observation_to_policy_state(observation)
    targets = adapter.policy_action_to_nero_ee_targets(np.arange(14, dtype=float))

    assert state.shape == (14,)
    assert targets.base_or_head_xy.tolist() == [0.0, 0.0]
    assert targets.right_gripper_width == pytest.approx(6.0)
    assert targets.left_gripper_width == pytest.approx(13.0)


def test_ee_safe_nero_robot_solves_ik_then_reuses_joint_high_rate_executor():
    robot = FakeEERobot()
    ee_adapter = FakeEEAdapter()
    ik_adapter = FakeIKAdapter()
    safety = NeroAsyncSafetyConfig(
        high_rate_control=True,
        max_policy_step_rad=float("inf"),
        max_gripper_step_m=float("inf"),
        max_executor_step_rad=float("inf"),
        max_executor_gripper_step_m=float("inf"),
        high_rate_interpolation_steps=1,
    )
    safe_robot = EESafeNeroRobot(robot, safety, ee_adapter=ee_adapter, ik_adapter=ik_adapter)
    ee_action = {name: float(idx) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)}

    returned = safe_robot.send_action(ee_action)

    np.testing.assert_allclose(ee_adapter.actions[0], np.arange(16, dtype=float))
    assert ik_adapter.calls[0][0] == "ee-targets"
    np.testing.assert_allclose(ik_adapter.calls[0][1], np.zeros(7))
    np.testing.assert_allclose(ik_adapter.calls[0][2], np.zeros(7))
    assert robot.sent == []
    assert returned["right_nero_joint_1"] == pytest.approx(0.4)

    sent = safe_robot.execute_high_rate_step()

    assert robot.sent == [sent]
    assert sent["right_nero_joint_1"] == pytest.approx(0.4)
    assert sent["left_gripper_width"] == pytest.approx(0.05)
    assert robot.send_kwargs[-1] == {"send_gripper": True, "read_feedback": False}


def test_ee_safe_nero_robot_limits_ee_policy_action_before_ik():
    robot = FakeEERobot()
    ee_adapter = FakeEEAdapter()
    ik_adapter = FakeIKAdapter()
    safety = NeroAsyncSafetyConfig(
        max_ee_position_step_m=0.01,
        max_ee_rotation_step_rad=0.05,
    )
    safe_robot = EESafeNeroRobot(robot, safety, ee_adapter=ee_adapter, ik_adapter=ik_adapter)
    ee_action = {name: 0.0 for name in EE_LOCAL_SE3_ACTION_NAMES}
    ee_action["right_ee_x"] = 0.03
    ee_action["right_ee_y"] = 0.04
    ee_action["right_ee_rotvec_z"] = 0.20
    ee_action["left_ee_x"] = -0.06
    ee_action["left_ee_y"] = 0.08
    ee_action["left_ee_rotvec_z"] = -0.30
    ee_action["right_gripper_width"] = 0.08
    ee_action["left_gripper_width"] = 0.02

    safe_robot.send_action(ee_action)

    limited = ee_adapter.actions[0]
    current = np.arange(len(EE_LOCAL_SE3_ACTION_NAMES), dtype=float)
    assert np.linalg.norm(limited[0:3] - current[0:3]) == pytest.approx(0.01)
    right_delta_angle = (
        Rotation.from_rotvec(limited[3:6]) * Rotation.from_rotvec(current[3:6]).inv()
    ).magnitude()
    assert np.linalg.norm(limited[7:10] - current[7:10]) == pytest.approx(0.01)
    left_delta_angle = (
        Rotation.from_rotvec(limited[10:13]) * Rotation.from_rotvec(current[10:13]).inv()
    ).magnitude()
    assert right_delta_angle == pytest.approx(0.05)
    assert left_delta_angle == pytest.approx(0.05)
    assert limited[6] == 0.08
    assert limited[13] == 0.02


def test_ee_safe_nero_robot_records_ik_request_and_success_current_joints(tmp_path):
    robot = FakeEERobot()
    robot.right.joints = np.arange(7, dtype=float) + 0.1
    robot.left.joints = np.arange(7, dtype=float) + 10.1
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="ee-ik-success"),
        meta={"task": "trace"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    safe_robot = EESafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(),
        ee_adapter=FakeTraceEEAdapter(),
        ik_adapter=FakeIKAdapter(),
        tracer=tracer,
    )

    safe_robot.send_action({name: float(idx) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)})
    tracer.close()

    trace_lines = [
        json.loads(line)
        for line in (tmp_path / "ee-ik-success/trace.jsonl").read_text().splitlines()
    ]
    events = [line["event"] for line in trace_lines]
    assert "ee_ik_request" in events
    assert "ee_ik_joint_target" in events
    request = next(line["data"] for line in trace_lines if line["event"] == "ee_ik_request")
    success = next(line["data"] for line in trace_lines if line["event"] == "ee_ik_joint_target")
    np.testing.assert_allclose(request["right_current_joints"], np.arange(7, dtype=float) + 0.1)
    np.testing.assert_allclose(request["left_current_joints"], np.arange(7, dtype=float) + 10.1)
    np.testing.assert_allclose(success["right_current_joints"], request["right_current_joints"])
    np.testing.assert_allclose(success["left_current_joints"], request["left_current_joints"])


def test_ee_safe_nero_robot_records_ik_failure_with_current_joints(tmp_path):
    robot = FakeEERobot()
    robot.right.joints = np.arange(7, dtype=float) + 0.2
    robot.left.joints = np.arange(7, dtype=float) + 20.2
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="ee-ik-failure"),
        meta={"task": "trace"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    safe_robot = EESafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(),
        ee_adapter=FakeTraceEEAdapter(),
        ik_adapter=FailingIKAdapter(),
        tracer=tracer,
    )

    with pytest.raises(RuntimeError, match="synthetic IK failure"):
        safe_robot.send_action({name: float(idx) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)})
    tracer.close()

    trace_lines = [
        json.loads(line)
        for line in (tmp_path / "ee-ik-failure/trace.jsonl").read_text().splitlines()
    ]
    failure = next(line["data"] for line in trace_lines if line["event"] == "ee_ik_failed")
    np.testing.assert_allclose(failure["right_current_joints"], np.arange(7, dtype=float) + 0.2)
    np.testing.assert_allclose(failure["left_current_joints"], np.arange(7, dtype=float) + 20.2)
    assert failure["error_type"] == "RuntimeError"
    assert "synthetic IK failure" in failure["error"]


def test_recover_nero_async_client_from_ik_failure_resets_queue_ready_and_server(
    monkeypatch,
):
    pose = default_test_pose()
    events = []
    sync_calls = []

    class Stub:
        def __init__(self):
            self.ready_calls = 0

        def Ready(self, request):
            self.ready_calls += 1
            events.append("server_ready")
            return None

    class SafeWrapper:
        def __init__(self):
            self.robot = FakeRobot()

        def stop_high_rate_executor(self):
            events.append("stop_executor")

        def start_high_rate_executor(self):
            events.append("start_executor")

        def set_last_action(self, action):
            events.append(("set_last", dict(action)))

    class Client:
        def __init__(self):
            self.action_queue = Queue()
            self.action_queue.put("stale-action")
            self.action_queue_lock = threading.Lock()
            self.latest_action_lock = threading.Lock()
            self.latest_action = 12
            self.action_chunk_size = 50
            self.must_go = threading.Event()
            self.stub = Stub()
            self.robot = SafeWrapper()
            self._ik_recovery_active = threading.Event()
            self._ik_recovery_generation = 0
            self._ik_recovery_lock = threading.Lock()

    def fake_sync(robot, ready_pose, **kwargs):
        sync_calls.append((robot, dict(ready_pose), kwargs))
        events.append("sync_ready")
        return 0.0

    client = Client()
    monkeypatch.setattr("lerobot_robot_nero.async_client.sync_to_fixed_ready_pose", fake_sync)

    recover_nero_async_client_from_ik_failure(
        client,
        NeroAsyncSafetyConfig(
            fixed_ready_pose=pose,
            recover_on_ik_failure=True,
            ik_failure_recovery_wait_for_enter=True,
        ),
        wait_for_resume=lambda: events.append("wait_enter"),
        error=RuntimeError("cuRobo IK failed for pose=[...]"),
    )

    assert client.action_queue.empty()
    assert client.latest_action == -1
    assert client.action_chunk_size == -1
    assert client.must_go.is_set()
    assert client.stub.ready_calls == 1
    assert sync_calls == [
        (
            client.robot.robot,
            pose,
            {
                "takeover_time_s": 4.0,
                "takeover_dt_s": 0.02,
                "tolerance_rad": 0.05,
            },
        )
    ]
    assert events == [
        "stop_executor",
        "sync_ready",
        ("set_last", pose),
        "server_ready",
        "wait_enter",
        "start_executor",
    ]
    assert not client._ik_recovery_active.is_set()
    assert client._ik_recovery_generation == 2


def test_recover_nero_async_client_from_ik_failure_dry_run_skips_ready_sync(monkeypatch):
    pose = default_test_pose()

    class Stub:
        def Ready(self, request):
            return None

    class SafeWrapper:
        robot = FakeRobot()

        def __init__(self):
            self.last_action = None

        def stop_high_rate_executor(self):
            pass

        def start_high_rate_executor(self):
            pass

        def set_last_action(self, action):
            self.last_action = dict(action)

    class Client:
        def __init__(self):
            self.action_queue = Queue()
            self.action_queue_lock = threading.Lock()
            self.latest_action_lock = threading.Lock()
            self.latest_action = 3
            self.action_chunk_size = 20
            self.must_go = threading.Event()
            self.stub = Stub()
            self.robot = SafeWrapper()
            self._ik_recovery_active = threading.Event()
            self._ik_recovery_generation = 0
            self._ik_recovery_lock = threading.Lock()

    sync_calls = []
    monkeypatch.setattr(
        "lerobot_robot_nero.async_client.sync_to_fixed_ready_pose",
        lambda *args, **kwargs: sync_calls.append((args, kwargs)),
    )
    client = Client()

    recover_nero_async_client_from_ik_failure(
        client,
        NeroAsyncSafetyConfig(
            fixed_ready_pose=pose,
            recover_on_ik_failure=True,
            dry_run=True,
        ),
    )

    assert sync_calls == []
    assert client.robot.last_action == pose
    assert client.latest_action == -1
    assert client.action_chunk_size == -1
    assert not client._ik_recovery_active.is_set()
    assert client._ik_recovery_generation == 2


def test_ee_safe_nero_robot_passes_fallback_seeds_and_records_selected_seed(tmp_path):
    robot = FakeEERobot()
    robot.right.joints = np.arange(7, dtype=float) + 0.3
    robot.left.joints = np.arange(7, dtype=float) + 30.3
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="ee-ik-fallback"),
        meta={"task": "trace"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    ik_adapter = MetadataIKAdapter()
    safe_robot = EESafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(),
        ee_adapter=FakeTraceEEAdapter(),
        ik_adapter=ik_adapter,
        tracer=tracer,
    )
    sent_seed = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    for idx in range(1, 8):
        sent_seed[f"right_nero_joint_{idx}"] = 1.0
        sent_seed[f"left_nero_joint_{idx}"] = 1.5
    safe_robot.set_last_action(sent_seed)

    safe_robot.send_action({name: float(idx) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)})
    safe_robot.send_action({name: float(idx) for idx, name in enumerate(EE_LOCAL_SE3_ACTION_NAMES)})
    tracer.close()

    first_call, second_call = ik_adapter.calls
    assert first_call["right_seed_candidates"][0][0] == "last_sent_action"
    np.testing.assert_allclose(first_call["right_seed_candidates"][0][1], np.ones(7))
    assert len(first_call["right_seed_candidates"]) == 1
    assert second_call["right_seed_candidates"][1][0] == "last_success_ik_solution"
    np.testing.assert_allclose(second_call["left_seed_candidates"][1][1], [0.51, 0, 0, 0, 0, 0, 0])

    trace_lines = [
        json.loads(line)
        for line in (tmp_path / "ee-ik-fallback/trace.jsonl").read_text().splitlines()
    ]
    success_events = [line["data"] for line in trace_lines if line["event"] == "ee_ik_joint_target"]
    assert success_events[-1]["ik_metadata"]["right"]["seed_source"] == "last_sent_action"
    assert success_events[-1]["ik_metadata"]["left"]["seed_source"] == "last_success_ik_solution"


def test_safe_nero_robot_records_policy_executor_and_replay_actions(tmp_path):
    robot = FakeRobot()
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="safe"),
        meta={"task": "trace"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    safety = NeroAsyncSafetyConfig(
        high_rate_control=True,
        max_policy_step_rad=0.1,
        max_gripper_step_m=0.1,
        max_executor_step_rad=0.02,
        max_executor_gripper_step_m=0.01,
    )
    safe_robot = SafeNeroRobot(robot, safety, tracer=tracer)
    target = {name: 0.0 for name in DEFAULT_DUAL_ACTION_NAMES}
    target["right_nero_joint_1"] = 1.0

    safe_robot.send_action(target)
    safe_robot.execute_high_rate_step(dt_s=0.005556)
    tracer.close()

    trace_text = (tmp_path / "safe/trace.jsonl").read_text()
    assert '"event": "policy_raw_action"' in trace_text
    assert '"event": "policy_limited_target"' in trace_text
    assert '"event": "executor_step"' in trace_text
    assert (tmp_path / "safe/replay_actions.jsonl").read_text().count("\n") == 1


def test_safe_nero_robot_records_high_rate_timing(tmp_path, monkeypatch):
    robot = FakeRobot()
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="timing"),
        meta={"task": "trace"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )
    safety = NeroAsyncSafetyConfig(high_rate_control=True)
    safe_robot = SafeNeroRobot(robot, safety, tracer=tracer)
    sleeps = []
    times = iter([10.0, 10.001, 10.008, 10.009])
    monkeypatch.setattr("lerobot_robot_nero.async_client.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("lerobot_robot_nero.async_client.precise_sleep", lambda value: sleeps.append(value))

    safe_robot._run_high_rate_executor_once(dt_s=1.0 / 180.0)
    tracer.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "timing/trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    timing = next(record for record in records if record["event"] == "executor_timing")
    assert timing["data"]["overrun_s"] == pytest.approx(0.008 - 1.0 / 180.0)
    assert timing["data"]["actual_hz"] == pytest.approx(125.0)
    assert sleeps == [0.0]


def test_observation_image_saver_writes_rgb_images_with_stride_and_limit(tmp_path):
    saver = ObservationImageSaver(
        NeroDebugImageSaveConfig(
            enabled=True,
            dir=str(tmp_path),
            every_n=2,
            max_frames=1,
        )
    )
    observation = {
        "front": np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8),
        "left_wrist": np.full((2, 3, 3), [0, 255, 0], dtype=np.uint8),
        "right_wrist": np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8),
        "right_nero_joint_1": 0.0,
    }

    saver.maybe_save(observation)
    saver.maybe_save(observation)
    saver.maybe_save(observation)
    saver.maybe_save(observation)

    files = sorted(path.name for path in tmp_path.glob("*.png"))
    assert files == [
        "000000_front.png",
        "000000_left_wrist.png",
        "000000_right_wrist.png",
    ]
    assert Image.open(tmp_path / "000000_front.png").getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(tmp_path / "000000_left_wrist.png").getpixel((0, 0)) == (0, 255, 0)
    assert Image.open(tmp_path / "000000_right_wrist.png").getpixel((0, 0)) == (0, 0, 255)


def test_observation_video_saver_writes_frames_and_encodes_dataset_style_videos(tmp_path, monkeypatch):
    encode_calls = []

    def fake_encode_video_frames(imgs_dir, video_path, fps, camera_encoder=None, overwrite=False, **kwargs):
        encode_calls.append((imgs_dir, video_path, fps, camera_encoder, overwrite))
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).write_bytes(b"fake mp4")

    monkeypatch.setattr(
        "lerobot_robot_nero.async_client.encode_video_frames",
        fake_encode_video_frames,
    )
    saver = ObservationVideoSaver(
        NeroDebugVideoSaveConfig(
            enabled=True,
            dir=str(tmp_path),
            fps=30,
            every_n=2,
            max_frames=2,
        )
    )
    observation = {
        "front": np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8),
        "left_wrist": np.full((2, 3, 3), [0, 255, 0], dtype=np.uint8),
        "right_wrist": np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8),
        "right_nero_joint_1": 0.0,
    }

    for _ in range(5):
        saver.maybe_save(observation)
    saver.close()

    assert sorted(path.name for path in (tmp_path / "frames/front").glob("*.png")) == [
        "frame-000000.png",
        "frame-000001.png",
    ]
    assert sorted(path.name for path in (tmp_path / "frames/left_wrist").glob("*.png")) == [
        "frame-000000.png",
        "frame-000001.png",
    ]
    assert sorted(path.name for path in (tmp_path / "frames/right_wrist").glob("*.png")) == [
        "frame-000000.png",
        "frame-000001.png",
    ]
    assert sorted(path.relative_to(tmp_path).as_posix() for path in (tmp_path / "videos").glob("**/*.mp4")) == [
        "videos/observation.images.front/chunk-000/file-000.mp4",
        "videos/observation.images.left_wrist/chunk-000/file-000.mp4",
        "videos/observation.images.right_wrist/chunk-000/file-000.mp4",
    ]
    assert [call[2] for call in encode_calls] == [30, 30, 30]
    assert all(call[4] is True for call in encode_calls)
    assert all(call[3].vcodec == "libsvtav1" for call in encode_calls)
    assert all(call[3].pix_fmt == "yuv420p" for call in encode_calls)
    assert all(call[3].g == 2 for call in encode_calls)
    assert all(call[3].crf == 30 for call in encode_calls)
    assert all(call[3].preset == 12 for call in encode_calls)


def test_dry_run_skips_physical_ready_pose_sync(monkeypatch):
    events = []

    class Client:
        running = False
        action_queue_size = []

        def __init__(self):
            self.robot = type(
                "SafeRobot",
                (),
                {
                    "set_last_action": lambda _, __: events.append("set_last"),
                    "start_high_rate_executor": lambda _: events.append("start_executor"),
                    "stop_high_rate_executor": lambda _: events.append("stop_executor"),
                },
            )()

        def start(self):
            events.append("start")
            return True

        def receive_actions(self):
            events.append("receive")

        def control_loop(self, task):
            events.append(("control", task))

        def stop(self):
            events.append("stop")

    monkeypatch.setattr("lerobot_robot_nero.async_client.init_logging", lambda: None)
    monkeypatch.setattr("lerobot_robot_nero.async_client.register_third_party_plugins", lambda: None)
    monkeypatch.setattr("lerobot_robot_nero.async_client._make_client_config", lambda cfg: object())
    monkeypatch.setattr(
        "lerobot_robot_nero.async_client._make_safe_robot_client",
        lambda cfg, safety, debug_save_images, debug_save_videos, tracer=None, nero_cfg=None: Client(),
    )
    monkeypatch.setattr(
        "lerobot_robot_nero.async_client.sync_to_fixed_ready_pose",
        lambda *args, **kwargs: events.append("sync"),
    )

    run_nero_async_client(
        NeroAsyncClientConfig(
            wait_for_enter=False,
            keyboard_stop=False,
            safety=NeroAsyncSafetyConfig(dry_run=True),
            task="test task",
        )
    )

    assert "sync" not in events
    assert "set_last" in events
    assert "start_executor" in events
    assert ("control", "test task") in events
