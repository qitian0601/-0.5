import math

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_nero.ee_local_se3_adapter import (
    EE_LOCAL_SE3_ACTION_NAMES,
    NeroEELocalSE3Adapter,
    SE3Transform,
)


def _flange_observation(
    *,
    right_pose=(0.2, -0.1, 0.3, 0.1, -0.2, 0.3),
    left_pose=(-0.2, 0.1, 0.4, -0.2, 0.1, -0.3),
    right_gripper=0.03,
    left_gripper=0.04,
):
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    observation = {f"right_flange_{name}": float(value) for name, value in zip(keys, right_pose, strict=True)}
    observation.update(
        {f"left_flange_{name}": float(value) for name, value in zip(keys, left_pose, strict=True)}
    )
    observation["right_gripper_width"] = float(right_gripper)
    observation["left_gripper_width"] = float(left_gripper)
    return observation


def _transform(euler_xyz, translation):
    return SE3Transform(
        rotation=Rotation.from_euler("xyz", euler_xyz).as_matrix(),
        translation=np.asarray(translation, dtype=float),
    )


def test_flange_observation_to_policy_state_uses_camera_frame_rotvec_layout():
    right_tf = _transform((0.0, 0.0, 0.5), (1.0, 2.0, 3.0))
    left_tf = _transform((0.2, -0.1, 0.0), (-1.0, 0.5, 0.25))
    adapter = NeroEELocalSE3Adapter(
        camera_from_right_base=right_tf,
        camera_from_left_base=left_tf,
        base_or_head_xy=(0.6, -0.7),
    )
    observation = _flange_observation()

    state = adapter.flange_observation_to_policy_state(observation)

    assert EE_LOCAL_SE3_ACTION_NAMES == [
        "right_ee_x",
        "right_ee_y",
        "right_ee_z",
        "right_ee_rotvec_x",
        "right_ee_rotvec_y",
        "right_ee_rotvec_z",
        "right_gripper_width",
        "left_ee_x",
        "left_ee_y",
        "left_ee_z",
        "left_ee_rotvec_x",
        "left_ee_rotvec_y",
        "left_ee_rotvec_z",
        "left_gripper_width",
        "base_or_head_x",
        "base_or_head_y",
    ]
    assert state.shape == (16,)

    right_pose = np.asarray([observation[f"right_flange_{name}"] for name in ("x", "y", "z", "roll", "pitch", "yaw")])
    expected_right_position = right_tf.rotation @ right_pose[:3] + right_tf.translation
    expected_right_rotvec = (
        Rotation.from_matrix(right_tf.rotation) * Rotation.from_euler("xyz", right_pose[3:6])
    ).as_rotvec()
    np.testing.assert_allclose(state[0:3], expected_right_position, atol=1e-8)
    np.testing.assert_allclose(state[3:6], expected_right_rotvec, atol=1e-8)
    assert state[6] == 0.03
    np.testing.assert_allclose(state[14:16], np.array([0.6, -0.7]))


def test_policy_camera_action_to_nero_euler_targets_inverts_camera_transform():
    adapter = NeroEELocalSE3Adapter(
        camera_from_right_base=_transform((0.3, -0.1, 0.2), (0.4, -0.5, 0.6)),
        camera_from_left_base=_transform((-0.2, 0.4, -0.1), (-0.6, 0.2, 0.3)),
    )
    observation = _flange_observation(
        right_pose=(0.12, -0.23, 0.34, 0.2, -0.3, 0.4),
        left_pose=(-0.11, 0.22, 0.33, -0.4, 0.2, -0.1),
        right_gripper=0.031,
        left_gripper=0.042,
    )
    action = adapter.flange_observation_to_policy_state(observation)
    action[6] = 0.055
    action[13] = 0.066
    action[14:16] = np.array([0.2, -0.1])

    targets = adapter.policy_action_to_nero_ee_targets(action)

    np.testing.assert_allclose(
        targets.right_pose,
        np.array([observation[f"right_flange_{name}"] for name in ("x", "y", "z", "roll", "pitch", "yaw")]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        targets.left_pose,
        np.array([observation[f"left_flange_{name}"] for name in ("x", "y", "z", "roll", "pitch", "yaw")]),
        atol=1e-8,
    )
    assert targets.right_gripper_width == 0.055
    assert targets.left_gripper_width == 0.066
    np.testing.assert_allclose(targets.base_or_head_xy, np.array([0.2, -0.1]))


def test_adapter_reads_flange_state_observation_without_tcp_offset():
    observation = _flange_observation(right_pose=(0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 6))

    class FakeRobot:
        def __init__(self):
            self.calls = []

        def get_flange_state_observation(self):
            self.calls.append("get_flange_state_observation")
            return observation

    robot = FakeRobot()
    adapter = NeroEELocalSE3Adapter()

    state = adapter.read_robot_policy_state(robot)

    assert robot.calls == ["get_flange_state_observation"]
    np.testing.assert_allclose(state[0:3], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(state[3:6], np.array([0.0, 0.0, math.pi / 6]))


def test_adapter_loads_camera_to_base_handeye_yamls_and_inverts_them():
    right_yaml = "/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_right_arm_tsai.yml"
    left_yaml = "/home/chenglong/workplace/nero_teleop_ws/data/lerobot/pickplace/handeye_left_arm_tsai.yml"

    adapter = NeroEELocalSE3Adapter.from_camera_to_base_yamls(
        right_camera_to_base_yaml=right_yaml,
        left_camera_to_base_yaml=left_yaml,
    )
    right_base_from_camera = SE3Transform.from_opencv_yaml(right_yaml, key="T_base_cam")
    left_base_from_camera = SE3Transform.from_opencv_yaml(left_yaml, key="T_base_cam")

    np.testing.assert_allclose(
        adapter.camera_from_right_base.as_matrix(),
        np.linalg.inv(right_base_from_camera.as_matrix()),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        adapter.camera_from_left_base.as_matrix(),
        np.linalg.inv(left_base_from_camera.as_matrix()),
        atol=1e-10,
    )
