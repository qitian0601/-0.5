import numpy as np
import pytest

from scripts.convert_nero_joint_dataset_to_ee_rotvec import (
    EE_FEATURE_NAMES,
    convert_joint16_to_ee14,
    rpy_to_rotvec,
)


class FakeArm:
    def __init__(self, offset):
        self.offset = offset
        self.calls = []

    def fk(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        self.calls.append(joints.copy())
        return [
            self.offset + joints[0],
            self.offset + joints[1],
            self.offset + joints[2],
            0.0,
            0.0,
            joints[6],
        ]


def test_rpy_to_rotvec_matches_z_axis_yaw():
    rotvec = rpy_to_rotvec([0.0, 0.0, np.pi / 2])

    np.testing.assert_allclose(rotvec, [0.0, 0.0, np.pi / 2], atol=1e-6)


def test_convert_joint16_to_ee14_preserves_arm_order_and_grippers():
    right = FakeArm(offset=10.0)
    left = FakeArm(offset=20.0)
    joint16 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            0.1,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            -0.2,
            0.03,
            0.04,
        ],
        dtype=np.float32,
    )

    ee14 = convert_joint16_to_ee14(joint16, right, left)

    assert ee14.dtype == np.float32
    assert ee14.shape == (14,)
    np.testing.assert_allclose(ee14[:7], [11.0, 12.0, 13.0, 0.0, 0.0, 0.1, 0.03], atol=1e-6)
    np.testing.assert_allclose(ee14[7:], [31.0, 32.0, 33.0, 0.0, 0.0, -0.2, 0.04], atol=1e-6)
    np.testing.assert_allclose(right.calls[0], joint16[:7])
    np.testing.assert_allclose(left.calls[0], joint16[7:14])


def test_convert_joint16_to_ee14_rejects_wrong_shape():
    with pytest.raises(ValueError, match="16-D"):
        convert_joint16_to_ee14(np.zeros(15, dtype=np.float32), FakeArm(0.0), FakeArm(0.0))


def test_ee_feature_names_are_right_first_left_second():
    assert EE_FEATURE_NAMES == [
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
    ]
