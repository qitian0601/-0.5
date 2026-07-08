import math

import numpy as np
import pytest

from scripts.convert_nero_flange_dataset_to_camera_rotvec_16d import (
    EE_ROTVEC_16D_FEATURE_NAMES,
    convert_flange14_to_camera_rotvec16,
    flange_pose6_to_matrix,
    matrix_to_rotvec,
)


def test_matrix_to_rotvec_matches_z_axis_yaw():
    matrix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rotvec = matrix_to_rotvec(matrix)

    np.testing.assert_allclose(rotvec, [0.0, 0.0, math.pi / 2], atol=1e-6)


def test_flange_pose6_to_matrix_uses_nero_rpy_convention():
    matrix = flange_pose6_to_matrix([1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2])

    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        matrix[:3, :3],
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-6,
    )


def test_convert_flange14_to_camera_rotvec16_uses_inverse_handeye_and_appends_base_zeros():
    right_base_cam = np.eye(4, dtype=np.float64)
    right_base_cam[:3, 3] = [10.0, 0.0, 0.0]
    left_base_cam = np.eye(4, dtype=np.float64)
    left_base_cam[:3, 3] = [0.0, 20.0, 0.0]
    flange14 = np.array(
        [
            11.0,
            2.0,
            3.0,
            0.0,
            0.0,
            math.pi / 2,
            4.0,
            22.0,
            6.0,
            0.0,
            0.0,
            0.0,
            0.03,
            0.04,
        ],
        dtype=np.float32,
    )

    converted = convert_flange14_to_camera_rotvec16(flange14, right_base_cam, left_base_cam)

    assert converted.dtype == np.float32
    assert converted.shape == (16,)
    np.testing.assert_allclose(converted[:7], [1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2, 0.03], atol=1e-6)
    np.testing.assert_allclose(converted[7:14], [4.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(converted[14:], [0.0, 0.0], atol=1e-6)


def test_convert_flange14_to_camera_rotvec16_rejects_wrong_shape():
    with pytest.raises(ValueError, match="14-D"):
        convert_flange14_to_camera_rotvec16(np.zeros(13, dtype=np.float32), np.eye(4), np.eye(4))


def test_ee_rotvec_16d_feature_names_match_training_contract():
    assert EE_ROTVEC_16D_FEATURE_NAMES == [
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
