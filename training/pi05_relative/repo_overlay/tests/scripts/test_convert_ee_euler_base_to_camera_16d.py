#!/usr/bin/env python

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert_ee_euler_base_to_camera_16d import (
    EE_EULER_GRIPPERS_LAST_14_NAMES,
    EE_EULER_14_NAMES,
    EE_ROTVEC_CAMERA_16_NAMES,
    convert_pose14_euler_base_to_camera16,
    euler_to_matrix,
    infer_input_layout_from_info,
    load_t_cam_base,
    update_info_json,
)


def _write_handeye(path: Path, matrix: np.ndarray) -> None:
    values = ", ".join(f"{value:.12g}" for value in matrix.reshape(-1))
    path.write_text(
        "%YAML:1.0\n"
        "---\n"
        "T_base_cam: !!opencv-matrix\n"
        "   rows: 4\n"
        "   cols: 4\n"
        "   dt: d\n"
        f"   data: [ {values} ]\n"
    )


def test_load_t_cam_base_inverts_t_base_cam() -> None:
    root = Path("/tmp/lerobot_convert_euler_handeye_test")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        t_base_cam = np.eye(4)
        t_base_cam[:3, 3] = [1.0, 2.0, 3.0]
        handeye_path = root / "handeye.yml"
        _write_handeye(handeye_path, t_base_cam)

        t_cam_base = load_t_cam_base(handeye_path)

        np.testing.assert_allclose(t_cam_base @ t_base_cam, np.eye(4), atol=1e-9)
    finally:
        shutil.rmtree(root)


def test_convert_pose14_euler_base_to_camera16_uses_camera_frame_and_rotvec() -> None:
    t_cam_base_right = np.eye(4)
    t_cam_base_right[:3, 3] = [1.0, 0.0, 0.0]
    t_cam_base_left = np.eye(4)
    t_cam_base_left[:3, :3] = euler_to_matrix([0.0, 0.0, np.pi / 2], order="xyz", degrees=False)

    pose14 = np.array(
        [
            0.2,
            0.3,
            0.4,
            0.0,
            0.0,
            np.pi / 2,
            0.01,
            1.0,
            0.0,
            0.0,
            0.0,
            np.pi / 2,
            0.0,
            0.02,
        ],
        dtype=np.float32,
    )

    converted = convert_pose14_euler_base_to_camera16(
        pose14,
        t_cam_base_right=t_cam_base_right,
        t_cam_base_left=t_cam_base_left,
        euler_order="xyz",
        degrees=False,
    )

    assert converted.shape == (16,)
    np.testing.assert_allclose(converted[:3], [1.2, 0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(
        euler_to_matrix(converted[3:6], order="rotvec", degrees=False),
        euler_to_matrix([0.0, 0.0, np.pi / 2], order="xyz", degrees=False),
        atol=1e-6,
    )
    np.testing.assert_allclose(converted[7:10], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(converted[[6, 13]], [0.01, 0.02], atol=1e-6)
    np.testing.assert_allclose(converted[14:16], [0.0, 0.0], atol=1e-6)


def test_convert_pose14_supports_grippers_last_layout() -> None:
    pose14 = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            1.1,
            1.2,
            0.0,
            0.0,
            0.0,
            0.01,
            0.02,
        ],
        dtype=np.float32,
    )

    converted = convert_pose14_euler_base_to_camera16(
        pose14,
        t_cam_base_right=np.eye(4),
        t_cam_base_left=np.eye(4),
        euler_order="xyz",
        degrees=False,
        input_layout="grippers_last",
    )

    np.testing.assert_allclose(converted[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(converted[7:10], [1.0, 1.1, 1.2], atol=1e-6)
    np.testing.assert_allclose(converted[[6, 13]], [0.01, 0.02], atol=1e-6)


def test_update_info_json_changes_14d_euler_features_to_16d_rotvec_camera_features() -> None:
    root = Path("/tmp/lerobot_convert_euler_info_test")
    if root.exists():
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    try:
        info = {
            "features": {
                "observation.state": {"dtype": "float32", "shape": [14], "names": EE_EULER_14_NAMES},
                "action": {"dtype": "float32", "shape": [14], "names": EE_EULER_14_NAMES},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            }
        }
        (root / "meta" / "info.json").write_text(json.dumps(info))

        update_info_json(root)

        updated = json.loads((root / "meta" / "info.json").read_text())
        assert updated["features"]["observation.state"]["shape"] == [16]
        assert updated["features"]["action"]["shape"] == [16]
        assert updated["features"]["observation.state"]["names"] == EE_ROTVEC_CAMERA_16_NAMES
        assert updated["features"]["action"]["names"] == EE_ROTVEC_CAMERA_16_NAMES
    finally:
        shutil.rmtree(root)


def test_infer_input_layout_from_info_detects_grippers_last() -> None:
    root = Path("/tmp/lerobot_convert_euler_layout_test")
    if root.exists():
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    try:
        info = {
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [14],
                    "names": EE_EULER_GRIPPERS_LAST_14_NAMES,
                }
            }
        }
        (root / "meta" / "info.json").write_text(json.dumps(info))

        assert infer_input_layout_from_info(root) == "grippers_last"
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_load_t_cam_base_inverts_t_base_cam()
    test_convert_pose14_euler_base_to_camera16_uses_camera_frame_and_rotvec()
    test_convert_pose14_supports_grippers_last_layout()
    test_update_info_json_changes_14d_euler_features_to_16d_rotvec_camera_features()
    test_infer_input_layout_from_info_detects_grippers_last()
