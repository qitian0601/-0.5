#!/usr/bin/env python

"""Convert base-frame Euler EE LeRobot data to camera-frame rotvec 16-D data.

Input parquet rows are expected to store absolute dual-arm EE poses in each
arm's own base frame:

    [right xyz, right euler, right gripper, left xyz, left euler, left gripper]

The output dataset keeps absolute target poses, but expresses both arms in the
center camera frame, converts orientation to rotation vectors, and appends the
2-D base/head channel required by the EE-local SE(3) PI0.5 interface:

    [right camera xyz, right camera rotvec, right gripper,
     left camera xyz, left camera rotvec, left gripper,
     base_or_head_x, base_or_head_y]

The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path


def import_numpy_with_retries(max_attempts: int = 8):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            import numpy as np

            return np
        except (ImportError, ModuleNotFoundError) as error:
            last_error = error
            for name in list(sys.modules):
                if name == "numpy" or name.startswith("numpy."):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
            wait_s = min(2 ** (attempt - 1), 10)
            print(f"Import numpy failed on attempt {attempt}/{max_attempts}: {error}")
            if attempt < max_attempts:
                print(f"Retrying import in {wait_s}s...")
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to import numpy after {max_attempts} attempts") from last_error


np = import_numpy_with_retries()

EE_EULER_14_NAMES = [
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_roll",
    "right_ee_pitch",
    "right_ee_yaw",
    "right_gripper_width",
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_roll",
    "left_ee_pitch",
    "left_ee_yaw",
    "left_gripper_width",
]

EE_EULER_GRIPPERS_LAST_14_NAMES = [
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

EE_ROTVEC_CAMERA_16_NAMES = [
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

_AXES = {"x", "y", "z"}


def _parse_opencv_matrix(text: str, key: str) -> np.ndarray:
    match = re.search(rf"{re.escape(key)}:.*?data:\s*\[([^\]]+)\]", text, re.S)
    if match is None:
        raise ValueError(f"Missing OpenCV matrix key {key!r}")
    values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", match.group(1))]
    if len(values) != 16:
        raise ValueError(f"Expected 16 values for {key!r}, got {len(values)}")
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{key!r} is not a homogeneous transform")
    return matrix


def _validate_rotation_matrix(name: str, rotation: np.ndarray) -> None:
    det = np.linalg.det(rotation)
    orth_err = np.linalg.norm(rotation.T @ rotation - np.eye(3))
    if not np.isclose(det, 1.0, atol=1e-6) or orth_err > 1e-6:
        raise ValueError(f"{name} rotation is invalid: det={det:.8f}, orth_err={orth_err:.3e}")


def load_t_cam_base(handeye_path: Path) -> np.ndarray:
    """Load OpenCV hand-eye YAML with T_base_cam and return T_cam_base."""
    text = handeye_path.read_text()
    t_base_cam = _parse_opencv_matrix(text, "T_base_cam")
    _validate_rotation_matrix(f"{handeye_path}:T_base_cam", t_base_cam[:3, :3])
    return np.linalg.inv(t_base_cam)


def _axis_matrix(axis: str, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported Euler axis {axis!r}")


def euler_to_matrix(values: np.ndarray | list[float], order: str, degrees: bool = False) -> np.ndarray:
    """Convert extrinsic Euler angles or a rotvec to a rotation matrix.

    `order="xyz"` matches fixed-axis roll-pitch-yaw composition:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 rotation values, got shape {values.shape}")
    if order == "rotvec":
        return rotvec_to_matrix(values)
    if len(order) != 3 or any(axis not in _AXES for axis in order):
        raise ValueError(f"Euler order must contain three axes from x/y/z, got {order!r}")
    if degrees:
        values = np.deg2rad(values)

    rotation = np.eye(3, dtype=np.float64)
    for axis, angle in zip(order, values, strict=True):
        rotation = _axis_matrix(axis, float(angle)) @ rotation
    return rotation


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec)
    k = _skew(rotvec)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) + k + 0.5 * (k @ k)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) / theta * k
        + (1.0 - math.cos(theta)) / (theta * theta) * (k @ k)
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    _validate_rotation_matrix("matrix_to_rotvec input", matrix)
    cos_angle = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(float(cos_angle))
    if angle < 1e-8:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        )
    if math.pi - angle < 1e-5:
        axis = np.empty(3, dtype=np.float64)
        axis[0] = math.sqrt(max(0.0, (matrix[0, 0] + 1.0) / 2.0))
        axis[1] = math.sqrt(max(0.0, (matrix[1, 1] + 1.0) / 2.0))
        axis[2] = math.sqrt(max(0.0, (matrix[2, 2] + 1.0) / 2.0))
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("Cannot recover rotation axis near pi")
        return angle * axis / norm
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    return angle / (2.0 * math.sin(angle)) * axis


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def import_pandas_with_retries(max_attempts: int = 8):
    last_error = None
    prefixes = ("pandas", "numpy.random")
    for attempt in range(1, max_attempts + 1):
        try:
            import pandas as pd

            return pd
        except (ImportError, ModuleNotFoundError) as error:
            last_error = error
            for name in list(sys.modules):
                if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
            wait_s = min(2 ** (attempt - 1), 10)
            print(f"Import pandas failed on attempt {attempt}/{max_attempts}: {error}")
            if attempt < max_attempts:
                print(f"Retrying import in {wait_s}s...")
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to import pandas after {max_attempts} attempts") from last_error


def pose7_euler_to_transform(pose: np.ndarray, euler_order: str, degrees: bool) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_to_matrix(pose[3:6], order=euler_order, degrees=degrees)
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose7_rotvec(transform: np.ndarray, gripper: float) -> np.ndarray:
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = transform[:3, 3].astype(np.float32)
    pose[3:6] = matrix_to_rotvec(transform[:3, :3]).astype(np.float32)
    pose[6] = np.float32(gripper)
    return pose


def convert_pose14_euler_base_to_camera16(
    pose14: np.ndarray,
    t_cam_base_right: np.ndarray,
    t_cam_base_left: np.ndarray,
    euler_order: str,
    degrees: bool,
    base_head_xy: tuple[float, float] = (0.0, 0.0),
    input_layout: str = "right7_left7",
) -> np.ndarray:
    pose14 = np.asarray(pose14, dtype=np.float64)
    if pose14.shape != (14,):
        raise ValueError(f"Expected 14-D pose, got shape {pose14.shape}")

    right_pose, left_pose = split_pose14(pose14, input_layout=input_layout)
    right_base = pose7_euler_to_transform(right_pose, euler_order=euler_order, degrees=degrees)
    left_base = pose7_euler_to_transform(left_pose, euler_order=euler_order, degrees=degrees)
    right_camera = t_cam_base_right @ right_base
    left_camera = t_cam_base_left @ left_base

    return np.concatenate(
        (
            transform_to_pose7_rotvec(right_camera, right_pose[6]),
            transform_to_pose7_rotvec(left_camera, left_pose[6]),
            np.asarray(base_head_xy, dtype=np.float32),
        )
    ).astype(np.float32)


def split_pose14(pose14: np.ndarray, input_layout: str) -> tuple[np.ndarray, np.ndarray]:
    if input_layout == "right7_left7":
        return pose14[:7], pose14[7:14]
    if input_layout == "grippers_last":
        right = np.concatenate((pose14[:6], pose14[12:13]))
        left = np.concatenate((pose14[6:12], pose14[13:14]))
        return right, left
    raise ValueError(f"Unsupported input_layout {input_layout!r}")


def infer_input_layout_from_info(dst_root: Path) -> str:
    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    names = info["features"]["observation.state"].get("names")
    if names == EE_EULER_GRIPPERS_LAST_14_NAMES:
        return "grippers_last"
    if names == EE_EULER_14_NAMES:
        return "right7_left7"
    if names and len(names) == 14:
        if names[6] == "left_flange_x" and names[12] == "right_gripper_width":
            return "grippers_last"
        if names[6] == "right_gripper_width" and names[13] == "left_gripper_width":
            return "right7_left7"
    raise ValueError(
        "Cannot infer 14-D input layout from observation.state names. "
        "Pass --input-layout right7_left7 or --input-layout grippers_last."
    )


def copy_tree(src: Path, dst: Path, use_hardlinks: bool = False) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src_path in src.rglob("*"):
        rel_path = src_path.relative_to(src)
        dst_path = dst / rel_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if use_hardlinks:
            try:
                dst_path.hardlink_to(src_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def copy_dataset_skeleton(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        raise FileExistsError(f"Output dataset already exists: {dst_root}")
    dst_root.mkdir(parents=True)

    for item in src_root.iterdir():
        if item.name == "data":
            continue
        target = dst_root / item.name
        if item.is_dir():
            copy_tree(item, target, use_hardlinks=(item.name == "videos"))
        else:
            shutil.copy2(item, target)


def update_info_json(dst_root: Path) -> None:
    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature.get("shape") not in ([14], (14,)):
            raise ValueError(f"Expected {key} shape [14] before conversion, got {feature.get('shape')}")
        feature["shape"] = [16]
        feature["names"] = EE_ROTVEC_CAMERA_16_NAMES
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def backup_old_stats(dst_root: Path) -> None:
    stats_path = dst_root / "meta" / "stats.json"
    if stats_path.exists():
        backup_path = dst_root / "meta" / "stats_14d_euler_base_before_camera_16d.json"
        stats_path.replace(backup_path)


def convert_parquet_files(
    src_root: Path,
    dst_root: Path,
    t_cam_base_right: np.ndarray,
    t_cam_base_left: np.ndarray,
    euler_order: str,
    degrees: bool,
    base_head_xy: tuple[float, float],
    input_layout: str,
) -> None:
    parquet_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")

    pd = import_pandas_with_retries()
    for parquet_path in parquet_files:
        rel_path = parquet_path.relative_to(src_root)
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(parquet_path)
        for column in ("observation.state", "action"):
            if column not in df.columns:
                raise ValueError(f"Missing column {column!r} in {parquet_path}")
            df[column] = df[column].map(
                lambda pose: convert_pose14_euler_base_to_camera16(
                    pose,
                    t_cam_base_right=t_cam_base_right,
                    t_cam_base_left=t_cam_base_left,
                    euler_order=euler_order,
                    degrees=degrees,
                    base_head_xy=base_head_xy,
                    input_layout=input_layout,
                )
            )
        df.to_parquet(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--right-handeye", type=Path, required=True)
    parser.add_argument("--left-handeye", type=Path, required=True)
    parser.add_argument(
        "--euler-order",
        default="xyz",
        help="Extrinsic/fixed-axis Euler order. Default xyz means R = Rz(yaw) @ Ry(pitch) @ Rx(roll).",
    )
    parser.add_argument("--degrees", action="store_true", help="Set if source Euler angles are in degrees.")
    parser.add_argument(
        "--input-layout",
        choices=["auto", "right7_left7", "grippers_last"],
        default="auto",
        help=(
            "14-D source layout. right7_left7 means [right xyz/euler/gripper, left xyz/euler/gripper]. "
            "grippers_last means [right xyz/euler, left xyz/euler, right gripper, left gripper]."
        ),
    )
    parser.add_argument("--base-head-x", type=float, default=0.0)
    parser.add_argument("--base-head-y", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    right_handeye = args.right_handeye.expanduser().resolve()
    left_handeye = args.left_handeye.expanduser().resolve()

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing source dataset metadata: {src_root / 'meta' / 'info.json'}")

    t_cam_base_right = load_t_cam_base(right_handeye)
    t_cam_base_left = load_t_cam_base(left_handeye)

    copy_dataset_skeleton(src_root, dst_root)
    backup_old_stats(dst_root)
    input_layout = infer_input_layout_from_info(dst_root) if args.input_layout == "auto" else args.input_layout
    update_info_json(dst_root)
    convert_parquet_files(
        src_root=src_root,
        dst_root=dst_root,
        t_cam_base_right=t_cam_base_right,
        t_cam_base_left=t_cam_base_left,
        euler_order=args.euler_order,
        degrees=args.degrees,
        base_head_xy=(args.base_head_x, args.base_head_y),
        input_layout=input_layout,
    )

    print(f"Converted dataset written to {dst_root}")
    print(f"Input layout: {input_layout}")
    print("Next step: run scripts/recompute_ee_local_se3_relative_stats.py on the converted dataset.")


if __name__ == "__main__":
    main()
