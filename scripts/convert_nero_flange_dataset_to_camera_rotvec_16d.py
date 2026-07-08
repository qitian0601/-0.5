#!/usr/bin/env python

"""Convert a Nero dual-arm flange-pose dataset to camera-frame EE rotvec 16D.

Input ``observation.state`` and ``action`` are expected to be 14-D:

    [right_x, right_y, right_z, right_roll, right_pitch, right_yaw,
     left_x,  left_y,  left_z,  left_roll,  left_pitch,  left_yaw,
     right_gripper_width, left_gripper_width]

Each arm's hand-eye YAML stores ``T_base_cam``.  This script converts flange
poses from the corresponding arm base frame into wrist-camera frame with
``T_cam_flange = inv(T_base_cam) * T_base_flange``.

Output ``observation.state`` and ``action`` are 16-D:

    [right_x, right_y, right_z, right_rx, right_ry, right_rz, right_gripper,
     left_x,  left_y,  left_z,  left_rx,  left_ry,  left_rz,  left_gripper,
     base_or_head_x, base_or_head_y]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTION = "action"
OBS_STATE = "observation.state"

FLANGE_14_FEATURE_NAMES = [
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

EE_ROTVEC_16D_FEATURE_NAMES = [
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

NUMERIC_STAT_KEYS = {ACTION, OBS_STATE, "timestamp", "frame_index", "episode_index", "index", "task_index"}


def rpy_to_matrix(rpy: list[float] | np.ndarray) -> np.ndarray:
    """Convert Nero SDK RPY to a rotation matrix.

    Nero flange pose uses ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``.
    """

    roll, pitch, yaw = [float(v) for v in rpy]
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def flange_pose6_to_matrix(pose6: list[float] | np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    if pose6.shape != (6,):
        raise ValueError(f"Expected 6-D flange pose [x,y,z,roll,pitch,yaw], got shape {pose6.shape}")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rpy_to_matrix(pose6[3:6])
    matrix[:3, 3] = pose6[:3]
    return matrix


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diag = np.diag(matrix)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {matrix.shape}")

    trace = float(np.trace(matrix))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(math.acos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    vee = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if math.pi - angle < 1e-5:
        quat = matrix_to_quat_xyzw(matrix)
        if quat[3] < 0:
            quat = -quat
        axis_norm = np.linalg.norm(quat[:3])
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return quat[:3] / axis_norm * angle

    return vee * (angle / (2.0 * math.sin(angle)))


def transform_pose6_to_camera_ee7(pose6: np.ndarray, gripper_width: float, t_base_cam: np.ndarray) -> np.ndarray:
    t_base_flange = flange_pose6_to_matrix(pose6)
    t_cam_flange = np.linalg.inv(t_base_cam) @ t_base_flange
    rotvec = matrix_to_rotvec(t_cam_flange[:3, :3])
    return np.asarray([*t_cam_flange[:3, 3].tolist(), *rotvec.tolist(), float(gripper_width)], dtype=np.float32)


def convert_flange14_to_camera_rotvec16(
    flange14: list[float] | np.ndarray,
    right_t_base_cam: np.ndarray,
    left_t_base_cam: np.ndarray,
) -> np.ndarray:
    flange14 = np.asarray(flange14, dtype=np.float64)
    if flange14.shape != (14,):
        raise ValueError(f"Expected 14-D Nero flange vector, got shape {flange14.shape}")

    right = transform_pose6_to_camera_ee7(flange14[:6], float(flange14[12]), right_t_base_cam)
    left = transform_pose6_to_camera_ee7(flange14[6:12], float(flange14[13]), left_t_base_cam)
    return np.concatenate((right, left, np.zeros(2, dtype=np.float32))).astype(np.float32)


def load_opencv_yaml_matrix(path: Path, key: str = "T_base_cam") -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    pattern = rf"{re.escape(key)}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[(.*?)\]"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find OpenCV matrix {key!r} in {path}")

    rows = int(match.group(1))
    cols = int(match.group(2))
    values = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(3))]
    if len(values) != rows * cols:
        raise ValueError(f"Matrix {key!r} in {path} has {len(values)} values, expected {rows * cols}")
    return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def collect_numeric_batches(df: pd.DataFrame) -> dict[str, np.ndarray]:
    batches: dict[str, np.ndarray] = {}
    for key in NUMERIC_STAT_KEYS:
        if key not in df.columns:
            continue
        values = df[key].values
        if key in (ACTION, OBS_STATE):
            batches[key] = np.stack(values).astype(np.float64)
        else:
            batches[key] = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return batches


def merge_stats(batches_by_file: list[dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for batches in batches_by_file:
        for key, value in batches.items():
            grouped.setdefault(key, []).append(value)

    stats: dict[str, dict[str, list[float]]] = {}
    for key, arrays in grouped.items():
        data = np.concatenate(arrays, axis=0).astype(np.float64)
        stats[key] = {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "count": [int(data.shape[0])],
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q10": np.quantile(data, 0.10, axis=0).tolist(),
            "q50": np.quantile(data, 0.50, axis=0).tolist(),
            "q90": np.quantile(data, 0.90, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def convert_parquet_file(
    src: Path,
    dst: Path,
    right_t_base_cam: np.ndarray,
    left_t_base_cam: np.ndarray,
) -> dict[str, np.ndarray]:
    df = pd.read_parquet(src)
    for column in (OBS_STATE, ACTION):
        if column not in df.columns:
            raise ValueError(f"{src} is missing required column {column!r}")

    df = df.copy()
    df[OBS_STATE] = [
        convert_flange14_to_camera_rotvec16(value, right_t_base_cam, left_t_base_cam)
        for value in df[OBS_STATE].values
    ]
    df[ACTION] = [
        convert_flange14_to_camera_rotvec16(value, right_t_base_cam, left_t_base_cam)
        for value in df[ACTION].values
    ]

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return collect_numeric_batches(df)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def update_info(src_root: Path, dst_root: Path) -> None:
    info = load_json(src_root / "meta" / "info.json")
    for key in (ACTION, OBS_STATE):
        info["features"][key]["shape"] = [16]
        info["features"][key]["names"] = EE_ROTVEC_16D_FEATURE_NAMES
    write_json(dst_root / "meta" / "info.json", info)


def copy_static_metadata_and_videos(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {dst_root}. Use --overwrite to replace it.")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True)
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for path in src_meta.iterdir():
        if path.name in {"info.json", "stats.json"}:
            continue
        target = dst_meta / path.name
        if path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target)

    src_videos = src_root / "videos"
    if src_videos.exists():
        shutil.copytree(src_videos, dst_root / "videos", symlinks=True)


def validate_source_info(src_root: Path) -> None:
    info = load_json(src_root / "meta" / "info.json")
    for key in (ACTION, OBS_STATE):
        feature = info["features"].get(key)
        if feature is None or feature.get("shape") not in ([14], (14,)):
            raise ValueError(f"Expected {key} to be 14-D in source dataset, got {feature}")


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    right_handeye: Path,
    left_handeye: Path,
    overwrite: bool = False,
    limit_files: int | None = None,
) -> None:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    right_handeye = right_handeye.expanduser().resolve()
    left_handeye = left_handeye.expanduser().resolve()

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot info.json under {src_root}")
    validate_source_info(src_root)

    parquet_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")
    if limit_files is not None:
        parquet_files = parquet_files[:limit_files]

    right_t_base_cam = load_opencv_yaml_matrix(right_handeye)
    left_t_base_cam = load_opencv_yaml_matrix(left_handeye)
    copy_static_metadata_and_videos(src_root, dst_root, overwrite=overwrite)
    update_info(src_root, dst_root)

    all_batches = []
    for idx, src in enumerate(parquet_files, start=1):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        print(f"[{idx}/{len(parquet_files)}] converting {rel}", flush=True)
        all_batches.append(convert_parquet_file(src, dst, right_t_base_cam, left_t_base_cam))

    write_json(dst_root / "meta" / "stats.json", merge_stats(all_batches))
    print(f"Done: {dst_root}", flush=True)


def parse_args() -> argparse.Namespace:
    root = Path("/home/chenglong/workplace/nero_teleop_ws")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/lerobot/pickplace_new/pickplace_flange_pose_new_001",
        help="Source LeRobot dataset root with 14-D Nero flange observation.state/action.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/lerobot/pickplace_new/pickplace_flange_pose_new_001_camera_rotvec_16d",
        help="Output LeRobot dataset root with 16-D camera-frame EE rotvec observation.state/action.",
    )
    parser.add_argument(
        "--right-handeye",
        type=Path,
        default=root / "lerobot/相机_机械臂标定/handeye_result_right(1).yml",
        help="Right arm OpenCV YAML containing T_base_cam.",
    )
    parser.add_argument(
        "--left-handeye",
        type=Path,
        default=root / "lerobot/相机_机械臂标定/handeye_result_left(1).yml",
        help="Left arm OpenCV YAML containing T_base_cam.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory first if it exists.")
    parser.add_argument("--limit-files", type=int, default=None, help="Convert only first N parquet files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.input,
        args.output,
        args.right_handeye,
        args.left_handeye,
        overwrite=args.overwrite,
        limit_files=args.limit_files,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
