#!/usr/bin/env python

"""Convert a Nero dual-arm joint LeRobot dataset to absolute EE rotvec pose.

Input ``observation.state`` and ``action`` are expected to be 16-D:

    [right_j1..right_j7, left_j1..left_j7, right_gripper_width, left_gripper_width]

Output ``observation.state`` and ``action`` are 14-D:

    [right_x, right_y, right_z, right_rx, right_ry, right_rz, right_gripper_width,
     left_x,  left_y,  left_z,  left_rx,  left_ry,  left_rz,  left_gripper_width]

The Nero SDK FK returns ``[x, y, z, roll, pitch, yaw]``.  This script converts
that RPY orientation to a rotation vector using the same convention documented
by the SDK: ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
except ImportError as exc:  # pragma: no cover - exercised by real CLI environment
    raise SystemExit(
        "Could not import pyAgxArm. Run this script from the lerobot conda env inside "
        "nero_teleop_ws, where third_party/pyAgxArm is available."
    ) from exc


ACTION = "action"
OBS_STATE = "observation.state"

JOINT_FEATURE_NAMES = [
    "right_nero_joint_1",
    "right_nero_joint_2",
    "right_nero_joint_3",
    "right_nero_joint_4",
    "right_nero_joint_5",
    "right_nero_joint_6",
    "right_nero_joint_7",
    "left_nero_joint_1",
    "left_nero_joint_2",
    "left_nero_joint_3",
    "left_nero_joint_4",
    "left_nero_joint_5",
    "left_nero_joint_6",
    "left_nero_joint_7",
    "right_gripper_width",
    "left_gripper_width",
]

EE_FEATURE_NAMES = [
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

NUMERIC_STAT_KEYS = {ACTION, OBS_STATE, "timestamp", "frame_index", "episode_index", "index", "task_index"}


def rpy_to_rotvec(rpy: list[float] | np.ndarray) -> np.ndarray:
    """Convert Nero SDK RPY to rotation vector.

    Nero documents the RPY convention as ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``.
    The returned vector has direction equal to the rotation axis and magnitude
    equal to the rotation angle in radians.
    """

    roll, pitch, yaw = [float(v) for v in rpy]
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    matrix = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return matrix_to_rotvec(matrix)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
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


def convert_pose6_to_ee7(pose6: list[float] | np.ndarray, gripper_width: float) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    if pose6.shape != (6,):
        raise ValueError(f"FK pose must be 6-D [x,y,z,roll,pitch,yaw], got shape {pose6.shape}")
    rotvec = rpy_to_rotvec(pose6[3:6])
    return np.asarray([pose6[0], pose6[1], pose6[2], *rotvec, gripper_width], dtype=np.float32)


def convert_joint16_to_ee14(joint16: np.ndarray, right_arm: Any, left_arm: Any) -> np.ndarray:
    joint16 = np.asarray(joint16, dtype=np.float64)
    if joint16.shape != (16,):
        raise ValueError(f"Expected 16-D Nero joint vector, got shape {joint16.shape}")

    right_pose = right_arm.fk(joint16[:7].tolist())
    left_pose = left_arm.fk(joint16[7:14].tolist())
    right_ee = convert_pose6_to_ee7(right_pose, float(joint16[14]))
    left_ee = convert_pose6_to_ee7(left_pose, float(joint16[15]))
    return np.concatenate((right_ee, left_ee)).astype(np.float32)


def make_nero_fk_arm(channel: str = "can0") -> Any:
    config = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V120, channel=channel)
    return AgxArmFactory.create_arm(config)


def convert_parquet_file(src: Path, dst: Path, right_arm: Any, left_arm: Any) -> dict[str, np.ndarray]:
    df = pd.read_parquet(src)
    for column in (OBS_STATE, ACTION):
        if column not in df.columns:
            raise ValueError(f"{src} is missing required column {column!r}")

    df = df.copy()
    df[OBS_STATE] = [convert_joint16_to_ee14(value, right_arm, left_arm) for value in df[OBS_STATE].values]
    df[ACTION] = [convert_joint16_to_ee14(value, right_arm, left_arm) for value in df[ACTION].values]

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return collect_numeric_batches(df)


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
        info["features"][key]["shape"] = [14]
        info["features"][key]["names"] = EE_FEATURE_NAMES
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


def convert_dataset(src_root: Path, dst_root: Path, overwrite: bool = False, limit_files: int | None = None) -> None:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot info.json under {src_root}")

    info = load_json(src_root / "meta" / "info.json")
    for key in (ACTION, OBS_STATE):
        feature = info["features"].get(key)
        if feature is None or feature.get("shape") not in ([16], (16,)):
            raise ValueError(f"Expected {key} to be 16-D in source dataset, got {feature}")

    parquet_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")
    if limit_files is not None:
        parquet_files = parquet_files[:limit_files]

    copy_static_metadata_and_videos(src_root, dst_root, overwrite=overwrite)
    update_info(src_root, dst_root)

    right_arm = make_nero_fk_arm("nero_right")
    left_arm = make_nero_fk_arm("nero_left")
    all_batches = []
    for idx, src in enumerate(parquet_files, start=1):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        print(f"[{idx}/{len(parquet_files)}] converting {rel}", flush=True)
        all_batches.append(convert_parquet_file(src, dst, right_arm, left_arm))

    stats = merge_stats(all_batches)
    write_json(dst_root / "meta" / "stats.json", stats)
    print(f"Done: {dst_root}", flush=True)
    print("Note: action stats are absolute EE stats. Run EE/SO3 relative stats recompute before PI0.5 training.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_final"),
        help="Source LeRobot dataset root with 16-D Nero joint observation.state/action.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/chenglong/workplace/nero_teleop_ws/data/lerobot/fold_towel/fold_towel_final_ee_rotvec"),
        help="Output LeRobot dataset root with 14-D absolute EE rotvec observation.state/action.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory first if it exists.")
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Convert only the first N parquet files. Intended for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.input, args.output, overwrite=args.overwrite, limit_files=args.limit_files)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
