#!/usr/bin/env python

"""Append fixed base/head XY channels to a 14-D camera-frame EE dataset.

The input dataset must already store absolute camera-frame poses as:

    [right xyz, right rotvec, right gripper, left xyz, left rotvec, left gripper]

The output keeps those absolute values and appends two fixed channels:

    [right xyz, right rotvec, right gripper,
     left xyz, left rotvec, left gripper,
     base_or_head_x, base_or_head_y]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EE_14_NAMES = [
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

EE_16_NAMES = [
    *EE_14_NAMES,
    "base_or_head_x",
    "base_or_head_y",
]


def hardlink_or_copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src_path in src.rglob("*"):
        rel_path = src_path.relative_to(src)
        dst_path = dst / rel_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_path.hardlink_to(src_path)
        except OSError:
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
            hardlink_or_copy_tree(item, target)
        else:
            try:
                target.hardlink_to(item)
            except OSError:
                shutil.copy2(item, target)


def append_base_head_xy(pose: object, base_head_xy: tuple[float, float]) -> np.ndarray:
    pose_array = np.asarray(pose, dtype=np.float32)
    if pose_array.shape != (14,):
        raise ValueError(f"Expected 14-D pose, got shape {pose_array.shape}")
    return np.concatenate((pose_array, np.asarray(base_head_xy, dtype=np.float32)))


def convert_parquet_files(src_root: Path, dst_root: Path, base_head_xy: tuple[float, float]) -> None:
    parquet_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")

    for parquet_path in parquet_files:
        rel_path = parquet_path.relative_to(src_root)
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(parquet_path)
        for column in ("observation.state", "action"):
            if column not in df.columns:
                raise ValueError(f"Missing column {column!r} in {parquet_path}")
            df[column] = df[column].map(lambda pose: append_base_head_xy(pose, base_head_xy))
        df.to_parquet(out_path, index=False)


def update_info_json(dst_root: Path, src_root: Path, base_head_xy: tuple[float, float]) -> None:
    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature.get("shape") not in ([14], (14,)):
            raise ValueError(f"Expected {key} shape [14] before conversion, got {feature.get('shape')}")
        feature["shape"] = [16]
        feature["names"] = EE_16_NAMES
    info["base_head_xy_extension"] = {
        "source_dataset": str(src_root),
        "source_layout": "[xyz, rotvec, gripper] * 2 camera-frame absolute EE pose",
        "added_dims": ["base_or_head_x", "base_or_head_y"],
        "base_or_head_x": base_head_xy[0],
        "base_or_head_y": base_head_xy[1],
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def update_absolute_stats(dst_root: Path, base_head_xy: tuple[float, float]) -> None:
    stats_path = dst_root / "meta" / "stats.json"
    if not stats_path.exists():
        return

    stats = json.loads(stats_path.read_text())
    backup_path = dst_root / "meta" / "stats_14d_camera_rotvec_before_base_head_xy.json"
    stats_path.replace(backup_path)

    x, y = base_head_xy
    for key in ("observation.state", "action"):
        if key not in stats:
            continue
        for stat_name in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            stats[key][stat_name].extend([x, y])
        stats[key]["std"].extend([0.0, 0.0])

    stats_path.write_text(json.dumps(stats, indent=4) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--base-head-x", type=float, default=0.0)
    parser.add_argument("--base-head-y", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    base_head_xy = (args.base_head_x, args.base_head_y)

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing source dataset metadata: {src_root / 'meta' / 'info.json'}")

    copy_dataset_skeleton(src_root, dst_root)
    convert_parquet_files(src_root, dst_root, base_head_xy)
    update_info_json(dst_root, src_root, base_head_xy)
    update_absolute_stats(dst_root, base_head_xy)
    print(f"Wrote 16-D dataset to {dst_root}")


if __name__ == "__main__":
    main()
