#!/usr/bin/env python3
"""Convert Nero camera-frame EE rotvec 16D datasets to fixed-base EE14.

The source dataset is expected to store:

    [right xyz rotvec gripper, left xyz rotvec gripper, base_or_head_x, base_or_head_y]

The output drops the final two base/head dimensions, adds per-sample masks for
mixed robot/human training, and can re-encode videos to h264/yuv420p for
compatibility with the converted human-video datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTION = "action"
OBS_STATE = "observation.state"
VALID_ACTION_MASK = "valid_action_mask"
VALID_IMAGE_MASK = "valid_image_mask"
VIDEO_KEYS = [
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
SCALAR_FEATURES = ["timestamp", "frame_index", "episode_index", "index", "task_index"]

EE_ROTVEC_14D_FEATURE_NAMES = [
    "right_x",
    "right_y",
    "right_z",
    "right_rx",
    "right_ry",
    "right_rz",
    "right_gripper",
    "left_x",
    "left_y",
    "left_z",
    "left_rx",
    "left_ry",
    "left_rz",
    "left_gripper",
]
ROBOT_VALID_ACTION_MASK_14D = [True] * 14
ROBOT_VALID_IMAGE_MASK = [True, True, True]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _as_1d_float_array(value: Any, *, expected: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (expected,):
        raise ValueError(f"Expected {field} to have shape ({expected},), got {array.shape}")
    return array


def convert_vector16_to_14(value: Any, *, field: str) -> np.ndarray:
    return _as_1d_float_array(value, expected=16, field=field)[:14].astype(np.float32)


def _numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def _constant_mask_stats(mask: list[bool], *, total_frames: int) -> dict[str, list[float] | list[int]]:
    values = np.tile(np.asarray(mask, dtype=np.float32), (total_frames, 1))
    return _numeric_stats(values)


def _reorder_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [OBS_STATE, ACTION, VALID_ACTION_MASK, VALID_IMAGE_MASK, *SCALAR_FEATURES]
    ordered = [column for column in preferred if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return df[ordered]


def convert_parquet_file(src: Path, dst: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(src)
    for column in (OBS_STATE, ACTION):
        if column not in df.columns:
            raise ValueError(f"{src} is missing required column {column!r}")

    df = df.copy()
    df[OBS_STATE] = [convert_vector16_to_14(value, field=OBS_STATE) for value in df[OBS_STATE].values]
    df[ACTION] = [convert_vector16_to_14(value, field=ACTION) for value in df[ACTION].values]
    df[VALID_ACTION_MASK] = [list(ROBOT_VALID_ACTION_MASK_14D) for _ in range(len(df))]
    df[VALID_IMAGE_MASK] = [list(ROBOT_VALID_IMAGE_MASK) for _ in range(len(df))]
    df = _reorder_data_columns(df)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return {
        OBS_STATE: np.stack(df[OBS_STATE].values).astype(np.float32),
        ACTION: np.stack(df[ACTION].values).astype(np.float32),
        VALID_ACTION_MASK: np.asarray(df[VALID_ACTION_MASK].tolist(), dtype=np.float32),
        **{
            column: np.asarray(df[column].values, dtype=np.float32).reshape(-1, 1)
            for column in SCALAR_FEATURES
            if column in df.columns
        },
    }


def _slice_stat_vector(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == (16,):
            return value[:14].astype(float).tolist()
        return value
    if isinstance(value, list) and len(value) == 16:
        return value[:14]
    return value


def _slice_stats_payload(stats: dict[str, Any], *, total_frames: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in stats.items():
        if key in {ACTION, OBS_STATE}:
            output[key] = {stat_key: _slice_stat_vector(stat_value) for stat_key, stat_value in value.items()}
        elif key == VALID_IMAGE_MASK:
            continue
        else:
            output[key] = value
    output[VALID_ACTION_MASK] = _constant_mask_stats(ROBOT_VALID_ACTION_MASK_14D, total_frames=total_frames)
    return output


def _slice_episode_stats_value(value: Any) -> Any:
    return _slice_stat_vector(value)


def convert_episode_metadata_file(src: Path, dst: Path) -> None:
    df = pd.read_parquet(src).copy()
    for column in df.columns:
        if column.startswith("stats/action/") or column.startswith("stats/observation.state/"):
            df[column] = [_slice_episode_stats_value(value) for value in df[column].values]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def _video_feature_h264(feature: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(feature))
    info = updated.setdefault("info", {})
    info["video.codec"] = "h264"
    info["video.pix_fmt"] = "yuv420p"
    info["has_audio"] = False
    info["video.video_backend"] = "pyav"
    info.setdefault("video.g", 2)
    info.setdefault("video.crf", 30)
    info.setdefault("video.preset", 12)
    info.setdefault("video.fast_decode", 0)
    info.setdefault("video.extra_options", {})
    return updated


def update_info(src_root: Path, dst_root: Path, *, reencode_videos: bool) -> None:
    info = load_json(src_root / "meta" / "info.json")
    for key in (ACTION, OBS_STATE):
        feature = info["features"].get(key)
        if feature is None or feature.get("shape") not in ([16], (16,)):
            raise ValueError(f"Expected {key} to be 16-D in source info.json, got {feature}")
        feature["shape"] = [14]
        feature["names"] = EE_ROTVEC_14D_FEATURE_NAMES

    info["features"][VALID_ACTION_MASK] = {
        "dtype": "bool",
        "shape": [14],
        "names": EE_ROTVEC_14D_FEATURE_NAMES,
    }
    info["features"][VALID_IMAGE_MASK] = {
        "dtype": "bool",
        "shape": [3],
        "names": ["front", "left_wrist", "right_wrist"],
    }
    if reencode_videos:
        for key in VIDEO_KEYS:
            if key in info["features"]:
                info["features"][key] = _video_feature_h264(info["features"][key])

    write_json(dst_root / "meta" / "info.json", info)


def _copy_tasks(src_root: Path, dst_root: Path) -> None:
    src = src_root / "meta" / "tasks.parquet"
    if not src.exists():
        return
    tasks = pd.read_parquet(src)
    if "task" in tasks.columns:
        tasks = tasks.set_index("task")
    tasks.index.name = "task"
    dst = dst_root / "meta" / "tasks.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(dst)


def _copy_extra_metadata(src_root: Path, dst_root: Path) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for path in src_meta.iterdir():
        if path.name in {"info.json", "stats.json", "tasks.parquet", "episodes"}:
            continue
        target = dst_meta / path.name
        if path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target)
    _copy_tasks(src_root, dst_root)


def _copy_or_reencode_videos(src_root: Path, dst_root: Path, *, reencode_videos: bool) -> None:
    src_videos = src_root / "videos"
    if not src_videos.exists():
        return
    dst_videos = dst_root / "videos"
    if not reencode_videos:
        shutil.copytree(src_videos, dst_videos, symlinks=True)
        return

    for src in sorted(src_videos.glob("*/*/*.mp4")):
        dst = dst_videos / src.relative_to(src_videos)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {src} with exit code {result.returncode}:\n{result.stderr}")


def _prepare_output_root(dst_root: Path, *, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {dst_root}. Use --overwrite to replace it.")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)


def convert_dataset(
    src_root: str | Path,
    dst_root: str | Path,
    *,
    overwrite: bool = False,
    reencode_videos: bool = True,
    limit_files: int | None = None,
) -> None:
    src_root = Path(src_root).expanduser().resolve()
    dst_root = Path(dst_root).expanduser().resolve()

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot info.json under {src_root}")

    parquet_files = sorted((src_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")
    if limit_files is not None:
        parquet_files = parquet_files[:limit_files]

    _prepare_output_root(dst_root, overwrite=overwrite)
    _copy_extra_metadata(src_root, dst_root)
    update_info(src_root, dst_root, reencode_videos=reencode_videos)
    _copy_or_reencode_videos(src_root, dst_root, reencode_videos=reencode_videos)

    batches: dict[str, list[np.ndarray]] = {}
    for idx, src in enumerate(parquet_files, start=1):
        rel = src.relative_to(src_root)
        print(f"[{idx}/{len(parquet_files)}] converting {rel}", flush=True)
        converted = convert_parquet_file(src, dst_root / rel)
        for key, value in converted.items():
            batches.setdefault(key, []).append(value)

    src_stats = load_json(src_root / "meta" / "stats.json")
    total_frames = sum(batch.shape[0] for batch in batches[ACTION])
    stats = _slice_stats_payload(src_stats, total_frames=total_frames)
    for key in (ACTION, OBS_STATE, VALID_ACTION_MASK, *SCALAR_FEATURES):
        if key in batches:
            stats[key] = _numeric_stats(np.concatenate(batches[key], axis=0))
    stats.pop(VALID_IMAGE_MASK, None)
    write_json(dst_root / "meta" / "stats.json", stats)

    for src in sorted((src_root / "meta" / "episodes").glob("*/*.parquet")):
        convert_episode_metadata_file(src, dst_root / src.relative_to(src_root))

    print(f"Done: {dst_root}", flush=True)


def parse_args() -> argparse.Namespace:
    root = Path("/home/chenglong/workplace/nero_teleop_ws")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "data/lerobot/pickplace_new/pickplace_flange_pose_new_001_camera_rotvec_16d",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data/lerobot/pickplace_new/pickplace_flange_pose_new_001_camera_rotvec_14d_h264",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos without h264 re-encoding.")
    parser.add_argument("--limit-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.input_root,
        args.output_root,
        overwrite=args.overwrite,
        reencode_videos=not args.copy_videos,
        limit_files=args.limit_files,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
