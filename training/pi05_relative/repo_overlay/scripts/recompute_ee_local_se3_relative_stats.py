#!/usr/bin/env python

"""Recompute PI0.5 EE-local SE(3) relative action stats for a LeRobot dataset.

The dataset parquet remains absolute egocentric pose:

    observation.state = [xyz, rotvec, gripper] * 2 + [base_x, base_y]
    action            = [xyz, rotvec, gripper] * 2 + [base_x, base_y]

This script recomputes numeric dataset stats, then rewrites ``stats["action"]``
so action normalization matches the relative action distribution produced during
training with:

    --policy.use_relative_actions=true
    --policy.relative_action_type=ee_local_se3
"""

from __future__ import annotations

import argparse
import importlib
import logging
from types import SimpleNamespace
from pathlib import Path
import sys
import time

import numpy as np

DATA_DIR = "data"
ACTION = "action"
OBS_STATE = "observation.state"
_META_KEYS = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
_BACKUP_STATS_PATTERNS = (
    "stats_*before*.json",
    "stats_absolute_ee_before_relative.json",
)


def clear_partial_imports() -> None:
    prefixes = (
        "torch",
        "datasets",
        "pandas",
        "lerobot.datasets",
        "lerobot.utils",
        "lerobot.processor",
        "multiprocess",
    )
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


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
            logging.warning("Import pandas failed on attempt %d/%d: %s", attempt, max_attempts, error)
            if attempt < max_attempts:
                logging.warning("Retrying import in %ds...", wait_s)
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to import pandas after {max_attempts} attempts") from last_error


def import_stats_dependencies(max_attempts: int = 8):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            from lerobot.datasets.compute_stats import (
                aggregate_stats,
                compute_ee_local_se3_relative_action_stats,
                compute_episode_stats,
            )
            from lerobot.datasets.io_utils import cast_stats_to_numpy, write_stats
            from lerobot.utils.io_utils import load_json

            return (
                aggregate_stats,
                compute_ee_local_se3_relative_action_stats,
                compute_episode_stats,
                cast_stats_to_numpy,
                load_json,
                write_stats,
            )
        except (ImportError, ModuleNotFoundError, RuntimeError) as error:
            last_error = error
            clear_partial_imports()
            wait_s = min(2 ** (attempt - 1), 10)
            logging.warning(
                "Import LeRobot stats dependencies failed on attempt %d/%d: %s",
                attempt,
                max_attempts,
                error,
            )
            if attempt < max_attempts:
                logging.warning("Retrying import in %ds...", wait_s)
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to import LeRobot stats dependencies after {max_attempts} attempts") from last_error


def _load_local_dataset_view(dataset_root: Path, cast_stats_to_numpy, load_json):
    pd = import_pandas_with_retries()
    info = load_json(dataset_root / "meta" / "info.json")
    stats_path = dataset_root / "meta" / "stats.json"
    stats = cast_stats_to_numpy(load_json(stats_path)) if stats_path.exists() else {}

    parquet_files = sorted((dataset_root / DATA_DIR).glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_root / DATA_DIR}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    hf_dataset = {column: df[column].to_list() for column in df.columns}
    meta = SimpleNamespace(root=dataset_root, features=info["features"], stats=stats)
    return SimpleNamespace(root=dataset_root, meta=meta, hf_dataset=hf_dataset)


def _compute_numeric_stats(dataset, aggregate_stats, compute_episode_stats) -> dict:
    features = dataset.meta.features
    numeric_features = {
        key: value
        for key, value in features.items()
        if value["dtype"] not in ["image", "video", "string", "language"] and key not in _META_KEYS
    }
    features_to_compute = {key: value for key, value in numeric_features.items() if key != ACTION}

    parquet_files = sorted((dataset.root / DATA_DIR).glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset.root / DATA_DIR}")

    pd = import_pandas_with_retries()
    all_episode_stats = []
    numeric_keys = list(features_to_compute)
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]
            episode_data = {}
            for key in numeric_keys:
                if key not in ep_df.columns:
                    continue
                values = ep_df[key].values
                if len(values) > 0 and hasattr(values[0], "__len__"):
                    episode_data[key] = np.stack(values)
                else:
                    episode_data[key] = np.array(values)
            if episode_data:
                all_episode_stats.append(compute_episode_stats(episode_data, features_to_compute))

    stats = aggregate_stats(all_episode_stats) if all_episode_stats else {}
    if dataset.meta.stats:
        for key, value in dataset.meta.stats.items():
            if key not in stats and key != ACTION:
                stats[key] = value
    return stats


def _load_backup_stats(dataset_root: Path, cast_stats_to_numpy, load_json) -> dict:
    meta_dir = dataset_root / "meta"
    backup_paths = []
    for pattern in _BACKUP_STATS_PATTERNS:
        backup_paths.extend(meta_dir.glob(pattern))
    backup_paths = sorted(set(backup_paths))

    backup_stats = {}
    for path in backup_paths:
        if path.name == "stats.json":
            continue
        loaded = cast_stats_to_numpy(load_json(path))
        for key, value in loaded.items():
            backup_stats.setdefault(key, value)
    return backup_stats


def _merge_preserved_image_video_stats(dataset, stats: dict, cast_stats_to_numpy, load_json) -> dict:
    """Keep image/video stats like official recompute_stats(skip_image_video=True)."""
    preserved_sources = []
    source_stats = []
    if dataset.meta.stats:
        source_stats.append(("current stats.json", dataset.meta.stats))
    backup_stats = _load_backup_stats(dataset.root, cast_stats_to_numpy, load_json)
    if backup_stats:
        source_stats.append(("backup stats", backup_stats))

    for key, feature in dataset.meta.features.items():
        if feature["dtype"] not in ["image", "video"]:
            continue
        if key in stats:
            continue
        for source_name, source in source_stats:
            if key in source:
                stats[key] = source[key]
                preserved_sources.append(f"{key} <- {source_name}")
                break

    if preserved_sources:
        logging.info("Preserved image/video stats: %s", ", ".join(preserved_sources))
    else:
        visual_keys = [key for key, feature in dataset.meta.features.items() if feature["dtype"] in ["image", "video"]]
        if visual_keys:
            logging.warning(
                "Dataset has image/video features but no matching stats were found to preserve: %s",
                visual_keys,
            )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local LeRobot v3 dataset root containing meta/info.json and data/**/*.parquet.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/ee_local_se3_dataset",
        help="Repo id used only to instantiate the local LeRobotDataset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="PI0.5 action chunk size. Must match policy.chunk_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker threads for chunk relative-action computation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and print summary without writing meta/stats.json.",
    )
    return parser.parse_args()


def recompute_ee_local_se3_relative_stats(
    dataset_root: Path,
    repo_id: str,
    chunk_size: int,
    num_workers: int,
    dry_run: bool = False,
) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {dataset_root / 'meta' / 'info.json'}")

    (
        aggregate_stats,
        compute_ee_local_se3_relative_action_stats,
        compute_episode_stats,
        cast_stats_to_numpy,
        load_json,
        write_stats,
    ) = import_stats_dependencies()
    dataset = _load_local_dataset_view(dataset_root, cast_stats_to_numpy, load_json)
    action_shape = dataset.meta.features.get(ACTION, {}).get("shape")
    state_shape = dataset.meta.features.get(OBS_STATE, {}).get("shape")
    if action_shape != [16] and action_shape != (16,):
        raise ValueError(f"Expected 16-D action feature for EE-local SE(3) stats, got {action_shape}")
    if state_shape != [16] and state_shape != (16,):
        raise ValueError(f"Expected 16-D observation.state feature for EE-local SE(3) stats, got {state_shape}")

    stats = _compute_numeric_stats(dataset, aggregate_stats, compute_episode_stats)
    action_stats = compute_ee_local_se3_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=dataset.meta.features,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    stats[ACTION] = action_stats
    stats = _merge_preserved_image_video_stats(dataset, stats, cast_stats_to_numpy, load_json)

    if not dry_run:
        write_stats(stats, dataset_root)
        dataset.meta.stats = stats

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = recompute_ee_local_se3_relative_stats(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    action_stats = stats[ACTION]
    logging.info(
        "EE-local SE(3) action stats ready: dims=%d mean_abs=%.6f std_mean=%.6f%s",
        len(action_stats["mean"]),
        abs(action_stats["mean"]).mean(),
        action_stats["std"].mean(),
        " (dry-run, not written)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
