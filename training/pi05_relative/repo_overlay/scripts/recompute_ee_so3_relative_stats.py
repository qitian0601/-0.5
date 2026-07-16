#!/usr/bin/env python

"""Recompute PI0.5 EE/SO(3) relative action stats for a LeRobot dataset.

The dataset parquet remains absolute EE pose:

    observation.state = [xyz, rotvec, gripper] * 2
    action            = [xyz, rotvec, gripper] * 2

This script recomputes numeric dataset stats, then rewrites ``stats["action"]``
so action normalization matches the relative action distribution produced during
training with:

    --policy.use_relative_actions=true
    --policy.relative_action_type=ee_so3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_ee_so3_relative_action_stats,
    compute_episode_stats,
)
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.utils import DATA_DIR
from lerobot.utils.constants import ACTION, ACTION_MASK, OBS_IMAGES_MASK, OBS_STATE

_META_KEYS = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
_NON_STATS_KEYS = {ACTION_MASK, OBS_IMAGES_MASK}


def _compute_numeric_stats(dataset: LeRobotDataset) -> dict:
    features = dataset.meta.features
    numeric_features = {
        key: value
        for key, value in features.items()
        if value["dtype"] not in ["image", "video", "string", "language"]
        and key not in _META_KEYS
        and key not in _NON_STATS_KEYS
    }
    features_to_compute = {key: value for key, value in numeric_features.items() if key != ACTION}

    parquet_files = sorted((dataset.root / DATA_DIR).glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset.root / DATA_DIR}")

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
        default="local/ee_so3_dataset",
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


def recompute_ee_so3_relative_stats(
    dataset_root: Path,
    repo_id: str,
    chunk_size: int,
    num_workers: int,
    dry_run: bool = False,
) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {dataset_root / 'meta' / 'info.json'}")

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=False)
    action_shape = dataset.meta.features.get(ACTION, {}).get("shape")
    state_shape = dataset.meta.features.get(OBS_STATE, {}).get("shape")
    if action_shape != [14] and action_shape != (14,):
        raise ValueError(f"Expected 14-D action feature for EE/SO3 stats, got {action_shape}")
    if state_shape != [14] and state_shape != (14,):
        raise ValueError(f"Expected 14-D observation.state feature for EE/SO3 stats, got {state_shape}")

    stats = _compute_numeric_stats(dataset)
    action_stats = compute_ee_so3_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=dataset.meta.features,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    stats[ACTION] = action_stats

    if not dry_run:
        write_stats(stats, dataset_root)
        dataset.meta.stats = stats

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = recompute_ee_so3_relative_stats(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    action_stats = stats[ACTION]
    logging.info(
        "EE/SO3 action stats ready: dims=%d mean_abs=%.6f std_mean=%.6f%s",
        len(action_stats["mean"]),
        abs(action_stats["mean"]).mean(),
        action_stats["std"].mean(),
        " (dry-run, not written)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
