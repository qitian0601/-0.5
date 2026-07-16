#!/usr/bin/env python

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_CACHE_ROOT = Path("/tmp/lerobot_ee_local_se3_stats_script_hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT))

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.utils import serialize_dict
from lerobot.utils.io_utils import write_json
from lerobot.processor import to_relative_ee_local_se3_actions
from lerobot.utils.constants import ACTION, OBS_STATE
from scripts.recompute_ee_local_se3_relative_stats import recompute_ee_local_se3_relative_stats


def _make_pose(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pose = np.zeros(16, dtype=np.float32)
    pose[0:3] = rng.normal(loc=0.1 * seed, scale=0.02, size=3)
    pose[3:6] = rng.normal(loc=0.03 * seed, scale=0.01, size=3)
    pose[6] = 0.02 + 0.001 * seed
    pose[7:10] = rng.normal(loc=-0.1 * seed, scale=0.02, size=3)
    pose[10:13] = rng.normal(loc=-0.02 * seed, scale=0.01, size=3)
    pose[13] = 0.03 + 0.001 * seed
    pose[14:16] = rng.normal(loc=0.05 * seed, scale=0.01, size=2)
    return pose


def _create_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id="local/ee_local_se3_stats_test",
        root=root,
        fps=30,
        robot_type="test",
        use_videos=False,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (16,), "names": [f"state_{i}" for i in range(16)]},
            ACTION: {"dtype": "float32", "shape": (16,), "names": [f"action_{i}" for i in range(16)]},
        },
    )
    for ep_idx in range(2):
        for frame_idx in range(4):
            state = _make_pose(seed=ep_idx * 10 + frame_idx + 1)
            action = _make_pose(seed=ep_idx * 10 + frame_idx + 2)
            dataset.add_frame(
                {
                    OBS_STATE: state,
                    ACTION: action,
                    "task": "fold towel",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_recompute_stats_rebuilds_state_and_ee_local_se3_relative_action_stats() -> None:
    root = Path("/tmp/lerobot_ee_local_se3_stats_script_test")
    try:
        _create_dataset(root)
        write_stats({ACTION: load_stats(root)[ACTION]}, root)

        stats = recompute_ee_local_se3_relative_stats(
            dataset_root=root,
            repo_id="local/ee_local_se3_stats_test",
            chunk_size=3,
            num_workers=0,
            dry_run=False,
        )

        assert OBS_STATE in stats
        assert ACTION in stats
        assert stats[OBS_STATE]["mean"].shape == (16,)
        assert stats[ACTION]["mean"].shape == (16,)

        dataset = LeRobotDataset(
            repo_id="local/ee_local_se3_stats_test",
            root=root,
            download_videos=False,
        )
        first_state = torch.as_tensor(np.asarray(dataset.hf_dataset[OBS_STATE], dtype=np.float32)[0])
        first_action = torch.as_tensor(np.asarray(dataset.hf_dataset[ACTION], dtype=np.float32)[0])
        first_relative = to_relative_ee_local_se3_actions(first_action, first_state).numpy()

        np.testing.assert_allclose(first_relative[[6, 13]], first_action.numpy()[[6, 13]], atol=1e-6)
        np.testing.assert_allclose(
            first_relative[14:16],
            first_action.numpy()[14:16] - first_state.numpy()[14:16],
            atol=1e-6,
        )
    finally:
        if root.exists():
            shutil.rmtree(root)
        if _CACHE_ROOT.exists():
            shutil.rmtree(_CACHE_ROOT)


def test_dry_run_does_not_write_stats() -> None:
    root = Path("/tmp/lerobot_ee_local_se3_stats_dry_run_script_test")
    try:
        _create_dataset(root)
        original_stats = {ACTION: load_stats(root)[ACTION]}
        write_stats(original_stats, root)

        recompute_ee_local_se3_relative_stats(
            dataset_root=root,
            repo_id="local/ee_local_se3_stats_test",
            chunk_size=3,
            num_workers=0,
            dry_run=True,
        )

        after_stats = load_stats(root)
        assert set(after_stats) == {ACTION}
        np.testing.assert_allclose(after_stats[ACTION]["mean"], original_stats[ACTION]["mean"])
    finally:
        if root.exists():
            shutil.rmtree(root)
        if _CACHE_ROOT.exists():
            shutil.rmtree(_CACHE_ROOT)


def test_recompute_stats_preserves_visual_stats_from_backup_when_current_stats_lack_them() -> None:
    root = Path("/tmp/lerobot_ee_local_se3_stats_visual_backup_script_test")
    try:
        _create_dataset(root)
        current_stats = load_stats(root)
        write_stats({ACTION: current_stats[ACTION]}, root)

        visual_key = "observation.images.front"
        visual_stats = {
            "mean": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "std": np.array([0.01, 0.02, 0.03], dtype=np.float32),
            "min": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "max": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }
        write_json(
            serialize_dict({visual_key: visual_stats}),
            root / "meta" / "stats_14d_euler_base_before_camera_16d.json",
        )

        stats = recompute_ee_local_se3_relative_stats(
            dataset_root=root,
            repo_id="local/ee_local_se3_stats_test",
            chunk_size=3,
            num_workers=0,
            dry_run=False,
        )

        assert visual_key in stats
        after_stats = load_stats(root)
        np.testing.assert_allclose(after_stats[visual_key]["mean"], visual_stats["mean"])
        np.testing.assert_allclose(after_stats[visual_key]["std"], visual_stats["std"])
    finally:
        if root.exists():
            shutil.rmtree(root)
        if _CACHE_ROOT.exists():
            shutil.rmtree(_CACHE_ROOT)


if __name__ == "__main__":
    try:
        test_recompute_stats_rebuilds_state_and_ee_local_se3_relative_action_stats()
        test_dry_run_does_not_write_stats()
        test_recompute_stats_preserves_visual_stats_from_backup_when_current_stats_lack_them()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
