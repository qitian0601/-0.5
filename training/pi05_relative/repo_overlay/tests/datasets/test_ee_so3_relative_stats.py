#!/usr/bin/env python

import numpy as np
import torch

from lerobot.datasets.compute_stats import (
    compute_ee_local_se3_relative_action_stats,
    compute_ee_so3_relative_action_stats,
)
from lerobot.processor.ee_so3_relative_action_processor import (
    to_relative_ee_local_se3_actions,
    to_relative_ee_actions,
)


class _ColumnDataset:
    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, key):
        return self._columns[key]


def test_compute_ee_so3_relative_action_stats_uses_chunk_start_state():
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.03, 2.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.04],
            [2.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.05, 3.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.06],
        ],
        dtype=np.float32,
    )
    actions = states.copy()
    actions[:, 0] += np.array([0.5, 0.75, 1.0], dtype=np.float32)
    actions[:, 3:6] += np.array(
        [[0.0, 0.0, 0.1], [0.0, 0.2, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32
    )

    dataset = _ColumnDataset(
        {
            "action": [row for row in actions],
            "observation.state": [row for row in states],
            "episode_index": np.array([0, 0, 0], dtype=np.int64),
        }
    )
    stats = compute_ee_so3_relative_action_stats(
        hf_dataset=dataset,
        features={"action": {"shape": [14]}},
        chunk_size=2,
    )

    expected_chunks = []
    for start in [0, 1]:
        expected_chunks.append(
            to_relative_ee_actions(
                torch.from_numpy(actions[start : start + 2]).unsqueeze(0),
                torch.from_numpy(states[start]).unsqueeze(0),
            )
            .squeeze(0)
            .numpy()
        )
    expected = np.concatenate(expected_chunks, axis=0)

    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats["min"], expected.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats["max"], expected.max(axis=0), atol=1e-5)


def test_compute_ee_local_se3_relative_action_stats_uses_chunk_start_state_and_base_delta():
    states = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                np.pi / 2,
                0.01,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.02,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                np.pi / 2,
                0.03,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.04,
                0.5,
                -0.25,
            ],
            [
                0.0,
                2.0,
                0.0,
                0.0,
                0.3,
                0.0,
                0.05,
                3.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.06,
                0.75,
                -0.5,
            ],
        ],
        dtype=np.float32,
    )
    actions = states.copy()
    actions[:, 1] += np.array([0.5, 0.75, 1.0], dtype=np.float32)
    actions[:, 14] += np.array([0.1, 0.2, 0.3], dtype=np.float32)
    actions[:, 15] += np.array([-0.1, -0.2, -0.3], dtype=np.float32)

    dataset = _ColumnDataset(
        {
            "action": [row for row in actions],
            "observation.state": [row for row in states],
            "episode_index": np.array([0, 0, 0], dtype=np.int64),
        }
    )
    stats = compute_ee_local_se3_relative_action_stats(
        hf_dataset=dataset,
        features={"action": {"shape": [16]}},
        chunk_size=2,
    )

    expected_chunks = []
    for start in [0, 1]:
        expected_chunks.append(
            to_relative_ee_local_se3_actions(
                torch.from_numpy(actions[start : start + 2]).unsqueeze(0),
                torch.from_numpy(states[start]).unsqueeze(0),
            )
            .squeeze(0)
            .numpy()
        )
    expected = np.concatenate(expected_chunks, axis=0)

    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats["min"], expected.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats["max"], expected.max(axis=0), atol=1e-5)


if __name__ == "__main__":
    test_compute_ee_so3_relative_action_stats_uses_chunk_start_state()
    test_compute_ee_local_se3_relative_action_stats_uses_chunk_start_state_and_base_delta()
