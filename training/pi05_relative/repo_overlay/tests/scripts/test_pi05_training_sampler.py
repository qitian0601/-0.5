#!/usr/bin/env python

from __future__ import annotations

import sys
from types import SimpleNamespace

from lerobot.scripts.lerobot_train import _make_training_sampler


class _Episodes:
    def __getitem__(self, key: str) -> list[int]:
        values = {
            "dataset_from_index": [0, 5],
            "dataset_to_index": [5, 10],
        }
        return values[key]


class _Dataset:
    episodes = None
    meta = SimpleNamespace(episodes=_Episodes())


def test_non_streaming_pi05_uses_episode_sampler() -> None:
    shuffle, sampler = _make_training_sampler(
        active_cfg=SimpleNamespace(drop_n_last_frames=2),
        dataset=_Dataset(),
        streaming=False,
    )

    assert shuffle is False
    assert sampler is not None
    assert list(sampler.indices) == [0, 1, 2, 5, 6, 7]


def test_streaming_pi05_rejects_tail_drop_sampler() -> None:
    try:
        _make_training_sampler(
            active_cfg=SimpleNamespace(drop_n_last_frames=2),
            dataset=_Dataset(),
            streaming=True,
        )
    except ValueError as exc:
        assert "drop_n_last_frames" in str(exc)
        assert "streaming" in str(exc)
    else:
        raise AssertionError("Expected streaming dataset with drop_n_last_frames to fail")


def test_streaming_allows_zero_tail_drop() -> None:
    shuffle, sampler = _make_training_sampler(
        active_cfg=SimpleNamespace(drop_n_last_frames=0),
        dataset=_Dataset(),
        streaming=True,
    )

    assert shuffle is True
    assert sampler is None


if __name__ == "__main__":
    try:
        test_non_streaming_pi05_uses_episode_sampler()
        test_streaming_pi05_rejects_tail_drop_sampler()
        test_streaming_allows_zero_tail_drop()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
