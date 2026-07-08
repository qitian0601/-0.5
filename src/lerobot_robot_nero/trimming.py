from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config_nero import NeroTrimConfig
from .mapping import namespaced_action_names


def _action_vector(frame: dict, *, arm: str) -> np.ndarray:
    action = frame["action"]
    return np.asarray([float(action[name]) for name in namespaced_action_names(arm)], dtype=float)


def _movement_mask(
    frames: Sequence[dict],
    *,
    arm: str,
    joint_threshold_rad: float,
    gripper_threshold_m: float,
) -> np.ndarray:
    if len(frames) <= 1:
        return np.zeros(len(frames), dtype=bool)

    values = np.stack([_action_vector(frame, arm=arm) for frame in frames])
    diff = np.abs(np.diff(values, axis=0))
    thresholds = np.array([joint_threshold_rad] * 7 + [gripper_threshold_m], dtype=float)
    moved_between_frames = np.any(diff > thresholds, axis=1)

    mask = np.zeros(len(frames), dtype=bool)
    mask[1:] |= moved_between_frames
    mask[:-1] |= moved_between_frames
    return mask


def _movement_mask_multi_arm(
    frames: Sequence[dict],
    *,
    arms: tuple[str, ...],
    joint_threshold_rad: float,
    gripper_threshold_m: float,
) -> np.ndarray:
    if len(frames) <= 1:
        return np.zeros(len(frames), dtype=bool)

    values = np.stack(
        [
            np.concatenate([_action_vector(frame, arm=arm) for arm in arms])
            for frame in frames
        ]
    )
    diff = np.abs(np.diff(values, axis=0))
    thresholds = np.tile(np.array([joint_threshold_rad] * 7 + [gripper_threshold_m], dtype=float), len(arms))
    moved_between_frames = np.any(diff > thresholds, axis=1)

    mask = np.zeros(len(frames), dtype=bool)
    mask[1:] |= moved_between_frames
    mask[:-1] |= moved_between_frames
    return mask


def _trim_indices_from_movement_mask(moved: np.ndarray, *, fps: int, config: NeroTrimConfig) -> tuple[int, int] | None:
    movement_indices = np.flatnonzero(moved)
    if movement_indices.size == 0:
        return None

    static_frames = max(int(round(config.static_time_s * fps)), 0)
    preroll_frames = max(int(round(config.preroll_s * fps)), 0)
    postroll_frames = max(int(round(config.postroll_s * fps)), 0)

    first_movement = int(movement_indices[0])
    last_movement = int(movement_indices[-1]) + 1

    start = 0 if first_movement < static_frames else max(0, first_movement - preroll_frames)
    end = len(moved) if len(moved) - last_movement < static_frames else min(len(moved), last_movement + postroll_frames)

    if end - start < config.min_episode_frames:
        return None
    return start, end


def trim_static_head_tail_indices_from_action_array(
    action_values: np.ndarray, *, fps: int, config: NeroTrimConfig
) -> tuple[int, int] | None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    values = np.asarray(action_values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"action_values must be a 2D array, got shape {values.shape}.")
    if values.shape[1] % 8 != 0:
        raise ValueError(
            "action_values width must be a multiple of 8 "
            f"(7 joints + 1 gripper per arm), got {values.shape[1]}."
        )
    if len(values) <= 1:
        return None

    arms = values.shape[1] // 8
    diff = np.abs(np.diff(values, axis=0))
    thresholds = np.tile(np.array([config.joint_threshold_rad] * 7 + [config.gripper_threshold_m]), arms)
    moved_between_frames = np.any(diff > thresholds, axis=1)

    moved = np.zeros(len(values), dtype=bool)
    moved[1:] |= moved_between_frames
    moved[:-1] |= moved_between_frames
    return _trim_indices_from_movement_mask(moved, fps=fps, config=config)


def trim_static_head_tail(frames: Sequence[dict], *, fps: int, arm: str, config: NeroTrimConfig) -> list[dict]:
    if not frames:
        return []
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    moved = _movement_mask(
        frames,
        arm=arm,
        joint_threshold_rad=config.joint_threshold_rad,
        gripper_threshold_m=config.gripper_threshold_m,
    )
    trim_indices = _trim_indices_from_movement_mask(moved, fps=fps, config=config)
    if trim_indices is None:
        return []
    start, end = trim_indices
    trimmed = list(frames[start:end])
    return trimmed


def trim_static_head_tail_multi_arm(
    frames: Sequence[dict], *, fps: int, arms: tuple[str, ...], config: NeroTrimConfig
) -> list[dict]:
    if not frames:
        return []
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if not arms:
        raise ValueError("arms must not be empty.")

    moved = _movement_mask_multi_arm(
        frames,
        arms=arms,
        joint_threshold_rad=config.joint_threshold_rad,
        gripper_threshold_m=config.gripper_threshold_m,
    )
    trim_indices = _trim_indices_from_movement_mask(moved, fps=fps, config=config)
    if trim_indices is None:
        return []
    start, end = trim_indices
    trimmed = list(frames[start:end])
    return trimmed


def has_movement(frames: Sequence[dict], *, arm: str, config: NeroTrimConfig) -> bool:
    return bool(
        _movement_mask(
            frames,
            arm=arm,
            joint_threshold_rad=config.joint_threshold_rad,
            gripper_threshold_m=config.gripper_threshold_m,
        ).any()
    )
