from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AprilTagGridConfig:
    family: str = "tag36h11"
    rows: int = 6
    cols: int = 6
    tag_size_m: float = 0.033
    tag_spacing_m: float = 0.0099
    start_id: int = 0

    @property
    def tag_pitch_m(self) -> float:
        return self.tag_size_m + self.tag_spacing_m

    @property
    def ids(self) -> list[int]:
        return list(range(self.start_id, self.start_id + self.rows * self.cols))


@dataclass(frozen=True)
class HandEyeSample:
    camera: str
    sample_index: int
    detected_tags: int
    t_base_board: np.ndarray
    reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.t_base_board, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"t_base_board must have shape (4, 4), got {matrix.shape}.")
        object.__setattr__(self, "t_base_board", matrix)


def compose_base_board(t_base_camera: Any, t_camera_board: Any) -> np.ndarray:
    base_camera = np.asarray(t_base_camera, dtype=float)
    camera_board = np.asarray(t_camera_board, dtype=float)
    if base_camera.shape != (4, 4):
        raise ValueError(f"t_base_camera must have shape (4, 4), got {base_camera.shape}.")
    if camera_board.shape != (4, 4):
        raise ValueError(f"t_camera_board must have shape (4, 4), got {camera_board.shape}.")
    return base_camera @ camera_board


def _rotation_mean(rotations: list[Rotation]) -> Rotation:
    quats = np.asarray([rotation.as_quat() for rotation in rotations], dtype=float)
    reference = quats[0]
    for idx, quat in enumerate(quats):
        if float(np.dot(reference, quat)) < 0.0:
            quats[idx] = -quat
    mean_quat = np.mean(quats, axis=0)
    return Rotation.from_quat(mean_quat / np.linalg.norm(mean_quat))


def summarize_samples(samples: list[HandEyeSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot summarize an empty sample list.")
    cameras = {sample.camera for sample in samples}
    if len(cameras) != 1:
        raise ValueError(f"summarize_samples expects one camera, got {sorted(cameras)}.")

    translations = np.asarray([sample.t_base_board[:3, 3] for sample in samples], dtype=float)
    mean_translation = np.mean(translations, axis=0)
    translation_deviation_m = np.linalg.norm(translations - mean_translation, axis=1)
    translation_deviation_mm = translation_deviation_m * 1000.0

    rotations = [Rotation.from_matrix(sample.t_base_board[:3, :3]) for sample in samples]
    mean_rotation = _rotation_mean(rotations)
    rotation_deviation_deg = np.asarray(
        [(mean_rotation.inv() * rotation).magnitude() * 180.0 / np.pi for rotation in rotations],
        dtype=float,
    )

    reprojection_errors = [
        float(sample.reprojection_error_px) for sample in samples if sample.reprojection_error_px is not None
    ]
    return {
        "camera": samples[0].camera,
        "num_samples": len(samples),
        "mean_detected_tags": float(np.mean([sample.detected_tags for sample in samples])),
        "mean_translation_m": mean_translation.tolist(),
        "translation_std_mm": np.std(translations, axis=0).tolist(),
        "translation_rms_mm": float(np.sqrt(np.mean(np.square(translation_deviation_mm)))),
        "translation_max_mm": float(np.max(translation_deviation_mm)),
        "rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation_deviation_deg)))),
        "rotation_max_deg": float(np.max(rotation_deviation_deg)),
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)) if reprojection_errors else None,
    }


def _sample_row(sample: HandEyeSample) -> dict[str, Any]:
    translation = sample.t_base_board[:3, 3]
    euler_xyz = Rotation.from_matrix(sample.t_base_board[:3, :3]).as_euler("xyz", degrees=True)
    return {
        "camera": sample.camera,
        "sample_index": sample.sample_index,
        "detected_tags": sample.detected_tags,
        "x_m": float(translation[0]),
        "y_m": float(translation[1]),
        "z_m": float(translation[2]),
        "roll_deg": float(euler_xyz[0]),
        "pitch_deg": float(euler_xyz[1]),
        "yaw_deg": float(euler_xyz[2]),
        "reprojection_error_px": sample.reprojection_error_px,
    }


def write_reports(
    output_dir: str | Path,
    samples: list[HandEyeSample],
    summaries: dict[str, dict[str, Any]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = [_sample_row(sample) for sample in samples]
    csv_path = output_path / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = (
            list(rows[0].keys())
            if rows
            else [
                "camera",
                "sample_index",
                "detected_tags",
                "x_m",
                "y_m",
                "z_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "reprojection_error_px",
            ]
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, sort_keys=True)

    lines = ["# Nero Hand-Eye Validation Report", ""]
    for camera, summary in summaries.items():
        lines.extend(
            [
                f"## {camera}",
                "",
                f"- num_samples: {summary['num_samples']}",
                f"- mean_detected_tags: {summary['mean_detected_tags']:.2f}",
                f"- translation_rms_mm: {summary['translation_rms_mm']:.3f}",
                f"- translation_max_mm: {summary['translation_max_mm']:.3f}",
                f"- rotation_rms_deg: {summary['rotation_rms_deg']:.3f}",
                f"- rotation_max_deg: {summary['rotation_max_deg']:.3f}",
                f"- mean_reprojection_error_px: {summary['mean_reprojection_error_px']}",
                "",
            ]
        )
    (output_path / "report.md").write_text("\n".join(lines), encoding="utf-8")
