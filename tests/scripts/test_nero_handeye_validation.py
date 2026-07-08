from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_robot_nero.handeye_validation import (
    AprilTagGridConfig,
    HandEyeSample,
    compose_base_board,
    summarize_samples,
    write_reports,
)


def test_apriltag_grid_config_defaults_match_user_board():
    cfg = AprilTagGridConfig()

    assert cfg.family == "tag36h11"
    assert cfg.rows == 6
    assert cfg.cols == 6
    assert cfg.tag_size_m == pytest.approx(0.033)
    assert cfg.tag_spacing_m == pytest.approx(0.0099)
    assert cfg.tag_pitch_m == pytest.approx(0.0429)
    assert cfg.ids == list(range(36))


def test_compose_base_board_multiplies_transforms():
    t_base_camera = np.eye(4)
    t_base_camera[:3, 3] = [0.1, -0.2, 0.3]
    t_camera_board = np.eye(4)
    t_camera_board[:3, 3] = [0.4, 0.5, 0.6]

    result = compose_base_board(t_base_camera, t_camera_board)

    np.testing.assert_allclose(result[:3, 3], [0.5, 0.3, 0.9])
    np.testing.assert_allclose(result[:3, :3], np.eye(3))


def test_summarize_samples_reports_translation_and_rotation_deviation():
    base = np.eye(4)
    moved = np.eye(4)
    moved[:3, 3] = [0.001, 0.002, 0.002]
    moved[:3, :3] = Rotation.from_euler("z", 1.0, degrees=True).as_matrix()
    samples = [
        HandEyeSample("right_wrist", 0, 12, base, 0.2),
        HandEyeSample("right_wrist", 1, 12, moved, 0.3),
    ]

    summary = summarize_samples(samples)

    assert summary["camera"] == "right_wrist"
    assert summary["num_samples"] == 2
    assert summary["translation_rms_mm"] > 0.0
    assert summary["translation_max_mm"] == pytest.approx(1.5)
    assert summary["rotation_max_deg"] == pytest.approx(0.5, abs=1e-6)
    assert summary["mean_reprojection_error_px"] == pytest.approx(0.25)


def test_write_reports_creates_csv_json_and_markdown(tmp_path: Path):
    sample = HandEyeSample("left_wrist", 0, 20, np.eye(4), 0.1)
    summary = summarize_samples([sample])

    write_reports(tmp_path, [sample], {"left_wrist": summary})

    assert (tmp_path / "samples.csv").exists()
    assert (tmp_path / "summary.json").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "left_wrist" in report
    assert "translation_rms_mm" in report


def test_validate_nero_handeye_script_parser_defaults():
    from scripts.validate_nero_handeye_apriltag import build_parser

    args = build_parser().parse_args([])

    assert args.cameras == ["right_wrist", "left_wrist"]
    assert args.samples == 15
    assert args.min_tags == 6
    assert args.right_can == "nero_right"
    assert args.left_can == "nero_left"


def test_validate_nero_handeye_script_builds_grid_board():
    from scripts.validate_nero_handeye_apriltag import _make_board, _require_aruco

    aruco = _require_aruco()
    board = _make_board(aruco, AprilTagGridConfig())

    assert len(board.getObjPoints()) == 36
    assert board.getIds().flatten()[:6].tolist() == list(range(6))
