from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot_robot_nero.config_nero import NeroDualRobotConfig
from lerobot_robot_nero.ee_local_se3_adapter import SE3Transform
from lerobot_robot_nero.handeye_validation import (
    AprilTagGridConfig,
    HandEyeSample,
    compose_base_board,
    summarize_samples,
    write_reports,
)
from lerobot_robot_nero.robot_nero_dual import NeroDualRobot


ROOT = Path("/home/chenglong/workplace/nero_teleop_ws")
LEROBOT_ROOT = ROOT / "lerobot"
DEFAULT_RIGHT_HANDEYE = ROOT / "data/lerobot/pickplace/handeye_right_arm_tsai.yml"
DEFAULT_LEFT_HANDEYE = ROOT / "data/lerobot/pickplace/handeye_left_arm_tsai.yml"
CAMERA_DEFAULTS = {
    "right_wrist": {
        "arm": "right",
        "serial": "244222070153",
        "width": 640,
        "height": 480,
        "fps": 30,
        "handeye": DEFAULT_RIGHT_HANDEYE,
    },
    "left_wrist": {
        "arm": "left",
        "serial": "244222077114",
        "width": 640,
        "height": 480,
        "fps": 30,
        "handeye": DEFAULT_LEFT_HANDEYE,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Nero wrist-camera hand-eye calibration with a fixed AprilTag grid."
    )
    parser.add_argument("--cameras", nargs="+", default=["right_wrist", "left_wrist"], choices=sorted(CAMERA_DEFAULTS))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--min-tags", type=int, default=6)
    parser.add_argument("--right-can", default=os.environ.get("NERO_RIGHT_CAN", "nero_right"))
    parser.add_argument("--left-can", default=os.environ.get("NERO_LEFT_CAN", "nero_left"))
    parser.add_argument("--right-handeye", default=str(DEFAULT_RIGHT_HANDEYE))
    parser.add_argument("--left-handeye", default=str(DEFAULT_LEFT_HANDEYE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag-size-m", type=float, default=0.033)
    parser.add_argument("--tag-spacing-m", type=float, default=0.0099)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--warmup-s", type=int, default=3)
    return parser


def _require_aruco() -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build has no cv2.aruco module. Install opencv-contrib-python.")
    required = ["DICT_APRILTAG_36h11", "ArucoDetector", "DetectorParameters", "GridBoard"]
    missing = [name for name in required if not hasattr(cv2.aruco, name)]
    if missing:
        raise RuntimeError(f"This OpenCV ArUco build is missing: {missing}")
    return cv2.aruco


def _make_board(aruco: Any, cfg: AprilTagGridConfig) -> Any:
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    ids = np.asarray(cfg.ids, dtype=np.int32)
    try:
        return aruco.GridBoard(
            (cfg.cols, cfg.rows),
            cfg.tag_size_m,
            cfg.tag_spacing_m,
            dictionary,
            ids,
        )
    except TypeError:
        return aruco.GridBoard_create(
            cfg.cols,
            cfg.rows,
            cfg.tag_size_m,
            cfg.tag_spacing_m,
            dictionary,
            ids,
        )


def _camera_matrix_from_realsense(cam: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(cam, "rs_profile", None) is None:
        raise RuntimeError("RealSense camera has no active rs_profile; connect it before reading intrinsics.")
    import pyrealsense2 as rs

    stream = cam.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    camera_matrix = np.asarray(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


def _make_robot_config(args: argparse.Namespace) -> NeroDualRobotConfig:
    cameras = {}
    for name in args.cameras:
        defaults = CAMERA_DEFAULTS[name]
        cameras[name] = RealSenseCameraConfig(
            serial_number_or_name=defaults["serial"],
            width=defaults["width"],
            height=defaults["height"],
            fps=defaults["fps"],
            warmup_s=args.warmup_s,
        )
    cfg = NeroDualRobotConfig(cameras=cameras)
    cfg.right.connection.channel = args.right_can
    cfg.left.connection.channel = args.left_can
    cfg.right.connection.reset_on_connect = False
    cfg.left.connection.reset_on_connect = False
    return cfg


def _handeye_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "right_wrist": Path(args.right_handeye),
        "left_wrist": Path(args.left_handeye),
    }


def _load_handeye(args: argparse.Namespace) -> dict[str, np.ndarray]:
    transforms = {}
    for camera in args.cameras:
        path = _handeye_paths(args)[camera]
        transforms[camera] = SE3Transform.from_opencv_yaml(path, key="T_base_cam").as_matrix()
    return transforms


def _board_object_points(board: Any) -> dict[int, np.ndarray]:
    obj_points = board.getObjPoints()
    ids = board.getIds().flatten()
    return {int(tag_id): np.asarray(points, dtype=np.float32).reshape(4, 3) for tag_id, points in zip(ids, obj_points)}


def _reprojection_error(
    board: Any,
    corners: list[np.ndarray],
    ids: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    object_points_by_id = _board_object_points(board)
    errors = []
    for tag_corners, tag_id_array in zip(corners, ids):
        tag_id = int(tag_id_array[0])
        object_points = object_points_by_id.get(tag_id)
        if object_points is None:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        observed = np.asarray(tag_corners, dtype=np.float32).reshape(4, 2)
        errors.extend(np.linalg.norm(projected.reshape(4, 2) - observed, axis=1).tolist())
    if not errors:
        return None
    return float(np.mean(errors))


def _estimate_pose(
    aruco: Any,
    board: Any,
    image_rgb: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray, int, float | None, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    debug = image_rgb.copy()
    if ids is None or len(ids) == 0:
        return False, np.eye(4), 0, None, debug

    aruco.drawDetectedMarkers(debug, corners, ids)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    ok, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
    if int(ok) <= 0:
        return False, np.eye(4), len(ids), None, debug

    rotation, _ = cv2.Rodrigues(rvec)
    t_camera_board = np.eye(4, dtype=float)
    t_camera_board[:3, :3] = rotation
    t_camera_board[:3, 3] = tvec.reshape(3)
    cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    reprojection_error = _reprojection_error(board, corners, ids, rvec, tvec, camera_matrix, dist_coeffs)
    return True, t_camera_board, len(ids), reprojection_error, debug


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LEROBOT_ROOT / "outputs/handeye_validation" / timestamp


def _save_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> Path:
    aruco = _require_aruco()
    board_cfg = AprilTagGridConfig(
        rows=args.rows,
        cols=args.cols,
        tag_size_m=args.tag_size_m,
        tag_spacing_m=args.tag_spacing_m,
        start_id=args.start_id,
    )
    board = _make_board(aruco, board_cfg)
    handeye = _load_handeye(args)
    output_dir = _output_dir(args)
    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(output_dir, args)

    robot = NeroDualRobot(_make_robot_config(args))
    samples: list[HandEyeSample] = []
    per_camera_counts = defaultdict(int)

    print(f"Output directory: {output_dir}")
    print("Connecting Nero arms and requested wrist cameras...")
    robot.connect()
    try:
        intrinsics = {name: _camera_matrix_from_realsense(robot.cameras[name]) for name in args.cameras}
        print("Move the arm(s) so the fixed AprilTag board is visible. Press Enter to sample, or q then Enter to finish.")
        while min([per_camera_counts[name] for name in args.cameras] or [0]) < args.samples:
            command = input("Capture sample? [Enter/q] ").strip().lower()
            if command == "q":
                break
            observation = robot.get_flange_observation()
            for camera in args.cameras:
                image = observation[camera]
                camera_matrix, dist_coeffs = intrinsics[camera]
                ok, t_camera_board, detected_tags, reproj_error, debug = _estimate_pose(
                    aruco,
                    board,
                    image,
                    camera_matrix,
                    dist_coeffs,
                )
                print(f"{camera}: detected_tags={detected_tags}, reprojection_error_px={reproj_error}")
                if not ok or detected_tags < args.min_tags:
                    print(f"{camera}: skipped; need at least {args.min_tags} detected tags.")
                    continue
                sample_index = per_camera_counts[camera]
                t_base_board = compose_base_board(handeye[camera], t_camera_board)
                samples.append(HandEyeSample(camera, sample_index, detected_tags, t_base_board, reproj_error))
                per_camera_counts[camera] += 1
                debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"{camera}_{sample_index:03d}.png"), debug_bgr)
    finally:
        robot.disconnect()

    summaries = {}
    for camera in args.cameras:
        camera_samples = [sample for sample in samples if sample.camera == camera]
        if camera_samples:
            summaries[camera] = summarize_samples(camera_samples)
    write_reports(output_dir, samples, summaries)
    print(f"Wrote report: {output_dir / 'report.md'}")
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
