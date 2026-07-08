import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_nero.replay_calibration_poses import (
    CalibrationPoseTarget,
    EE_FLANGE_EULER_QUATERNION_NAMES,
    CALIBRATION_READY_LEFT_JOINTS,
    CALIBRATION_READY_RIGHT_JOINTS,
    ORIGINAL_READY_LEFT_JOINTS,
    ORIGINAL_READY_RIGHT_JOINTS,
    ReplayCalibrationPosesConfig,
    flange_dict_from_poses,
    load_replay_targets,
    limit_joint_step,
    make_base_flange_features,
    make_policy_frame,
    move_dual_to_ready,
    move_arm_to_pose,
    move_p,
    prepare_between_arms_board_swap,
    prepare_ready_and_grip_board,
    prepare_target_arm_for_replay,
    prepare_template_output_root,
    prepare_grippers_for_board,
    prompt_before_capture,
    send_limited_joint_target,
    smooth_move_p,
    validate_target_counts,
    write_template_capture,
)


class FakeGripper:
    def __init__(self, *, arm: str = "right", events: list[str] | None = None) -> None:
        self.arm = arm
        self.events = events
        self.calls = []

    def move_gripper_m(self, *, value: float, force: float) -> None:
        if self.events is not None:
            self.events.append(f"{self.arm}_gripper_{value:.3f}")
        self.calls.append({"value": value, "force": force})


class FakeSdkRobot:
    def __init__(self, *, arm: str = "right", events: list[str] | None = None) -> None:
        self.arm = arm
        self.events = events
        self.motion_modes = []
        self.moves = []
        self.joint_moves = []

    def set_motion_mode(self, mode: str) -> None:
        self.motion_modes.append(mode)

    def move_p(self, pose: list[float]) -> None:
        self.moves.append(list(pose))

    def move_js(self, joints: list[float]) -> None:
        if self.events is not None:
            rounded = ",".join(f"{value:.3f}" for value in joints)
            self.events.append(f"{self.arm}_move_js:{rounded}")
        self.joint_moves.append(list(joints))


class FakeArmRuntime:
    def __init__(
        self,
        current_pose: list[float],
        *,
        current_joints: list[float] | None = None,
        arm: str = "right",
        events: list[str] | None = None,
    ) -> None:
        self.arm = arm
        self.robot = FakeSdkRobot(arm=arm, events=events)
        self.end_effector = FakeGripper(arm=arm, events=events)
        self.current_pose = np.asarray(current_pose, dtype=float)
        self.current_joints = np.asarray(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if current_joints is None else current_joints,
            dtype=float,
        )

    def read_flange_pose(self) -> np.ndarray:
        return self.current_pose.copy()

    def read_joints(self) -> np.ndarray:
        return self.current_joints.copy()


class FakeArmIK:
    def __init__(self, solution: list[float]) -> None:
        self.solution = np.asarray(solution, dtype=float)
        self.calls = []

    def solve(self, pose, current_joints=None):
        self.calls.append(
            (
                np.asarray(pose, dtype=float),
                None if current_joints is None else np.asarray(current_joints, dtype=float),
            )
        )
        return self.solution.copy()


class FakeDualRobot:
    def __init__(self, *, events: list[str] | None = None) -> None:
        self.right = FakeArmRuntime([0, 0, 0, 0, 0, 0], arm="right", events=events)
        self.left = FakeArmRuntime([0, 0, 0, 0, 0, 0], arm="left", events=events)


def _write_pose(path: Path, arm: str, frame_index: int, pose: list[float]) -> None:
    path.mkdir(parents=True)
    with (path / "pose.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frame_index": frame_index,
                f"{arm}_observation_pose": {
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2],
                    "roll": pose[3],
                    "pitch": pose[4],
                    "yaw": pose[5],
                },
            },
            f,
        )


def test_load_replay_targets_sorts_numeric_prefix_and_plays_right_then_left(tmp_path: Path) -> None:
    right_dir = tmp_path / "right"
    left_dir = tmp_path / "left"
    _write_pose(right_dir / "02_frame_0020", "right", 20, [2, 0, 0, 0, 0, 0])
    _write_pose(right_dir / "01_frame_0010", "right", 10, [1, 0, 0, 0, 0, 0])
    _write_pose(left_dir / "01_frame_0030", "left", 30, [3, 0, 0, 0, 0, 0])

    targets = load_replay_targets(right_dir=right_dir, left_dir=left_dir)

    assert [target.arm for target in targets] == ["right", "right", "left"]
    assert [target.frame_index for target in targets] == [10, 20, 30]
    assert [target.sequence_index for target in targets] == [1, 2, 1]


def test_load_replay_targets_can_use_summary_when_pose_dirs_are_incomplete(tmp_path: Path) -> None:
    right_dir = tmp_path / "right_arm_stable_front_frames_new_002"
    left_dir = tmp_path / "left_arm_stable_front_frames_new_003"
    right_dir.mkdir()
    left_dir.mkdir()
    summary = {
        "right_arm_stable_front_frames_new_002": {
            "items": [
                {
                    "index": 29,
                    "frame_index": 2507,
                    "timestamp_s": 83.56666564941406,
                    "pause_frame_range": {"start": 2500, "end": 2512},
                    "right_observation_pose": {
                        "x": 1,
                        "y": 2,
                        "z": 3,
                        "roll": 4,
                        "pitch": 5,
                        "yaw": 6,
                    },
                }
            ]
        },
        "left_arm_stable_front_frames_new_003": {
            "items": [
                {
                    "index": 12,
                    "frame_index": 878,
                    "left_observation_pose": {
                        "x": 7,
                        "y": 8,
                        "z": 9,
                        "roll": 10,
                        "pitch": 11,
                        "yaw": 12,
                    },
                }
            ]
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    targets = load_replay_targets(right_dir=right_dir, left_dir=left_dir, summary_json=summary_path)

    assert [target.arm for target in targets] == ["right", "left"]
    assert [target.sequence_index for target in targets] == [29, 12]
    assert [target.frame_index for target in targets] == [2507, 878]
    np.testing.assert_allclose(targets[0].pose, [1, 2, 3, 4, 5, 6])
    assert targets[0].timestamp_s == 83.56666564941406
    assert targets[0].pause_frame_range == {"start": 2500, "end": 2512}
    validate_target_counts(targets, expected_right=1, expected_left=1)


def test_default_expected_target_counts_match_deduplicated_replay_summary() -> None:
    cfg = ReplayCalibrationPosesConfig(robot=None)  # type: ignore[arg-type]

    assert cfg.expected_right_targets == 27
    assert cfg.expected_left_targets == 27


def test_prepare_template_output_root_overwrites_existing_directory_by_default(tmp_path: Path) -> None:
    output_root = tmp_path / "replay_actual"
    stale_file = output_root / "stale.txt"
    output_root.mkdir()
    stale_file.write_text("old", encoding="utf-8")

    prepare_template_output_root(output_root, overwrite=True)

    assert output_root.exists()
    assert not stale_file.exists()


def test_write_template_capture_matches_stable_frame_folder_format(tmp_path: Path) -> None:
    output_root = tmp_path / "replay_actual"
    prepare_template_output_root(output_root, overwrite=True)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    actual_observation = flange_dict_from_poses(
        right_pose=[1, 2, 3, 0.1, 0.2, 0.3],
        left_pose=[4, 5, 6, -0.1, -0.2, -0.3],
        right_gripper_width=0.01,
        left_gripper_width=0.02,
    )
    actual_observation["front"] = image
    target = CalibrationPoseTarget(
        arm="right",
        sequence_index=1,
        frame_index=139,
        source_dir=Path("right_arm_stable_front_frames_new_002/01_frame_0139"),
        pose=np.asarray([0, 0, 0, 0, 0, 0], dtype=float),
        timestamp_s=4.633333206176758,
        pause_frame_range={"start": 132, "end": 143},
    )

    capture_dir = write_template_capture(
        output_root=output_root,
        target=target,
        actual_observation=actual_observation,
        front_key="front",
        euler_order="xyz",
    )

    assert capture_dir == output_root / "right_arm_stable_front_frames_actual" / "01_frame_0139"
    assert (capture_dir / "front.png").exists()
    with (capture_dir / "pose.json").open("r", encoding="utf-8") as f:
        pose_json = json.load(f)
    expected_quat = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    assert pose_json["frame_index"] == 139
    assert pose_json["timestamp_s"] == 4.633333206176758
    assert pose_json["pause_frame_range"] == {"start": 132, "end": 143}
    assert pose_json["right_observation_pose"] == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "roll": 0.1,
        "pitch": 0.2,
        "yaw": 0.3,
    }
    np.testing.assert_allclose(
        [
            pose_json["right_observation_pose_quaternion"]["qx"],
            pose_json["right_observation_pose_quaternion"]["qy"],
            pose_json["right_observation_pose_quaternion"]["qz"],
            pose_json["right_observation_pose_quaternion"]["qw"],
        ],
        expected_quat,
    )


def test_make_base_flange_features_front_video_and_20d_state() -> None:
    features = make_base_flange_features(front_shape=(800, 1280, 3), use_videos=True)

    assert features["action"]["shape"] == (20,)
    assert features["action"]["names"] == EE_FLANGE_EULER_QUATERNION_NAMES
    assert features["observation.state"]["shape"] == (20,)
    assert features["observation.state"]["names"] == EE_FLANGE_EULER_QUATERNION_NAMES
    assert features["observation.images.front"]["dtype"] == "video"
    assert features["observation.images.front"]["shape"] == (800, 1280, 3)


def test_make_policy_frame_uses_actual_base_flange_pose_for_action_and_observation() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    actual_observation = flange_dict_from_poses(
        right_pose=[1, 2, 3, 0.1, 0.2, 0.3],
        left_pose=[4, 5, 6, -0.1, -0.2, -0.3],
        right_gripper_width=0.01,
        left_gripper_width=0.02,
    )
    actual_observation["front"] = image
    expected_right_quat = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    expected_left_quat = Rotation.from_euler("xyz", [-0.1, -0.2, -0.3]).as_quat()

    frame = make_policy_frame(
        actual_observation=actual_observation,
        front_key="front",
        task="pickplace",
    )

    expected = np.asarray(
        [
            1,
            2,
            3,
            0.1,
            0.2,
            0.3,
            *expected_right_quat,
            4,
            5,
            6,
            -0.1,
            -0.2,
            -0.3,
            *expected_left_quat,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(frame["action"], expected)
    np.testing.assert_allclose(frame["observation.state"], expected)
    assert frame["observation.images.front"] is image
    assert frame["task"] == "pickplace"


def test_smooth_move_p_sends_intermediate_waypoints_and_target() -> None:
    arm = FakeArmRuntime([0, 0, 0, 0, 0, 0])

    smooth_move_p(
        arm,
        [0.3, 0.0, 0.0, 0.0, 0.0, np.pi / 2],
        enabled=True,
        duration_s=0.2,
        dt_s=0.1,
        min_steps=4,
        euler_order="xyz",
        sleep_fn=lambda _: None,
    )

    assert len(arm.robot.moves) == 4
    np.testing.assert_allclose(arm.robot.moves[0][:3], [0.075, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(arm.robot.moves[-1], [0.3, 0.0, 0.0, 0.0, 0.0, np.pi / 2], atol=1e-7)
    assert arm.robot.motion_modes == ["p"]


def test_smooth_move_p_can_be_disabled_for_single_target_command() -> None:
    arm = FakeArmRuntime([0, 0, 0, 0, 0, 0])

    smooth_move_p(
        arm,
        [0.3, 0.0, 0.0, 0.0, 0.0, np.pi / 2],
        enabled=False,
        duration_s=0.2,
        dt_s=0.1,
        min_steps=4,
        euler_order="xyz",
        sleep_fn=lambda _: None,
    )

    assert arm.robot.moves == [[0.3, 0.0, 0.0, 0.0, 0.0, np.pi / 2]]


def test_limit_joint_step_clips_each_joint_delta() -> None:
    current = np.asarray([0.0, 0.0, 0.0], dtype=float)
    target = np.asarray([0.2, -0.3, 0.01], dtype=float)

    limited = limit_joint_step(current, target, max_step_rad=0.05)

    np.testing.assert_allclose(limited, [0.05, -0.05, 0.01])


def test_send_limited_joint_target_uses_move_js_until_target() -> None:
    arm = FakeArmRuntime([0, 0, 0, 0, 0, 0])

    send_limited_joint_target(
        arm,
        np.asarray([0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.54], dtype=float),
        max_joint_step_rad=0.03,
        dt_s=0.01,
        sleep_fn=lambda _: None,
    )

    assert arm.robot.motion_modes == ["js"]
    assert len(arm.robot.joint_moves) == 2
    np.testing.assert_allclose(arm.robot.joint_moves[0], [0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.57])
    np.testing.assert_allclose(arm.robot.joint_moves[-1], [0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.54])


def test_move_arm_to_pose_with_curobo_backend_solves_from_current_joints_and_sends_move_js() -> None:
    arm = FakeArmRuntime([0, 0, 0, 0, 0, 0])
    ik = FakeArmIK([0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.54])
    target_pose = np.asarray([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], dtype=float)

    move_arm_to_pose(
        arm,
        target_pose,
        motion_backend="curobo_move_js",
        ik=ik,
        smooth_move=True,
        smooth_move_time_s=1.0,
        smooth_move_dt_s=0.05,
        smooth_min_steps=10,
        max_joint_step_rad=0.03,
        joint_command_dt_s=0.01,
        euler_order="xyz",
        sleep_fn=lambda _: None,
    )

    np.testing.assert_allclose(ik.calls[0][0], target_pose)
    np.testing.assert_allclose(ik.calls[0][1], arm.read_joints())
    assert arm.robot.moves == []
    assert len(arm.robot.joint_moves) == 2


def test_move_dual_to_ready_uses_inference_joint_ready_pose_with_smooth_move_js() -> None:
    robot = FakeDualRobot()
    right_ik = FakeArmIK([9, 9, 9, 9, 9, 9, 9])
    left_ik = FakeArmIK([8, 8, 8, 8, 8, 8, 8])
    cfg = ReplayCalibrationPosesConfig(
        robot=None,  # type: ignore[arg-type]
        ready_wait_s=0.0,
        stable_timeout_s=0.0,
        max_joint_step_rad=0.05,
        joint_command_dt_s=0.01,
    )

    move_dual_to_ready(
        robot,  # type: ignore[arg-type]
        cfg,
        ik_by_arm={"right": right_ik, "left": left_ik},
        sleep_fn=lambda _: None,
    )

    assert right_ik.calls == []
    assert left_ik.calls == []
    assert robot.right.robot.moves == []
    assert robot.left.robot.moves == []
    assert robot.right.robot.motion_modes == ["js"]
    assert robot.left.robot.motion_modes == ["js"]
    np.testing.assert_allclose(robot.right.robot.joint_moves[-1], ORIGINAL_READY_RIGHT_JOINTS)
    np.testing.assert_allclose(robot.left.robot.joint_moves[-1], ORIGINAL_READY_LEFT_JOINTS)
    assert len(robot.right.robot.joint_moves) > 1
    assert len(robot.left.robot.joint_moves) > 1


def test_prepare_ready_and_grip_board_opens_moves_original_ready_then_waits_to_close() -> None:
    events: list[str] = []
    robot = FakeDualRobot(events=events)
    cfg = ReplayCalibrationPosesConfig(
        robot=None,  # type: ignore[arg-type]
        ready_wait_s=0.0,
        stable_timeout_s=0.0,
        max_joint_step_rad=10.0,
        joint_command_dt_s=0.01,
    )

    prepare_ready_and_grip_board(
        robot,  # type: ignore[arg-type]
        cfg,
        ik_by_arm=None,
        input_fn=lambda prompt: events.append("prompt_close"),
        sleep_fn=lambda _: None,
    )

    right_open_index = events.index("right_gripper_0.100")
    left_open_index = events.index("left_gripper_0.100")
    first_move_index = min(index for index, event in enumerate(events) if "_move_js:" in event)
    prompt_index = events.index("prompt_close")
    right_close_index = events.index("right_gripper_0.000")
    left_close_index = events.index("left_gripper_0.000")

    assert right_open_index < first_move_index
    assert left_open_index < first_move_index
    assert first_move_index < prompt_index
    assert prompt_index < right_close_index
    assert prompt_index < left_close_index
    np.testing.assert_allclose(robot.right.robot.joint_moves[-1], ORIGINAL_READY_RIGHT_JOINTS)
    np.testing.assert_allclose(robot.left.robot.joint_moves[-1], ORIGINAL_READY_LEFT_JOINTS)


def test_prepare_target_arm_for_replay_moves_only_requested_arm_to_calibration_ready_then_waits() -> None:
    events: list[str] = []
    robot = FakeDualRobot(events=events)
    cfg = ReplayCalibrationPosesConfig(
        robot=None,  # type: ignore[arg-type]
        ready_wait_s=0.0,
        stable_timeout_s=0.0,
        max_joint_step_rad=10.0,
        joint_command_dt_s=0.01,
    )

    prepare_target_arm_for_replay(
        robot,  # type: ignore[arg-type]
        cfg,
        arm="right",
        input_fn=lambda prompt: events.append("prompt_start_right"),
        sleep_fn=lambda _: None,
    )

    assert any(event.startswith("right_move_js:") for event in events)
    assert not any(event.startswith("left_move_js:") for event in events)
    assert events[-1] == "prompt_start_right"
    np.testing.assert_allclose(robot.right.robot.joint_moves[-1], CALIBRATION_READY_RIGHT_JOINTS)
    assert robot.left.robot.joint_moves == []


def test_prepare_between_arms_board_swap_returns_ready_opens_waits_then_closes() -> None:
    events: list[str] = []
    robot = FakeDualRobot(events=events)
    cfg = ReplayCalibrationPosesConfig(
        robot=None,  # type: ignore[arg-type]
        ready_wait_s=0.0,
        stable_timeout_s=0.0,
        max_joint_step_rad=10.0,
        joint_command_dt_s=0.01,
    )

    prepare_between_arms_board_swap(
        robot,  # type: ignore[arg-type]
        cfg,
        ik_by_arm=None,
        input_fn=lambda prompt: events.append("prompt_reposition"),
        sleep_fn=lambda _: None,
    )

    first_move_index = min(index for index, event in enumerate(events) if "_move_js:" in event)
    right_open_index = events.index("right_gripper_0.100")
    left_open_index = events.index("left_gripper_0.100")
    prompt_index = events.index("prompt_reposition")
    right_close_index = events.index("right_gripper_0.000")
    left_close_index = events.index("left_gripper_0.000")

    assert first_move_index < right_open_index
    assert first_move_index < left_open_index
    assert right_open_index < prompt_index
    assert left_open_index < prompt_index
    assert prompt_index < right_close_index
    assert prompt_index < left_close_index
    np.testing.assert_allclose(robot.right.robot.joint_moves[-1], ORIGINAL_READY_RIGHT_JOINTS)
    np.testing.assert_allclose(robot.left.robot.joint_moves[-1], ORIGINAL_READY_LEFT_JOINTS)


def test_prepare_grippers_for_board_opens_waits_then_closes_with_one_newton() -> None:
    robot = FakeDualRobot()
    prompts = []
    sleeps = []

    prepare_grippers_for_board(
        robot,
        enabled=True,
        open_width=0.1,
        close_width=0.0,
        force=1.0,
        settle_s=1.0,
        input_fn=lambda prompt: prompts.append(prompt),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    assert len(prompts) == 1
    assert "标定板" in prompts[0]
    assert sleeps == [1.0, 1.0]
    assert robot.right.end_effector.calls == [
        {"value": 0.1, "force": 1.0},
        {"value": 0.0, "force": 1.0},
    ]
    assert robot.left.end_effector.calls == [
        {"value": 0.1, "force": 1.0},
        {"value": 0.0, "force": 1.0},
    ]


def test_prompt_before_capture_waits_for_enter_when_enabled() -> None:
    prompts = []

    prompt_before_capture(
        enabled=True,
        target_arm="right",
        target_index=3,
        total_targets=60,
        input_fn=lambda prompt: prompts.append(prompt),
    )

    assert len(prompts) == 1
    assert "3/60" in prompts[0]
    assert "right" in prompts[0]
    assert "采集" in prompts[0]


def test_prompt_before_capture_skips_when_disabled() -> None:
    prompts = []

    prompt_before_capture(
        enabled=False,
        target_arm="right",
        target_index=3,
        total_targets=60,
        input_fn=lambda prompt: prompts.append(prompt),
    )

    assert prompts == []
