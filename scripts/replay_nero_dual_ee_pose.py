#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


READY_RIGHT_POSE = [
    0.2732181621364244,
    -0.030025818406302345,
    0.35750673546543393,
    -1.6118205859733734,
    0.09584266857003589,
    3.066317491332911,
]
READY_LEFT_POSE = [
    0.2873361223957065,
    -0.01389206349906485,
    0.35802865521534843,
    -1.6379877598473152,
    0.05157470481268328,
    -3.124208233763824,
]
READY_RIGHT_GRIPPER = 0.02853
READY_LEFT_GRIPPER = 0.023507


def nero_pose_to_curobo_pose(pose: list[float], euler_order: str) -> tuple[np.ndarray, np.ndarray]:
    if len(pose) != 6:
        raise ValueError(f"Expected Nero EE pose with 6 values [x, y, z, rx, ry, rz], got {len(pose)}")
    pose_array = np.asarray(pose, dtype=float)
    position = pose_array[:3]
    quaternion_xyzw = Rotation.from_euler(euler_order, pose_array[3:6]).as_quat()
    quaternion_wxyz = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=float,
    )
    return position, quaternion_wxyz


class DualCuroboIKBackend:
    def __init__(self, *, right: object, left: object) -> None:
        self.right = right
        self.left = left


class ReplayProfileCsv:
    fieldnames = [
        "dataset",
        "phase",
        "frame_index",
        "frame_count",
        "fps",
        "frame_total_s",
        "right_ik_s",
        "left_ik_s",
        "right_command_s",
        "left_command_s",
        "right_wait_s",
        "left_wait_s",
        "right_steps",
        "left_steps",
        "right_soft_timeouts",
        "left_soft_timeouts",
        "right_max_feedback_error_rad",
        "left_max_feedback_error_rad",
    ]

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()

    def write_row(self, row: dict[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("ReplayProfileCsv must be used as a context manager")
        self._writer.writerow({field: row.get(field, "") for field in self.fieldnames})
        self._file.flush()


class CuroboArmIK:
    def __init__(
        self,
        *,
        robot_file: str,
        euler_order: str,
        num_seeds: int,
        position_threshold: float,
        rotation_threshold: float,
    ) -> None:
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, Pose

        self.euler_order = euler_order
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.GoalToolPose = GoalToolPose
        self.Pose = Pose
        from curobo.types import JointState

        self.JointState = JointState

        config = InverseKinematicsCfg.create(robot=robot_file, num_seeds=num_seeds)
        self.ik = InverseKinematics(config)
        self.target_link = self.ik.tool_frames[0]
        self.return_seeds = min(8, max(1, num_seeds))
        self.locked_gripper_joints = np.array([-0.02500000037252903, 0.02500000037252903], dtype=float)

    def solve(self, pose: list[float], seed: np.ndarray | None = None) -> np.ndarray:
        position, quaternion = nero_pose_to_curobo_pose(pose, self.euler_order)
        goal_pose = self.Pose(
            position=torch.as_tensor(position, device="cuda", dtype=torch.float32).view(1, 3),
            quaternion=torch.as_tensor(quaternion, device="cuda", dtype=torch.float32).view(1, 4),
        )
        goal = self.GoalToolPose.from_poses({self.target_link: goal_pose}, num_goalset=1)
        current_state = None
        seed_q7 = None
        if seed is not None:
            seed_q7 = np.asarray(seed, dtype=float).reshape(7)
            seed_q9 = np.concatenate([seed_q7, self.locked_gripper_joints])
            current_state = self.JointState.from_position(
                torch.as_tensor(seed_q9, device="cuda", dtype=torch.float32).view(1, -1),
                joint_names=self.ik.kinematics.joint_names,
            )
        result = self.ik.solve_pose(goal_tool_poses=goal, current_state=current_state, return_seeds=self.return_seeds)
        success = result.success.detach().cpu().numpy().reshape(-1).astype(bool)
        if not np.any(success):
            pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
            rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
            raise RuntimeError(
                f"cuRobo IK failed for pose={pose}; position_error={pos_err:.6f}, rotation_error={rot_err:.6f}"
            )
        solutions = result.js_solution.position.detach().cpu().numpy().reshape(-1, self.ik.action_dim)[:, :7]
        pos_errors = result.position_error.detach().cpu().numpy().reshape(-1)
        rot_errors = result.rotation_error.detach().cpu().numpy().reshape(-1)
        valid_idx = np.flatnonzero(success)
        if seed_q7 is None:
            idx = int(valid_idx[0])
        else:
            valid_solutions = solutions[valid_idx]
            idx = int(valid_idx[np.argmin(np.linalg.norm(valid_solutions - seed_q7, axis=1))])
        pos_err = float(pos_errors[idx])
        rot_err = float(rot_errors[idx])
        if pos_err > self.position_threshold or rot_err > self.rotation_threshold:
            raise RuntimeError(
                f"cuRobo IK error too large for pose={pose}; "
                f"position_error={pos_err:.6f} > {self.position_threshold:.6f} or "
                f"rotation_error={rot_err:.6f} > {self.rotation_threshold:.6f}"
            )
        return solutions[idx]


def make_robot(channel: str, firmware_version: str, interface: str):
    firmware = getattr(NeroFW, firmware_version)
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=firmware,
        interface=interface,
        channel=channel,
    )
    return AgxArmFactory.create_arm(cfg)


def enable_robot(robot, *, speed_percent: int, reset_on_connect: bool, enable_retry_s: float, motion_mode: str) -> None:
    robot.connect()
    robot.set_follower_mode()
    time.sleep(0.2)
    robot.set_motion_mode(motion_mode)
    time.sleep(0.2)
    if reset_on_connect:
        robot.reset()
        time.sleep(0.2)
    robot.clear_joint_error(255)
    time.sleep(0.2)
    while True:
        robot.enable()
        time.sleep(1.0)
        if all(robot.get_joints_enable_status_list()):
            break
        print("Waiting for Nero arm enable...")
        time.sleep(enable_retry_s)
    robot.set_speed_percent(speed_percent)


def load_action_names(dataset_root: Path) -> list[str] | None:
    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("features", {}).get("action", {}).get("names")
    if names is None:
        return None
    return list(names)


def load_actions(dataset_root: Path, *, episode: int = 0) -> tuple[np.ndarray, int, list[str] | None]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    df = pd.concat([pd.read_parquet(path, columns=["episode_index", "frame_index", "action"]) for path in parquet_files])
    episode_df = df[df["episode_index"] == episode].sort_values("frame_index")
    if episode_df.empty:
        available = sorted(int(value) for value in df["episode_index"].unique())
        raise ValueError(f"Episode {episode} not found in {dataset_root}. Available episodes: {available}")
    actions = np.stack(episode_df["action"].to_numpy()).astype(float)
    if actions.shape[1] != 14:
        raise ValueError(f"Expected EE action shape (N, 14), got {actions.shape}")

    fps = 30
    info_path = dataset_root / "meta/info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            fps = int(json.load(f).get("fps", fps))
    return actions, fps, load_action_names(dataset_root)


def load_available_episodes(dataset_root: Path) -> list[int]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    df = pd.concat([pd.read_parquet(path, columns=["episode_index"]) for path in parquet_files])
    return sorted(int(value) for value in df["episode_index"].unique())


def _pose_from_action_names(action: np.ndarray, action_names: list[str], prefix: str) -> list[float]:
    aliases = {
        "x": ("x",),
        "y": ("y",),
        "z": ("z",),
        "roll": ("roll",),
        "pitch": ("pitch",),
        "yaw": ("yaw",),
    }
    values = []
    for component, suffixes in aliases.items():
        matches = [
            idx
            for idx, name in enumerate(action_names)
            if name.startswith(prefix) and any(name.endswith(f"_{suffix}") for suffix in suffixes)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {prefix} {component} action name, found {len(matches)} in {action_names}"
            )
        values.append(float(action[matches[0]]))
    return values


def _value_from_action_name(action: np.ndarray, action_names: list[str], name: str) -> float:
    try:
        return float(action[action_names.index(name)])
    except ValueError as exc:
        raise ValueError(f"Expected action name {name!r} in {action_names}") from exc


def split_dual_ee_action(
    action: np.ndarray,
    action_names: list[str] | None = None,
) -> tuple[list[float], float, list[float], float]:
    if action_names is None:
        right_pose = action[0:6].astype(float).tolist()
        right_gripper = float(action[6])
        left_pose = action[7:13].astype(float).tolist()
        left_gripper = float(action[13])
        return right_pose, right_gripper, left_pose, left_gripper

    if len(action) != len(action_names):
        raise ValueError(f"Action has {len(action)} values, but action_names has {len(action_names)} names.")
    right_pose = _pose_from_action_names(action, action_names, "right_")
    left_pose = _pose_from_action_names(action, action_names, "left_")
    right_gripper = _value_from_action_name(action, action_names, "right_gripper_width")
    left_gripper = _value_from_action_name(action, action_names, "left_gripper_width")
    return right_pose, right_gripper, left_pose, left_gripper


def read_robot_joints(robot) -> np.ndarray:
    joints = robot.get_joint_angles()
    while joints is None:
        time.sleep(0.01)
        joints = robot.get_joint_angles()
    joints = getattr(joints, "msg", joints)
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (7,):
        raise ValueError(f"Nero SDK returned joint shape {joints.shape}; expected 7 values.")
    return joints


def limit_joint_step(current: np.ndarray, target: np.ndarray, max_step: float | None) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if max_step is None or max_step <= 0:
        return target
    delta = np.clip(target - current, -max_step, max_step)
    return current + delta


def send_limited_joint_command(
    robot,
    target_joints: np.ndarray,
    *,
    arm_name: str,
    max_joint_step_rad: float | None,
    interpolate_to_target: bool,
    step_sleep_s: float,
    current_joints: np.ndarray | None = None,
    wait_for_feedback: bool = False,
    joint_target_tolerance_rad: float = 0.02,
    joint_wait_timeout_s: float = 2.0,
    joint_timeout_error_rad: float = 0.08,
    monotonic_time=time.perf_counter,
) -> dict[str, float | int]:
    stats: dict[str, float | int] = {
        "steps": 0,
        "wait_s": 0.0,
        "soft_timeouts": 0,
        "max_feedback_error_rad": 0.0,
    }
    current = read_robot_joints(robot) if current_joints is None else np.asarray(current_joints, dtype=float)
    target = np.asarray(target_joints, dtype=float)
    if target.shape != (7,):
        raise ValueError(f"Expected target joint shape (7,), got {target.shape}")

    if not interpolate_to_target:
        robot.move_js(limit_joint_step(current, target, max_joint_step_rad).tolist())
        stats["steps"] = 1
        return stats

    while True:
        next_command = limit_joint_step(current, target, max_joint_step_rad)
        robot.move_js(next_command.tolist())
        stats["steps"] = int(stats["steps"]) + 1
        command_is_target = np.allclose(next_command, target, rtol=0.0, atol=1e-9)
        if wait_for_feedback:
            wait_start = monotonic_time()
            deadline = monotonic_time() + joint_wait_timeout_s
            while True:
                feedback = read_robot_joints(robot)
                max_error = float(np.max(np.abs(feedback - next_command)))
                stats["max_feedback_error_rad"] = max(float(stats["max_feedback_error_rad"]), max_error)
                if max_error <= joint_target_tolerance_rad:
                    stats["wait_s"] = float(stats["wait_s"]) + max(0.0, monotonic_time() - wait_start)
                    current = feedback
                    break
                if monotonic_time() > deadline:
                    stats["wait_s"] = float(stats["wait_s"]) + max(0.0, monotonic_time() - wait_start)
                    if max_error <= joint_timeout_error_rad:
                        stats["soft_timeouts"] = int(stats["soft_timeouts"]) + 1
                        print(
                            f"Warning: {arm_name} Nero joints did not reach tolerance before timeout; "
                            f"max_error={max_error:.4f} rad, tolerance={joint_target_tolerance_rad:.4f} rad. Continuing."
                        )
                        current = feedback
                        break
                    raise TimeoutError(
                        f"Timed out waiting for {arm_name} Nero joints; max_error={max_error:.4f} rad, "
                        f"tolerance={joint_target_tolerance_rad:.4f} rad, "
                        f"hard_error={joint_timeout_error_rad:.4f} rad"
                    )
                if step_sleep_s > 0:
                    time.sleep(step_sleep_s)
        if command_is_target:
            break
        if not wait_for_feedback:
            current = next_command
        if step_sleep_s > 0:
            time.sleep(step_sleep_s)
    return stats


def move_dual_ee(
    right,
    left,
    right_gripper,
    left_gripper,
    *,
    right_pose: list[float],
    left_pose: list[float],
    right_width: float,
    left_width: float,
    gripper_force: float,
    ik_backend: DualCuroboIKBackend | None = None,
    max_joint_step_rad: float | None = None,
    interpolate_to_target: bool = False,
    step_sleep_s: float = 0.0,
    wait_for_feedback: bool = False,
    joint_target_tolerance_rad: float = 0.02,
    joint_wait_timeout_s: float = 2.0,
    joint_timeout_error_rad: float = 0.08,
    monotonic_time=time.perf_counter,
) -> dict[str, float | int]:
    stats: dict[str, float | int] = {
        "right_ik_s": 0.0,
        "left_ik_s": 0.0,
        "right_command_s": 0.0,
        "left_command_s": 0.0,
        "right_wait_s": 0.0,
        "left_wait_s": 0.0,
        "right_steps": 0,
        "left_steps": 0,
        "right_soft_timeouts": 0,
        "left_soft_timeouts": 0,
        "right_max_feedback_error_rad": 0.0,
        "left_max_feedback_error_rad": 0.0,
    }
    if ik_backend is None:
        right.move_p(right_pose)
        left.move_p(left_pose)
    else:
        right_current = read_robot_joints(right)
        left_current = read_robot_joints(left)
        right_ik_start = monotonic_time()
        right_target = ik_backend.right.solve(right_pose, seed=right_current)
        stats["right_ik_s"] = monotonic_time() - right_ik_start
        right_command_start = monotonic_time()
        right_command_stats = send_limited_joint_command(
            right,
            right_target,
            arm_name="right",
            max_joint_step_rad=max_joint_step_rad,
            interpolate_to_target=interpolate_to_target,
            step_sleep_s=step_sleep_s,
            current_joints=right_current,
            wait_for_feedback=wait_for_feedback,
            joint_target_tolerance_rad=joint_target_tolerance_rad,
            joint_wait_timeout_s=joint_wait_timeout_s,
            joint_timeout_error_rad=joint_timeout_error_rad,
            monotonic_time=monotonic_time,
        )
        stats["right_command_s"] = monotonic_time() - right_command_start
        stats["right_wait_s"] = right_command_stats["wait_s"]
        stats["right_steps"] = right_command_stats["steps"]
        stats["right_soft_timeouts"] = right_command_stats["soft_timeouts"]
        stats["right_max_feedback_error_rad"] = right_command_stats["max_feedback_error_rad"]

        left_ik_start = monotonic_time()
        left_target = ik_backend.left.solve(left_pose, seed=left_current)
        stats["left_ik_s"] = monotonic_time() - left_ik_start
        left_command_start = monotonic_time()
        left_command_stats = send_limited_joint_command(
            left,
            left_target,
            arm_name="left",
            max_joint_step_rad=max_joint_step_rad,
            interpolate_to_target=interpolate_to_target,
            step_sleep_s=step_sleep_s,
            current_joints=left_current,
            wait_for_feedback=wait_for_feedback,
            joint_target_tolerance_rad=joint_target_tolerance_rad,
            joint_wait_timeout_s=joint_wait_timeout_s,
            joint_timeout_error_rad=joint_timeout_error_rad,
            monotonic_time=monotonic_time,
        )
        stats["left_command_s"] = monotonic_time() - left_command_start
        stats["left_wait_s"] = left_command_stats["wait_s"]
        stats["left_steps"] = left_command_stats["steps"]
        stats["left_soft_timeouts"] = left_command_stats["soft_timeouts"]
        stats["left_max_feedback_error_rad"] = left_command_stats["max_feedback_error_rad"]
    right_gripper.move_gripper_m(value=right_width, force=gripper_force)
    left_gripper.move_gripper_m(value=left_width, force=gripper_force)
    return stats


def move_to_ready(args: argparse.Namespace, right, left, right_gripper, left_gripper) -> None:
    print("Moving to EE ready pose with SDK move_p...")
    if args.ik_backend == "curobo":
        right.set_motion_mode("p")
        left.set_motion_mode("p")
        time.sleep(0.2)
    move_dual_ee(
        right,
        left,
        right_gripper,
        left_gripper,
        right_pose=READY_RIGHT_POSE,
        left_pose=READY_LEFT_POSE,
        right_width=READY_RIGHT_GRIPPER,
        left_width=READY_LEFT_GRIPPER,
        gripper_force=args.gripper_force,
    )
    time.sleep(args.ready_wait_s)
    if args.ik_backend == "curobo":
        right.set_motion_mode("js")
        left.set_motion_mode("js")
        time.sleep(0.2)


def write_profile_row(
    profile_writer,
    *,
    dataset_root: Path,
    phase: str,
    frame_index: int,
    frame_count: int,
    fps: int,
    frame_total_s: float,
    move_stats: dict[str, float | int],
) -> None:
    if profile_writer is None:
        return
    row = {
        "dataset": str(dataset_root),
        "phase": phase,
        "frame_index": frame_index,
        "frame_count": frame_count,
        "fps": fps,
        "frame_total_s": frame_total_s,
    }
    row.update(move_stats)
    profile_writer.write_row(row)


def replay_dataset(
    args: argparse.Namespace,
    dataset_root: Path,
    episode: int,
    right,
    left,
    right_gripper,
    left_gripper,
    ik_backend=None,
    profile_writer=None,
) -> None:
    actions, fps, action_names = load_actions(dataset_root, episode=episode)
    if args.fps is not None:
        fps = args.fps
    dt = 1.0 / fps
    print(f"Replaying {dataset_root} episode={episode} frames={len(actions)} fps={fps}")

    first_right, first_rg, first_left, first_lg = split_dual_ee_action(actions[0], action_names)
    print(f"Moving to first EE pose with {args.ik_backend} IK...")
    first_start = time.perf_counter()
    first_stats = move_dual_ee(
        right,
        left,
        right_gripper,
        left_gripper,
        right_pose=first_right,
        left_pose=first_left,
        right_width=first_rg,
        left_width=first_lg,
        gripper_force=args.gripper_force,
        ik_backend=ik_backend,
        max_joint_step_rad=args.max_joint_step_rad,
        interpolate_to_target=args.interpolate_first_target,
        step_sleep_s=args.control_dt_s,
        wait_for_feedback=args.interpolate_first_target,
        joint_target_tolerance_rad=args.joint_target_tolerance_rad,
        joint_wait_timeout_s=args.joint_wait_timeout_s,
        joint_timeout_error_rad=args.joint_timeout_error_rad,
    )
    write_profile_row(
        profile_writer,
        dataset_root=dataset_root,
        phase=f"episode_{episode}_first_target",
        frame_index=-1,
        frame_count=len(actions),
        fps=fps,
        frame_total_s=time.perf_counter() - first_start,
        move_stats=first_stats,
    )
    time.sleep(args.takeover_time_s)

    for idx, action in enumerate(actions):
        start = time.perf_counter()
        right_pose, right_width, left_pose, left_width = split_dual_ee_action(action, action_names)
        move_stats = move_dual_ee(
            right,
            left,
            right_gripper,
            left_gripper,
            right_pose=right_pose,
            left_pose=left_pose,
            right_width=right_width,
            left_width=left_width,
            gripper_force=args.gripper_force,
            ik_backend=ik_backend,
            max_joint_step_rad=args.max_joint_step_rad,
            interpolate_to_target=args.interpolate_each_frame,
            step_sleep_s=args.control_dt_s if args.interpolate_each_frame else 0.0,
            wait_for_feedback=args.interpolate_each_frame,
            joint_target_tolerance_rad=args.joint_target_tolerance_rad,
            joint_wait_timeout_s=args.joint_wait_timeout_s,
            joint_timeout_error_rad=args.joint_timeout_error_rad,
        )
        elapsed = time.perf_counter() - start
        write_profile_row(
            profile_writer,
            dataset_root=dataset_root,
            phase=f"episode_{episode}_replay",
            frame_index=idx,
            frame_count=len(actions),
            fps=fps,
            frame_total_s=elapsed,
            move_stats=move_stats,
        )
        if idx % 100 == 0 or idx == len(actions) - 1:
            print(f"  frame {idx + 1}/{len(actions)}")
        time.sleep(max(dt - elapsed, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Nero dual-arm EE pose datasets with SDK move_p().")
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--episode", default="0", help="Episode index to replay, or 'all' for every episode.")
    parser.add_argument("--right-channel", default="nero_right")
    parser.add_argument("--left-channel", default="nero_left")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware-version", default="V120")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--enable-retry-s", type=float, default=0.2)
    parser.add_argument("--reset-on-connect", action="store_true")
    parser.add_argument("--skip-ready", action="store_true")
    parser.add_argument("--ready-wait-s", type=float, default=3.0)
    parser.add_argument("--takeover-time-s", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="Do not pause before each dataset.")
    parser.add_argument("--ik-backend", choices=("sdk", "curobo"), default="sdk")
    parser.add_argument("--curobo-robot", default="nero_custom.yml")
    parser.add_argument("--curobo-num-seeds", type=int, default=32)
    parser.add_argument("--curobo-position-threshold", type=float, default=0.01)
    parser.add_argument("--curobo-rotation-threshold", type=float, default=0.05)
    parser.add_argument("--euler-order", default="xyz")
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.03,
        help="Maximum per-command joint delta in cuRobo move_js mode. Set <=0 to disable.",
    )
    parser.add_argument(
        "--interpolate-first-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate from current joints to the first replay target before starting timed replay.",
    )
    parser.add_argument(
        "--interpolate-each-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for feedback to reach each replay target before processing the next frame.",
    )
    parser.add_argument("--joint-target-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--joint-wait-timeout-s", type=float, default=2.0)
    parser.add_argument("--joint-timeout-error-rad", type=float, default=0.08)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=None,
        help="Write per-frame timing diagnostics to this CSV path.",
    )
    args = parser.parse_args()
    if args.episode != "all":
        try:
            args.episode = int(args.episode)
        except ValueError as exc:
            raise ValueError("--episode must be an integer or 'all'") from exc

    right = make_robot(args.right_channel, args.firmware_version, args.interface)
    left = make_robot(args.left_channel, args.firmware_version, args.interface)
    right_gripper = right.init_effector(right.OPTIONS.EFFECTOR.AGX_GRIPPER)
    left_gripper = left.init_effector(left.OPTIONS.EFFECTOR.AGX_GRIPPER)
    ik_backend = None
    if args.ik_backend == "curobo":
        print(f"Initializing cuRobo IK with robot={args.curobo_robot}")
        ik_backend = DualCuroboIKBackend(
            right=CuroboArmIK(
                robot_file=args.curobo_robot,
                euler_order=args.euler_order,
                num_seeds=args.curobo_num_seeds,
                position_threshold=args.curobo_position_threshold,
                rotation_threshold=args.curobo_rotation_threshold,
            ),
            left=CuroboArmIK(
                robot_file=args.curobo_robot,
                euler_order=args.euler_order,
                num_seeds=args.curobo_num_seeds,
                position_threshold=args.curobo_position_threshold,
                rotation_threshold=args.curobo_rotation_threshold,
            ),
        )
    try:
        print(f"Connecting right={args.right_channel}, left={args.left_channel}")
        motion_mode = "js" if args.ik_backend == "curobo" else "p"
        enable_robot(
            right,
            speed_percent=args.speed_percent,
            reset_on_connect=args.reset_on_connect,
            enable_retry_s=args.enable_retry_s,
            motion_mode=motion_mode,
        )
        enable_robot(
            left,
            speed_percent=args.speed_percent,
            reset_on_connect=args.reset_on_connect,
            enable_retry_s=args.enable_retry_s,
            motion_mode=motion_mode,
        )
        if not args.skip_ready:
            move_to_ready(args, right, left, right_gripper, left_gripper)

        if args.profile_csv is None:
            for dataset in args.datasets:
                if not dataset.exists():
                    raise FileNotFoundError(dataset)
                episodes = load_available_episodes(dataset) if args.episode == "all" else [args.episode]
                for episode in episodes:
                    if not args.yes:
                        input(
                            f"\nReady to replay {dataset} episode {episode}. "
                            "Press Enter to start, or Ctrl-C to stop..."
                        )
                    replay_dataset(
                        args,
                        dataset,
                        episode,
                        right,
                        left,
                        right_gripper,
                        left_gripper,
                        ik_backend=ik_backend,
                    )
        else:
            print(f"Writing replay profile to {args.profile_csv}")
            with ReplayProfileCsv(args.profile_csv) as profile_writer:
                for dataset in args.datasets:
                    if not dataset.exists():
                        raise FileNotFoundError(dataset)
                    episodes = load_available_episodes(dataset) if args.episode == "all" else [args.episode]
                    for episode in episodes:
                        if not args.yes:
                            input(
                                f"\nReady to replay {dataset} episode {episode}. "
                                "Press Enter to start, or Ctrl-C to stop..."
                            )
                        replay_dataset(
                            args,
                            dataset,
                            episode,
                            right,
                            left,
                            right_gripper,
                            left_gripper,
                            ik_backend=ik_backend,
                            profile_writer=profile_writer,
                        )
    finally:
        for robot in (right, left):
            disconnect = getattr(robot, "disconnect", None)
            if callable(disconnect):
                disconnect()


if __name__ == "__main__":
    main()
