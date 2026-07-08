from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .ee_local_se3_adapter import NeroEETargets
from .mapping import namespaced_gripper_name, namespaced_joint_names

SeedCandidate = tuple[str, Any]


def _as_joint_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {vector.shape}.")
    return vector


def nero_pose_to_curobo_pose(pose: Any, euler_order: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    pose_array = np.asarray(pose, dtype=float)
    if pose_array.shape != (6,):
        raise ValueError(
            f"Expected Nero EE pose with shape (6,) [x, y, z, roll, pitch, yaw], got {pose_array.shape}."
        )
    position = pose_array[:3]
    quaternion_xyzw = Rotation.from_euler(euler_order, pose_array[3:6]).as_quat()
    quaternion_wxyz = np.asarray(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=float,
    )
    return position, quaternion_wxyz


def _selected_ik_index(solutions: Any, success: Any, seed: Any | None = None) -> int:
    solutions_array = np.asarray(solutions, dtype=float)
    success_array = np.asarray(success, dtype=bool).reshape(-1)
    if solutions_array.ndim != 2 or solutions_array.shape[1] != 7:
        raise ValueError(f"IK solutions must have shape (N, 7), got {solutions_array.shape}.")
    if solutions_array.shape[0] != success_array.shape[0]:
        raise ValueError(
            f"IK success mask length {success_array.shape[0]} does not match solutions {solutions_array.shape[0]}."
        )

    valid_idx = np.flatnonzero(success_array)
    if len(valid_idx) == 0:
        raise RuntimeError("cuRobo IK failed: no successful seed returned a valid joint solution.")
    if seed is None:
        return int(valid_idx[0])

    seed_q = _as_joint_vector(seed, name="seed")
    valid_solutions = solutions_array[valid_idx]
    nearest = np.argmin(np.linalg.norm(valid_solutions - seed_q, axis=1))
    return int(valid_idx[int(nearest)])


def select_ik_solution(solutions: Any, success: Any, seed: Any | None = None) -> np.ndarray:
    solutions_array = np.asarray(solutions, dtype=float)
    idx = _selected_ik_index(solutions_array, success, seed=seed)
    return np.asarray(solutions_array[idx], dtype=float).copy()


class NeroCuroboArmIK:
    def __init__(
        self,
        *,
        robot_file: str,
        euler_order: str = "xyz",
        num_seeds: int = 32,
        position_threshold: float = 0.01,
        rotation_threshold: float = 0.05,
        device: str = "cuda",
    ) -> None:
        if num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}.")
        if position_threshold < 0:
            raise ValueError(f"position_threshold must be non-negative, got {position_threshold}.")
        if rotation_threshold < 0:
            raise ValueError(f"rotation_threshold must be non-negative, got {rotation_threshold}.")

        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, JointState, Pose

        self.euler_order = euler_order
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.device = device
        self.GoalToolPose = GoalToolPose
        self.JointState = JointState
        self.Pose = Pose

        config = InverseKinematicsCfg.create(
            robot=robot_file,
            num_seeds=num_seeds,
            position_tolerance=position_threshold,
            orientation_tolerance=rotation_threshold,
        )
        self.ik = InverseKinematics(config)
        self.target_link = self.ik.tool_frames[0]
        self.return_seeds = min(8, max(1, num_seeds))
        self.locked_gripper_joints = np.asarray(
            [-0.02500000037252903, 0.02500000037252903],
            dtype=float,
        )

    def solve(self, pose: Any, current_joints: Any | None = None) -> np.ndarray:
        position, quaternion = nero_pose_to_curobo_pose(pose, self.euler_order)
        goal_pose = self.Pose(
            position=torch.as_tensor(position, device=self.device, dtype=torch.float32).view(1, 3),
            quaternion=torch.as_tensor(quaternion, device=self.device, dtype=torch.float32).view(1, 4),
        )
        goal = self.GoalToolPose.from_poses({self.target_link: goal_pose}, num_goalset=1)

        current_state = None
        seed_q7 = None
        if current_joints is not None:
            seed_q7 = _as_joint_vector(current_joints, name="current_joints")
            seed_q9 = np.concatenate([seed_q7, self.locked_gripper_joints])
            current_state = self.JointState.from_position(
                torch.as_tensor(seed_q9, device=self.device, dtype=torch.float32).view(1, -1),
                joint_names=self.ik.kinematics.joint_names,
            )

        result = self.ik.solve_pose(
            goal_tool_poses=goal,
            current_state=current_state,
            return_seeds=self.return_seeds,
        )
        success = result.success.detach().cpu().numpy().reshape(-1).astype(bool)
        solutions = result.js_solution.position.detach().cpu().numpy().reshape(-1, self.ik.action_dim)[:, :7]
        try:
            selected_idx = _selected_ik_index(solutions, success, seed=seed_q7)
        except RuntimeError as exc:
            pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
            rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
            raise RuntimeError(
                f"cuRobo IK failed for pose={np.asarray(pose, dtype=float).tolist()}; "
                f"position_error={pos_err:.6f}, rotation_error={rot_err:.6f}"
            ) from exc

        pos_errors = result.position_error.detach().cpu().numpy().reshape(-1)
        rot_errors = result.rotation_error.detach().cpu().numpy().reshape(-1)
        pos_err = float(pos_errors[selected_idx])
        rot_err = float(rot_errors[selected_idx])
        if pos_err > self.position_threshold or rot_err > self.rotation_threshold:
            raise RuntimeError(
                f"cuRobo IK error too large for pose={np.asarray(pose, dtype=float).tolist()}; "
                f"position_error={pos_err:.6f} > {self.position_threshold:.6f} or "
                f"rotation_error={rot_err:.6f} > {self.rotation_threshold:.6f}"
            )
        return np.asarray(solutions[selected_idx], dtype=float)


class NeroDualCuroboIKAdapter:
    def __init__(
        self,
        *,
        robot_file: str = "nero_custom.yml",
        euler_order: str = "xyz",
        num_seeds: int = 32,
        position_threshold: float = 0.01,
        rotation_threshold: float = 0.05,
        device: str = "cuda",
        right_ik: Any | None = None,
        left_ik: Any | None = None,
    ) -> None:
        self.right = right_ik or NeroCuroboArmIK(
            robot_file=robot_file,
            euler_order=euler_order,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            device=device,
        )
        self.left = left_ik or NeroCuroboArmIK(
            robot_file=robot_file,
            euler_order=euler_order,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            device=device,
        )

    @staticmethod
    def _solve_with_seed_candidates(
        ik: Any,
        pose: Any,
        *,
        arm: str,
        current_joints: Any,
        seed_candidates: list[SeedCandidate] | tuple[SeedCandidate, ...] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        candidates: list[SeedCandidate] = [("real_current_joints", current_joints)]
        if seed_candidates:
            candidates.extend(seed_candidates)

        for source, seed in candidates:
            seed_q = _as_joint_vector(seed, name=f"{arm} {source}")
            try:
                solution = _as_joint_vector(
                    ik.solve(pose, current_joints=seed_q),
                    name=f"{arm} IK solution",
                )
            except Exception as exc:
                attempts.append(
                    {
                        "seed_source": source,
                        "seed_joints": seed_q,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            return solution, {
                "seed_source": source,
                "seed_joints": seed_q,
                "attempts": attempts,
            }

        attempted_sources = [attempt["seed_source"] for attempt in attempts]
        errors = "; ".join(
            f"{attempt['seed_source']}: {attempt['error']}" for attempt in attempts
        )
        raise RuntimeError(
            f"{arm} IK failed for all seed candidates {attempted_sources}: {errors}"
        )

    def ee_targets_to_joint_action_with_metadata(
        self,
        targets: NeroEETargets,
        *,
        right_current_joints: Any,
        left_current_joints: Any,
        right_seed_candidates: list[SeedCandidate] | tuple[SeedCandidate, ...] | None = None,
        left_seed_candidates: list[SeedCandidate] | tuple[SeedCandidate, ...] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        right_joints, right_metadata = self._solve_with_seed_candidates(
            self.right,
            targets.right_pose,
            arm="right",
            current_joints=right_current_joints,
            seed_candidates=right_seed_candidates,
        )
        left_joints, left_metadata = self._solve_with_seed_candidates(
            self.left,
            targets.left_pose,
            arm="left",
            current_joints=left_current_joints,
            seed_candidates=left_seed_candidates,
        )
        action = {
            name: float(value)
            for name, value in zip(namespaced_joint_names("right"), right_joints, strict=True)
        }
        action.update(
            {
                name: float(value)
                for name, value in zip(namespaced_joint_names("left"), left_joints, strict=True)
            }
        )
        action[namespaced_gripper_name("right")] = float(targets.right_gripper_width)
        action[namespaced_gripper_name("left")] = float(targets.left_gripper_width)
        return action, {"right": right_metadata, "left": left_metadata}

    def ee_targets_to_joint_action(
        self,
        targets: NeroEETargets,
        *,
        right_current_joints: Any,
        left_current_joints: Any,
    ) -> dict[str, float]:
        action, _metadata = self.ee_targets_to_joint_action_with_metadata(
            targets,
            right_current_joints=right_current_joints,
            left_current_joints=left_current_joints,
        )
        return action


__all__ = [
    "NeroCuroboArmIK",
    "NeroDualCuroboIKAdapter",
    "nero_pose_to_curobo_pose",
    "select_ik_solution",
]
