from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

EE_LOCAL_SE3_ACTION_NAMES = [
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_rotvec_x",
    "right_ee_rotvec_y",
    "right_ee_rotvec_z",
    "right_gripper_width",
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_rotvec_x",
    "left_ee_rotvec_y",
    "left_ee_rotvec_z",
    "left_gripper_width",
    "base_or_head_x",
    "base_or_head_y",
]

FLANGE_POSE_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def _as_vector(values: Any, *, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {vector.shape}.")
    return vector


def _as_rotation_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    return matrix


def _parse_opencv_matrix_yaml(path: str | Path, key: str) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(key)}:\s*!!opencv-matrix\s*"
        r"rows:\s*(?P<rows>\d+)\s*"
        r"cols:\s*(?P<cols>\d+)\s*"
        r"dt:\s*(?P<dt>\w+)\s*"
        r"data:\s*\[(?P<data>.*?)\]",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find OpenCV matrix key {key!r} in {path}.")

    rows = int(match.group("rows"))
    cols = int(match.group("cols"))
    data = [float(item.strip()) for item in match.group("data").replace("\n", " ").split(",") if item.strip()]
    if len(data) != rows * cols:
        raise ValueError(f"OpenCV matrix {key!r} in {path} has {len(data)} values, expected {rows * cols}.")
    return np.asarray(data, dtype=float).reshape(rows, cols)


@dataclass(frozen=True)
class SE3Transform:
    """Rigid transform with convention: p_target = R_target_source @ p_source + t_target_source."""

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _as_rotation_matrix(self.rotation, name="rotation"))
        object.__setattr__(self, "translation", _as_vector(self.translation, dim=3, name="translation"))

    @classmethod
    def identity(cls) -> "SE3Transform":
        return cls()

    @classmethod
    def from_matrix(cls, matrix: Any) -> "SE3Transform":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"SE3 matrix must have shape (4, 4), got {matrix.shape}.")
        return cls(rotation=matrix[:3, :3], translation=matrix[:3, 3])

    @classmethod
    def from_opencv_yaml(cls, path: str | Path, *, key: str = "T_base_cam") -> "SE3Transform":
        return cls.from_matrix(_parse_opencv_matrix_yaml(path, key))

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> "SE3Transform":
        rotation_inv = self.rotation.T
        translation_inv = -(rotation_inv @ self.translation)
        return SE3Transform(rotation=rotation_inv, translation=translation_inv)

    def transform_position(self, position: Any) -> np.ndarray:
        position = _as_vector(position, dim=3, name="position")
        return self.rotation @ position + self.translation


@dataclass(frozen=True)
class NeroEETargets:
    right_pose: np.ndarray
    right_gripper_width: float
    left_pose: np.ndarray
    left_gripper_width: float
    base_or_head_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def __post_init__(self) -> None:
        object.__setattr__(self, "right_pose", _as_vector(self.right_pose, dim=6, name="right_pose"))
        object.__setattr__(self, "left_pose", _as_vector(self.left_pose, dim=6, name="left_pose"))
        object.__setattr__(
            self,
            "base_or_head_xy",
            _as_vector(self.base_or_head_xy, dim=2, name="base_or_head_xy"),
        )
        object.__setattr__(self, "right_gripper_width", float(self.right_gripper_width))
        object.__setattr__(self, "left_gripper_width", float(self.left_gripper_width))


@dataclass
class NeroEELocalSE3Adapter:
    """Convert between Nero flange Euler poses and the PI0.5 ee_local_se3 16D representation.

    Nero SDK flange poses are interpreted as link7/flange poses:
    [x, y, z, roll, pitch, yaw], with scipy Euler order ``xyz`` by default.
    No TCP/tool offset is applied.
    """

    camera_from_right_base: SE3Transform = field(default_factory=SE3Transform.identity)
    camera_from_left_base: SE3Transform = field(default_factory=SE3Transform.identity)
    base_or_head_xy: tuple[float, float] = (0.0, 0.0)
    euler_order: str = "xyz"

    @classmethod
    def from_camera_to_base_yamls(
        cls,
        *,
        right_camera_to_base_yaml: str | Path,
        left_camera_to_base_yaml: str | Path,
        base_or_head_xy: tuple[float, float] = (0.0, 0.0),
        euler_order: str = "xyz",
    ) -> "NeroEELocalSE3Adapter":
        """Build adapter from hand-eye files that store camera->base transforms.

        The OpenCV YAML files currently store ``T_base_cam``. That maps camera-frame
        coordinates into each Nero arm base frame, so it is inverted here to get
        the base->camera transform needed by the policy representation.
        """
        right_base_from_camera = SE3Transform.from_opencv_yaml(right_camera_to_base_yaml, key="T_base_cam")
        left_base_from_camera = SE3Transform.from_opencv_yaml(left_camera_to_base_yaml, key="T_base_cam")
        return cls(
            camera_from_right_base=right_base_from_camera.inverse(),
            camera_from_left_base=left_base_from_camera.inverse(),
            base_or_head_xy=base_or_head_xy,
            euler_order=euler_order,
        )

    def _pose_base_to_policy_ee(self, pose: Any, camera_from_base: SE3Transform) -> np.ndarray:
        pose = _as_vector(pose, dim=6, name="Nero flange pose")
        position_camera = camera_from_base.transform_position(pose[:3])
        rotation_camera = Rotation.from_matrix(camera_from_base.rotation) * Rotation.from_euler(
            self.euler_order, pose[3:6]
        )
        return np.concatenate((position_camera, rotation_camera.as_rotvec()))

    def _policy_ee_to_base_pose(self, ee_pose: Any, camera_from_base: SE3Transform) -> np.ndarray:
        ee_pose = _as_vector(ee_pose, dim=6, name="policy EE pose")
        base_from_camera = camera_from_base.inverse()
        position_base = base_from_camera.transform_position(ee_pose[:3])
        rotation_base = Rotation.from_matrix(base_from_camera.rotation) * Rotation.from_rotvec(ee_pose[3:6])
        euler_base = rotation_base.as_euler(self.euler_order)
        return np.concatenate((position_base, euler_base))

    @staticmethod
    def _flange_pose_from_observation(observation: dict[str, Any], arm: str) -> np.ndarray:
        keys = [f"{arm}_flange_{component}" for component in FLANGE_POSE_COMPONENTS]
        missing = [key for key in keys if key not in observation]
        if missing:
            raise KeyError(f"Missing Nero flange observation keys: {missing}")
        return np.asarray([float(observation[key]) for key in keys], dtype=float)

    @staticmethod
    def _gripper_from_observation(observation: dict[str, Any], arm: str) -> float:
        key = f"{arm}_gripper_width"
        if key not in observation:
            raise KeyError(f"Missing Nero gripper observation key: {key}")
        return float(observation[key])

    def flange_observation_to_policy_state(self, observation: dict[str, Any]) -> np.ndarray:
        right_pose = self._flange_pose_from_observation(observation, "right")
        left_pose = self._flange_pose_from_observation(observation, "left")
        right_ee = self._pose_base_to_policy_ee(right_pose, self.camera_from_right_base)
        left_ee = self._pose_base_to_policy_ee(left_pose, self.camera_from_left_base)
        return np.concatenate(
            (
                right_ee,
                np.asarray([self._gripper_from_observation(observation, "right")], dtype=float),
                left_ee,
                np.asarray([self._gripper_from_observation(observation, "left")], dtype=float),
                _as_vector(self.base_or_head_xy, dim=2, name="base_or_head_xy"),
            )
        )

    def read_robot_policy_state(self, robot: Any) -> np.ndarray:
        get_flange_state_observation = getattr(robot, "get_flange_state_observation", None)
        if not callable(get_flange_state_observation):
            raise TypeError("robot must expose get_flange_state_observation() for EE local SE3 inference.")
        return self.flange_observation_to_policy_state(get_flange_state_observation())

    def policy_action_to_nero_ee_targets(self, action: Any) -> NeroEETargets:
        action = _as_vector(action, dim=16, name="policy action")
        right_pose = self._policy_ee_to_base_pose(action[0:6], self.camera_from_right_base)
        left_pose = self._policy_ee_to_base_pose(action[7:13], self.camera_from_left_base)
        return NeroEETargets(
            right_pose=right_pose,
            right_gripper_width=float(action[6]),
            left_pose=left_pose,
            left_gripper_width=float(action[13]),
            base_or_head_xy=action[14:16],
        )


__all__ = [
    "EE_LOCAL_SE3_ACTION_NAMES",
    "FLANGE_POSE_COMPONENTS",
    "NeroEELocalSE3Adapter",
    "NeroEETargets",
    "SE3Transform",
]
