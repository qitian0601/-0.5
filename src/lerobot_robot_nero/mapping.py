from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config_nero import NeroMappingConfig

SO101_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
NERO_JOINT_NAMES = tuple(f"nero_joint_{idx}" for idx in range(1, 8))
GRIPPER_NAME = "gripper_width"


@dataclass
class SO101ToNeroMapping:
    joint_scale: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2], dtype=float)
    )
    joint_direction: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float)
    )
    so101_zero_deg: np.ndarray = field(
        default_factory=lambda: np.array(
            [4.087912, 4.131868, 0.307692, -6.065934, 22.637363, -8.703297, 1.494505],
            dtype=float,
        )
    )
    nero_zero_rad: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))
    nero_limit_low: np.ndarray = field(
        default_factory=lambda: np.array(
            [-2.705261, -1.745330, -2.757621, -1.012291, -2.757621, -0.733039, -1.570797],
            dtype=float,
        )
    )
    nero_limit_high: np.ndarray = field(
        default_factory=lambda: np.array(
            [2.705261, 1.745330, 2.757621, 2.146755, 2.757621, 0.959932, 1.570797],
            dtype=float,
        )
    )
    so101_gripper_min_deg: float = 6.185973
    so101_gripper_max_deg: float = 27.029157
    nero_gripper_width_min: float = 0.0
    nero_gripper_width_max: float = 0.1
    gripper_reverse: bool = False

    @classmethod
    def from_config(cls, config: NeroMappingConfig) -> SO101ToNeroMapping:
        return cls(
            joint_scale=np.asarray(config.joint_scale, dtype=float),
            joint_direction=np.asarray(config.joint_direction, dtype=float),
            so101_zero_deg=np.asarray(config.so101_zero_deg, dtype=float),
            nero_zero_rad=np.asarray(config.nero_zero_rad, dtype=float),
            nero_limit_low=np.asarray(config.nero_limit_low, dtype=float),
            nero_limit_high=np.asarray(config.nero_limit_high, dtype=float),
            so101_gripper_min_deg=config.so101_gripper_min_deg,
            so101_gripper_max_deg=config.so101_gripper_max_deg,
            nero_gripper_width_min=config.nero_gripper_width_min,
            nero_gripper_width_max=config.nero_gripper_width_max,
            gripper_reverse=config.gripper_reverse,
        )

    def __post_init__(self) -> None:
        for name in (
            "joint_scale",
            "joint_direction",
            "so101_zero_deg",
            "nero_zero_rad",
            "nero_limit_low",
            "nero_limit_high",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (7,):
                raise ValueError(f"{name} must contain exactly 7 values, got shape {value.shape}.")
            setattr(self, name, value)

        if self.so101_gripper_max_deg == self.so101_gripper_min_deg:
            raise ValueError("SO101 gripper min and max degrees must be different.")


def namespaced_joint_names(arm: str) -> tuple[str, ...]:
    return tuple(f"{arm}_{name}" for name in NERO_JOINT_NAMES)


def namespaced_gripper_name(arm: str) -> str:
    return f"{arm}_{GRIPPER_NAME}"


def namespaced_action_names(arm: str) -> tuple[str, ...]:
    return (*namespaced_joint_names(arm), namespaced_gripper_name(arm))


def action_to_so101_array(action: dict[str, float]) -> np.ndarray:
    values = []
    for name in SO101_NAMES:
        key = f"{name}.pos"
        if key not in action:
            raise KeyError(f"Missing SO101 action key '{key}'.")
        values.append(float(action[key]))
    return np.asarray(values, dtype=float)


def map_so101_gripper_to_width(gripper_deg: float, *, mapping: SO101ToNeroMapping) -> float:
    ratio = (gripper_deg - mapping.so101_gripper_min_deg) / (
        mapping.so101_gripper_max_deg - mapping.so101_gripper_min_deg
    )
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if mapping.gripper_reverse:
        ratio = 1.0 - ratio
    width = mapping.nero_gripper_width_min + ratio * (
        mapping.nero_gripper_width_max - mapping.nero_gripper_width_min
    )
    return float(np.clip(width, mapping.nero_gripper_width_min, mapping.nero_gripper_width_max))


def map_so101_joints_to_nero(so101_joint_deg: np.ndarray, *, mapping: SO101ToNeroMapping) -> np.ndarray:
    so101_joint_deg = np.asarray(so101_joint_deg, dtype=float)
    if so101_joint_deg.shape != (7,):
        raise ValueError(f"Expected 7 SO101 joint values, got shape {so101_joint_deg.shape}.")

    q_rel_deg = so101_joint_deg - mapping.so101_zero_deg
    q_rel_deg = q_rel_deg[[0, 1, 2, 3, 4, 6, 5]]
    q_rel_rad = np.deg2rad(q_rel_deg)
    target = mapping.nero_zero_rad + mapping.joint_scale * mapping.joint_direction * q_rel_rad
    return np.clip(target, mapping.nero_limit_low, mapping.nero_limit_high)


def map_so101_action_to_nero(
    action: dict[str, float], *, mapping: SO101ToNeroMapping, arm: str
) -> dict[str, float]:
    so101_values = action_to_so101_array(action)
    nero_joints = map_so101_joints_to_nero(so101_values[:7], mapping=mapping)
    gripper_width = map_so101_gripper_to_width(so101_values[7], mapping=mapping)

    mapped = {
        name: float(value) for name, value in zip(namespaced_joint_names(arm), nero_joints, strict=True)
    }
    mapped[namespaced_gripper_name(arm)] = gripper_width
    return mapped
