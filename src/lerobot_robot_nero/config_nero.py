from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class NeroConnectionConfig:
    channel: str = "can0"
    firmware_version: str = "V120"
    interface: str = "socketcan"
    speed_percent: int = 50
    gripper_force: float = 1.0
    enable_retry_s: float = 0.2
    reset_on_connect: bool = True


@dataclass
class NeroMappingConfig:
    arm: str = "right"
    joint_scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2])
    joint_direction: list[float] = field(default_factory=lambda: [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    so101_zero_deg: list[float] = field(
        default_factory=lambda: [
            4.087912,
            4.131868,
            0.307692,
            -6.065934,
            22.637363,
            -8.703297,
            1.494505,
        ]
    )
    nero_zero_rad: list[float] = field(default_factory=lambda: [0.0] * 7)
    nero_limit_low: list[float] = field(
        default_factory=lambda: [
            -2.705261,
            -1.745330,
            -2.757621,
            -1.012291,
            -2.757621,
            -0.733039,
            -1.570797,
        ]
    )
    nero_limit_high: list[float] = field(
        default_factory=lambda: [
            2.705261,
            1.745330,
            2.757621,
            2.146755,
            2.757621,
            0.959932,
            1.570797,
        ]
    )
    so101_gripper_min_deg: float = 6.185973
    so101_gripper_max_deg: float = 27.029157
    nero_gripper_width_min: float = 0.0
    nero_gripper_width_max: float = 0.1
    gripper_reverse: bool = False


def make_right_pair_connection_config() -> NeroConnectionConfig:
    return NeroConnectionConfig(channel="nero_right", speed_percent=100)


def make_left_pair_connection_config() -> NeroConnectionConfig:
    return NeroConnectionConfig(channel="nero_left", speed_percent=100)


def make_right_pair_mapping_config() -> NeroMappingConfig:
    return NeroMappingConfig(
        arm="right",
        joint_scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2],
        joint_direction=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
        so101_zero_deg=[
            -4.351648,
            -0.175824,
            -8.571429,
            -7.208791,
            -7.736264,
            -10.193407,
            -0.457143,
        ],
        so101_gripper_min_deg=13.957055,
        so101_gripper_max_deg=45.398773,
        gripper_reverse=False,
    )


def make_left_pair_mapping_config() -> NeroMappingConfig:
    return NeroMappingConfig(
        arm="left",
        joint_scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2],
        joint_direction=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
        so101_zero_deg=[
            -5.142857,
            0.527473,
            -1.802198,
            -3.384615,
            -7.868132,
            -8.967033,
            1.230769,
        ],
        so101_gripper_min_deg=70.451436,
        so101_gripper_max_deg=94.304965,
        gripper_reverse=True,
    )


@dataclass
class NeroCommandConfig:
    alpha: float = 0.80
    max_step_rad: float = float("inf")
    control_dt_s: float = 1.0 / 180.0
    move_method: str = "move_js"

    def __post_init__(self) -> None:
        if self.move_method not in {"move_js", "move_j"}:
            raise ValueError(f"Unsupported Nero move_method: {self.move_method!r}. Use 'move_js' or 'move_j'.")


@dataclass
class NeroArmConfig:
    connection: NeroConnectionConfig = field(default_factory=NeroConnectionConfig)
    mapping: NeroMappingConfig = field(default_factory=NeroMappingConfig)
    command: NeroCommandConfig = field(default_factory=NeroCommandConfig)


def make_right_pair_arm_config() -> NeroArmConfig:
    return NeroArmConfig(
        connection=make_right_pair_connection_config(),
        mapping=make_right_pair_mapping_config(),
    )


def make_left_pair_arm_config() -> NeroArmConfig:
    return NeroArmConfig(
        connection=make_left_pair_connection_config(),
        mapping=make_left_pair_mapping_config(),
    )


def _connection_only_has_identity_overrides(connection: NeroConnectionConfig) -> bool:
    generic_connection = NeroConnectionConfig()
    identity_fields = {"channel", "reset_on_connect"}
    return all(
        getattr(connection, field.name) == getattr(generic_connection, field.name)
        for field in fields(NeroConnectionConfig)
        if field.name not in identity_fields
    )


def _merge_pair_connection_defaults(
    connection: NeroConnectionConfig,
    pair_connection: NeroConnectionConfig,
) -> NeroConnectionConfig:
    generic_connection = NeroConnectionConfig()
    if connection == generic_connection:
        return pair_connection
    if _connection_only_has_identity_overrides(connection):
        channel = connection.channel if connection.channel != generic_connection.channel else pair_connection.channel
        return replace(pair_connection, channel=channel, reset_on_connect=connection.reset_on_connect)
    return connection


@dataclass
class NeroTrimConfig:
    joint_threshold_rad: float = 0.003
    gripper_threshold_m: float = 0.001
    static_time_s: float = 0.75
    preroll_s: float = 0.25
    postroll_s: float = 0.25
    min_episode_frames: int = 10


@RobotConfig.register_subclass("nero")
@dataclass
class NeroRobotConfig(RobotConfig):
    connection: NeroConnectionConfig = field(default_factory=NeroConnectionConfig)
    mapping: NeroMappingConfig = field(default_factory=NeroMappingConfig)
    command: NeroCommandConfig = field(default_factory=NeroCommandConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("nero_dual")
@dataclass
class NeroDualRobotConfig(RobotConfig):
    right: NeroArmConfig = field(default_factory=make_right_pair_arm_config)
    left: NeroArmConfig = field(default_factory=make_left_pair_arm_config)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        generic_connection = NeroConnectionConfig()
        generic_mapping = NeroMappingConfig()

        self.right.connection = _merge_pair_connection_defaults(
            self.right.connection,
            make_right_pair_connection_config(),
        )
        if self.right.mapping == generic_mapping:
            self.right.mapping = make_right_pair_mapping_config()
        if self.right.mapping.arm == "right" and self.right.connection.channel == generic_connection.channel:
            self.right.connection.channel = make_right_pair_connection_config().channel
        self.left.connection = _merge_pair_connection_defaults(
            self.left.connection,
            make_left_pair_connection_config(),
        )
        if self.left.mapping == generic_mapping:
            self.left.mapping = make_left_pair_mapping_config()
        if self.left.mapping.arm == "left" and self.left.connection.channel == generic_connection.channel:
            self.left.connection.channel = make_left_pair_connection_config().channel

        if self.right.mapping.arm != "right":
            raise ValueError(
                f"Nero dual right arm mapping.arm must be 'right', got {self.right.mapping.arm!r}."
            )
        if self.left.mapping.arm != "left":
            raise ValueError(
                f"Nero dual left arm mapping.arm must be 'left', got {self.left.mapping.arm!r}."
            )
