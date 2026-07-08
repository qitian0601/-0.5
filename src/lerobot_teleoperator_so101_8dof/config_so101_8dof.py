from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class SO1018DofLeaderBaseConfig:
    port: str
    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("so101_8dof_leader")
@dataclass
class SO1018DofLeaderConfig(TeleoperatorConfig, SO1018DofLeaderBaseConfig):
    pass
