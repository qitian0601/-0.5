from lerobot_teleoperator_so101_8dof.config_so101_8dof import SO1018DofLeaderConfig
import lerobot_teleoperator_so101_8dof.so101_8dof as so101_8dof_module
from lerobot_teleoperator_so101_8dof.so101_8dof import SO1018DofLeader
from lerobot.motors import MotorNormMode


class FakeBus:
    def __init__(self, *, port, motors, calibration):
        del port, calibration
        self.motors = motors
        self.is_connected = False


def test_so101_8dof_action_features_are_joint_1_to_7_plus_gripper(monkeypatch):
    monkeypatch.setattr(so101_8dof_module, "FeetechMotorsBus", FakeBus)
    leader = SO1018DofLeader(SO1018DofLeaderConfig(port="/dev/null"))

    assert list(leader.action_features) == [
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "joint_7.pos",
        "gripper.pos",
    ]


def test_so101_8dof_gripper_uses_reference_range_norm_mode(monkeypatch):
    monkeypatch.setattr(so101_8dof_module, "FeetechMotorsBus", FakeBus)
    leader = SO1018DofLeader(SO1018DofLeaderConfig(port="/dev/null", use_degrees=True))

    assert leader.bus.motors["joint_1"].norm_mode is MotorNormMode.DEGREES
    assert leader.bus.motors["gripper"].norm_mode is MotorNormMode.RANGE_0_100
