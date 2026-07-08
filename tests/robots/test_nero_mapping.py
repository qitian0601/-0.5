import draccus
import numpy as np
import pytest

from lerobot_robot_nero.async_client import NeroAsyncClientConfig
import pytest

from lerobot_robot_nero.config_nero import NeroCommandConfig, NeroDualRobotConfig, NeroMappingConfig
from lerobot_robot_nero.mapping import SO101_NAMES, SO101ToNeroMapping, map_so101_action_to_nero


def make_action(values: list[float]) -> dict[str, float]:
    return {f"{name}.pos": value for name, value in zip(SO101_NAMES, values, strict=True)}


def test_default_joint_direction_matches_reference_movejs_script():
    expected = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    assert np.allclose(SO101ToNeroMapping().joint_direction, expected)
    assert NeroMappingConfig().joint_direction == expected.tolist()


def test_dual_defaults_match_reference_movejs_pair_binding():
    cfg = NeroDualRobotConfig()

    assert cfg.right.connection.channel == "nero_right"
    assert cfg.right.connection.speed_percent == 100
    assert cfg.right.mapping.arm == "right"
    assert cfg.right.mapping.joint_direction == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    assert cfg.right.mapping.so101_zero_deg == [
        -4.351648,
        -0.175824,
        -8.571429,
        -7.208791,
        -7.736264,
        -10.193407,
        -0.457143,
    ]
    assert cfg.right.mapping.so101_gripper_min_deg == 13.957055
    assert cfg.right.mapping.so101_gripper_max_deg == 45.398773
    assert not cfg.right.mapping.gripper_reverse

    assert cfg.left.connection.channel == "nero_left"
    assert cfg.left.connection.speed_percent == 100
    assert cfg.left.mapping.arm == "left"
    assert cfg.left.mapping.joint_direction == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    assert cfg.left.mapping.so101_zero_deg == [
        -5.142857,
        0.527473,
        -1.802198,
        -3.384615,
        -7.868132,
        -8.967033,
        1.230769,
    ]
    assert cfg.left.mapping.so101_gripper_min_deg == 70.451436
    assert cfg.left.mapping.so101_gripper_max_deg == 94.304965
    assert cfg.left.mapping.gripper_reverse
    assert cfg.right.command.alpha == 0.8
    assert cfg.left.command.alpha == 0.8
    assert cfg.right.command.max_step_rad == float("inf")
    assert cfg.left.command.max_step_rad == float("inf")
    assert cfg.right.command.control_dt_s == pytest.approx(1.0 / 180.0)
    assert cfg.left.command.control_dt_s == pytest.approx(1.0 / 180.0)


def test_zero_pose_maps_to_nero_zero_and_min_gripper_width():
    mapping = SO101ToNeroMapping()
    action = make_action([*mapping.so101_zero_deg.tolist(), mapping.so101_gripper_min_deg])

    mapped = map_so101_action_to_nero(action, mapping=mapping, arm="right")

    assert [*mapped] == [
        "right_nero_joint_1",
        "right_nero_joint_2",
        "right_nero_joint_3",
        "right_nero_joint_4",
        "right_nero_joint_5",
        "right_nero_joint_6",
        "right_nero_joint_7",
        "right_gripper_width",
    ]
    assert np.allclose([mapped[f"right_nero_joint_{idx}"] for idx in range(1, 8)], np.zeros(7))
    assert mapped["right_gripper_width"] == 0.0


def test_joint_6_and_7_are_swapped_before_direction_and_scale():
    mapping = SO101ToNeroMapping(
        so101_zero_deg=np.zeros(7),
        nero_zero_rad=np.zeros(7),
        joint_scale=np.ones(7),
        joint_direction=np.ones(7),
        nero_limit_low=np.full(7, -10.0),
        nero_limit_high=np.full(7, 10.0),
    )
    action = make_action([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, mapping.so101_gripper_min_deg])

    mapped = map_so101_action_to_nero(action, mapping=mapping, arm="right")

    assert np.isclose(mapped["right_nero_joint_6"], np.deg2rad(20.0))
    assert np.isclose(mapped["right_nero_joint_7"], np.deg2rad(10.0))


def test_joint_targets_are_clipped_to_nero_limits():
    mapping = SO101ToNeroMapping(
        so101_zero_deg=np.zeros(7),
        nero_zero_rad=np.zeros(7),
        joint_scale=np.ones(7),
        joint_direction=np.ones(7),
        nero_limit_low=np.full(7, -0.5),
        nero_limit_high=np.full(7, 0.5),
    )
    action = make_action([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, mapping.so101_gripper_min_deg])

    mapped = map_so101_action_to_nero(action, mapping=mapping, arm="right")

    assert np.allclose([mapped[f"right_nero_joint_{idx}"] for idx in range(1, 8)], np.full(7, 0.5))


def test_gripper_maps_to_width_range_and_clips():
    mapping = SO101ToNeroMapping(
        so101_gripper_min_deg=5.0,
        so101_gripper_max_deg=25.0,
        nero_gripper_width_min=0.0,
        nero_gripper_width_max=0.1,
    )

    mid = map_so101_action_to_nero(make_action([0, 0, 0, 0, 0, 0, 0, 15.0]), mapping=mapping, arm="right")
    low = map_so101_action_to_nero(make_action([0, 0, 0, 0, 0, 0, 0, -100.0]), mapping=mapping, arm="right")
    high = map_so101_action_to_nero(make_action([0, 0, 0, 0, 0, 0, 0, 100.0]), mapping=mapping, arm="right")

    assert np.isclose(mid["right_gripper_width"], 0.05)
    assert np.isclose(low["right_gripper_width"], 0.0)
    assert np.isclose(high["right_gripper_width"], 0.1)


def test_nero_command_config_decodes_move_method_from_draccus():
    cfg = draccus.decode(NeroCommandConfig, {"move_method": "move_j"})

    assert cfg.move_method == "move_j"


def test_dual_client_partial_command_override_preserves_pair_binding():
    cfg = draccus.decode(
        NeroAsyncClientConfig,
        {
            "robot": {
                "type": "nero_dual",
                "right": {"command": {"move_method": "move_j"}},
                "left": {"command": {"move_method": "move_j"}},
            }
        },
    )

    assert cfg.robot.right.connection.channel == "nero_right"
    assert cfg.robot.right.connection.speed_percent == 100
    assert cfg.robot.right.mapping.arm == "right"
    assert cfg.robot.right.command.move_method == "move_j"
    assert cfg.robot.left.connection.channel == "nero_left"
    assert cfg.robot.left.connection.speed_percent == 100
    assert cfg.robot.left.mapping.arm == "left"
    assert cfg.robot.left.command.move_method == "move_j"


def test_dual_client_partial_connection_override_preserves_pair_binding():
    cfg = draccus.decode(
        NeroAsyncClientConfig,
        {
            "robot": {
                "type": "nero_dual",
                "right": {
                    "connection": {"speed_percent": 20},
                    "command": {"move_method": "move_j"},
                },
                "left": {
                    "connection": {"speed_percent": 20},
                    "command": {"move_method": "move_j"},
                },
            }
        },
    )

    assert cfg.robot.right.connection.channel == "nero_right"
    assert cfg.robot.right.connection.speed_percent == 20
    assert cfg.robot.right.mapping.arm == "right"
    assert cfg.robot.right.command.move_method == "move_j"
    assert cfg.robot.left.connection.channel == "nero_left"
    assert cfg.robot.left.connection.speed_percent == 20
    assert cfg.robot.left.mapping.arm == "left"
    assert cfg.robot.left.command.move_method == "move_j"


def test_nero_command_config_rejects_unknown_move_method():
    with pytest.raises(ValueError, match="move_method"):
        NeroCommandConfig(move_method="move_l")
