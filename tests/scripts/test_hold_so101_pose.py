from tools.hold_so101_pose import (
    abs_errors,
    all_torque_enabled,
    format_hold_status,
    make_position_command,
)


def test_make_position_command_filters_selected_motors():
    action = {
        "joint_1.pos": 1.0,
        "joint_2.pos": 2.0,
        "gripper.pos": 50.0,
    }

    assert make_position_command(action, ["joint_2", "gripper"]) == {
        "joint_2": 2.0,
        "gripper": 50.0,
    }


def test_abs_errors_uses_present_minus_goal_abs():
    goal = {"joint_1": 10.0, "joint_2": -3.0}
    present = {"joint_1": 7.5, "joint_2": -2.0}

    assert abs_errors(goal, present) == {"joint_1": 2.5, "joint_2": 1.0}


def test_all_torque_enabled_accepts_numeric_and_bool_values():
    assert all_torque_enabled({"joint_1": 1, "joint_2": True}, ["joint_1", "joint_2"])
    assert not all_torque_enabled({"joint_1": 1, "joint_2": 0}, ["joint_1", "joint_2"])


def test_format_hold_status_includes_torque_goal_present_error():
    status = format_hold_status(
        goal={"joint_1": 10.0},
        present={"joint_1": 9.5},
        torque={"joint_1": 1},
    )

    assert "joint_1" in status
    assert "torque=1" in status
    assert "goal=  10.000" in status
    assert "present=   9.500" in status
    assert "err=   0.500" in status
