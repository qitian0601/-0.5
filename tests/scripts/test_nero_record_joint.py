import numpy as np

from lerobot_robot_nero.mapping import SO101ToNeroMapping
from lerobot_robot_nero.record_joint import (
    IdleTeleopDecision,
    flatten_episode_frame,
    idle_decision_from_text,
    run_teleop_step,
    should_stop_episode,
)


def test_should_stop_episode_when_manual_stop_requested():
    assert should_stop_episode(
        start_time_s=10.0,
        now_s=11.0,
        episode_time_s=60.0,
        manual_stop_requested=True,
    )


def test_should_stop_episode_when_timeout_reached():
    assert should_stop_episode(
        start_time_s=10.0,
        now_s=70.0,
        episode_time_s=60.0,
        manual_stop_requested=False,
    )


def test_should_not_stop_episode_before_manual_stop_or_timeout():
    assert not should_stop_episode(
        start_time_s=10.0,
        now_s=20.0,
        episode_time_s=60.0,
        manual_stop_requested=False,
    )


def test_flatten_episode_frame_builds_lerobot_state_action_vectors():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["right_nero_joint_1", "right_gripper_width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["right_nero_joint_1", "right_gripper_width"],
        },
    }
    frame = {
        "observation": {"right_nero_joint_1": 0.1, "right_gripper_width": 0.02},
        "action": {"right_nero_joint_1": 0.2, "right_gripper_width": 0.03},
    }

    flattened = flatten_episode_frame(frame, features=features, task="push")

    assert np.allclose(flattened["observation.state"], [0.1, 0.02])
    assert np.allclose(flattened["action"], [0.2, 0.03])
    assert flattened["task"] == "push"


def test_run_teleop_step_can_sync_without_recording():
    class Leader:
        def get_action(self):
            return {
                "joint_1.pos": 10.0,
                "joint_2.pos": 0.0,
                "joint_3.pos": 0.0,
                "joint_4.pos": 0.0,
                "joint_5.pos": 0.0,
                "joint_6.pos": 0.0,
                "joint_7.pos": 0.0,
                "gripper.pos": 50.0,
            }

    class Robot:
        arm = "right"

        def __init__(self):
            self.sent = []

        def get_observation(self):
            return {"right_nero_joint_1": 0.0, "right_gripper_width": 0.0}

        def send_action(self, action):
            self.sent.append(action)
            return action

    robot = Robot()
    frame = run_teleop_step(
        leader=Leader(),
        robot=robot,
        mapping=SO101ToNeroMapping(
            so101_zero_deg=np.zeros(7),
            nero_limit_low=np.full(7, -10.0),
            nero_limit_high=np.full(7, 10.0),
            so101_gripper_min_deg=0.0,
            so101_gripper_max_deg=100.0,
        ),
        record=False,
    )

    assert frame is None
    assert len(robot.sent) == 1
    assert np.isclose(robot.sent[0]["right_nero_joint_1"], -np.deg2rad(10.0))
    assert np.isclose(robot.sent[0]["right_gripper_width"], 0.05)


def test_idle_decision_from_text_starts_or_quits():
    assert idle_decision_from_text("") is IdleTeleopDecision.START_RECORDING
    assert idle_decision_from_text("   ") is IdleTeleopDecision.START_RECORDING
    assert idle_decision_from_text("q") is IdleTeleopDecision.FINISH_AND_EXIT
    assert idle_decision_from_text("Q") is IdleTeleopDecision.FINISH_AND_EXIT
    assert idle_decision_from_text("quit") is IdleTeleopDecision.FINISH_AND_EXIT
