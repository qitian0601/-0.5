import json

import numpy as np

from lerobot_robot_nero.async_client import DEFAULT_DUAL_ACTION_NAMES
from lerobot_robot_nero.replay_inference_trace import load_trace_replay_actions, replay_trace_actions


class FakeArm:
    def __init__(self):
        self.joints = np.zeros(7)
        self.config = type(
            "Config",
            (),
            {"command": type("Command", (), {"alpha": 0.8, "max_step_rad": 0.1})()},
        )()

    def read_joints(self):
        return self.joints


class FakeRobot:
    def __init__(self):
        self.right = FakeArm()
        self.left = FakeArm()
        self.sent = []

    def send_action(self, action):
        self.sent.append(dict(action))
        self.right.joints = np.asarray([action[f"right_nero_joint_{idx}"] for idx in range(1, 8)])
        self.left.joints = np.asarray([action[f"left_nero_joint_{idx}"] for idx in range(1, 8)])
        return action


def _action(offset: float) -> dict[str, float]:
    return {name: float(idx) + offset for idx, name in enumerate(DEFAULT_DUAL_ACTION_NAMES)}


def test_load_trace_replay_actions_reads_final_commands(tmp_path):
    path = tmp_path / "replay_actions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"action": _action(0.0)}),
                json.dumps({"action": _action(10.0)}),
            ]
        )
    )

    actions = load_trace_replay_actions(path)

    assert actions.shape == (2, 16)
    assert actions[1, 0] == 10.0
    assert actions[1, 15] == 25.0


def test_replay_trace_actions_sends_recorded_commands_without_extra_smoothing(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.replay_inference_trace.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot()
    actions = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, 0.01, 0.02],
            [2, 3, 4, 5, 6, 7, 8, -2, -3, -4, -5, -6, -7, -8, 0.03, 0.04],
        ],
        dtype=float,
    )

    replay_trace_actions(
        robot,
        actions,
        fps=180,
        takeover_time_s=0.0,
        takeover_dt_s=0.02,
    )

    assert robot.sent[-1]["right_nero_joint_1"] == 2.0
    assert robot.sent[-1]["left_nero_joint_7"] == -8.0
    assert robot.right.config.command.alpha == 0.8
    assert robot.right.config.command.max_step_rad == 0.1
    assert sleeps[-1] <= 1.0 / 180
