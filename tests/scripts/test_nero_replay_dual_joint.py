from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot_robot_nero.replay_dual_joint import (
    action_dict_from_vector,
    load_episode_actions,
    replay_episode_actions,
)
from lerobot_robot_nero.async_client import NeroAsyncSafetyConfig, SafeNeroRobot


ACTION_NAMES = [
    "right_nero_joint_1",
    "right_nero_joint_2",
    "right_nero_joint_3",
    "right_nero_joint_4",
    "right_nero_joint_5",
    "right_nero_joint_6",
    "right_nero_joint_7",
    "left_nero_joint_1",
    "left_nero_joint_2",
    "left_nero_joint_3",
    "left_nero_joint_4",
    "left_nero_joint_5",
    "left_nero_joint_6",
    "left_nero_joint_7",
    "right_gripper_width",
    "left_gripper_width",
]


def _write_dataset(root: Path) -> None:
    data_dir = root / "data/chunk-000"
    data_dir.mkdir(parents=True)
    rows = []
    for episode in range(3):
        for frame in range(2):
            base = episode * 100 + frame * 10
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "action": np.arange(base, base + 16, dtype=np.float32),
                }
            )
    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet")


def test_load_episode_actions_selects_requested_episode(tmp_path):
    _write_dataset(tmp_path)

    actions = load_episode_actions(tmp_path, episode=1)

    assert actions.shape == (2, 16)
    assert np.allclose(actions[0], np.arange(100, 116, dtype=np.float32))
    assert np.allclose(actions[1], np.arange(110, 126, dtype=np.float32))


def test_load_episode_actions_rejects_missing_episode(tmp_path):
    _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="Episode 5 not found"):
        load_episode_actions(tmp_path, episode=5)


def test_action_dict_from_vector_uses_dataset_feature_order():
    action = action_dict_from_vector(np.arange(16, dtype=np.float32), ACTION_NAMES)

    assert action["right_nero_joint_1"] == 0.0
    assert action["left_nero_joint_1"] == 7.0
    assert action["right_gripper_width"] == 14.0
    assert action["left_gripper_width"] == 15.0


class FakeArm:
    def __init__(self, arm: str):
        self.arm = arm
        self.joints = np.zeros(7)
        self.config = type(
            "Config",
            (),
            {"command": type("Command", (), {"alpha": 0.8, "max_step_rad": float("inf")})()},
        )()

    def read_joints(self):
        return self.joints


class FakeRobot:
    def __init__(self):
        self.right = FakeArm("right")
        self.left = FakeArm("left")
        self.sent = []
        self.control_dt_s = 0.006

    def send_action(self, action):
        self.sent.append(action)
        self.right.joints = np.asarray([action[f"right_nero_joint_{i}"] for i in range(1, 8)])
        self.left.joints = np.asarray([action[f"left_nero_joint_{i}"] for i in range(1, 8)])
        return action


def test_replay_episode_actions_smooths_to_first_frame_then_replays(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.replay_dual_joint.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot()
    actions = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, 0.01, 0.02],
            [2, 3, 4, 5, 6, 7, 8, -2, -3, -4, -5, -6, -7, -8, 0.03, 0.04],
        ],
        dtype=np.float32,
    )

    replay_episode_actions(
        robot,
        actions,
        ACTION_NAMES,
        fps=30,
        takeover_time_s=0.04,
        takeover_dt_s=0.02,
    )

    assert len(robot.sent) == 4
    assert robot.sent[-1]["right_nero_joint_1"] == 2.0
    assert robot.sent[-1]["left_nero_joint_7"] == -8.0
    assert sleeps[:2] == [0.02, 0.02]


def test_replay_episode_actions_can_update_high_rate_target_without_direct_replay_sends(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lerobot_robot_nero.replay_dual_joint.precise_sleep", lambda value: sleeps.append(value))
    robot = FakeRobot()
    safe_robot = SafeNeroRobot(
        robot,
        NeroAsyncSafetyConfig(
            fixed_ready_pose={name: 0.0 for name in ACTION_NAMES},
            high_rate_control=True,
            max_policy_step_rad=10.0,
            max_gripper_step_m=10.0,
            max_executor_step_rad=0.5,
            max_executor_gripper_step_m=0.5,
        ),
    )
    actions = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, 0.01, 0.02],
            [2, 3, 4, 5, 6, 7, 8, -2, -3, -4, -5, -6, -7, -8, 0.03, 0.04],
        ],
        dtype=np.float32,
    )

    replay_episode_actions(
        safe_robot,
        actions,
        ACTION_NAMES,
        fps=30,
        takeover_time_s=0.04,
        takeover_dt_s=0.02,
        high_rate_control=True,
    )

    assert len(robot.sent) == 2
    safe_robot.execute_high_rate_step()
    assert robot.sent[-1]["right_nero_joint_1"] == 1.5
