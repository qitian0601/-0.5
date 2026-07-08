import pandas as pd

from lerobot_robot_nero.config_nero import NeroTrimConfig
from lerobot_robot_nero.postprocess_trim_dual_joint_dataset import (
    format_trim_summary,
    make_episode_trim_plans,
)


class HFDataset:
    def __init__(self, actions):
        self.actions = actions

    def select_columns(self, column):
        assert column == "action"
        return self

    def __getitem__(self, item):
        return {"action": self.actions[item]}


class Meta:
    fps = 10
    total_episodes = 2
    total_frames = 50

    def __init__(self):
        self.episodes = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "tasks": [["fold_towel"], ["fold_towel"]],
                "dataset_from_index": [0, 25],
                "dataset_to_index": [25, 50],
            }
        )


class Dataset:
    def __init__(self, actions):
        self.meta = Meta()
        self.hf_dataset = HFDataset(actions)


def action(right=0.0, left=0.0):
    return [right] * 7 + [0.0] + [left] * 7 + [0.0]


def test_make_episode_trim_plans_uses_saved_action_arrays():
    actions = []
    actions += [action() for _ in range(10)]
    actions += [action(left=idx * 0.01) for idx in range(1, 6)]
    actions += [action(left=0.05) for _ in range(10)]
    actions += [action() for _ in range(25)]
    dataset = Dataset(actions)
    config = NeroTrimConfig(static_time_s=0.3, preroll_s=0.1, postroll_s=0.1, min_episode_frames=1)

    plans, summary = make_episode_trim_plans(dataset, config=config)

    assert len(plans) == 1
    assert plans[0].source_episode_index == 0
    assert plans[0].source_from_index == 8
    assert plans[0].source_to_index == 16
    assert plans[0].output_episode_index == 0
    assert plans[0].task == "fold_towel"
    assert summary.original_episodes == 2
    assert summary.original_frames == 50
    assert summary.processed_episodes == 1
    assert summary.processed_frames == 8
    assert summary.rejected_episodes == 1


def test_format_trim_summary_prints_basic_counts():
    actions = [action() for _ in range(50)]
    _, summary = make_episode_trim_plans(Dataset(actions), config=NeroTrimConfig(min_episode_frames=1))

    text = format_trim_summary(summary)

    assert "原数据条数: 2" in text
    assert "原数据总帧数: 50" in text
    assert "处理后数据条数: 0" in text
    assert "处理后总帧数: 0" in text
