#!/usr/bin/env python

import math
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _ee_state(right_rotvec: torch.Tensor, left_rotvec: torch.Tensor | None = None) -> torch.Tensor:
    if left_rotvec is None:
        left_rotvec = torch.zeros(3)
    return torch.tensor(
        [
            0.2,
            -0.1,
            0.4,
            *right_rotvec.tolist(),
            0.03,
            -0.2,
            0.1,
            0.5,
            *left_rotvec.tolist(),
            0.04,
        ],
        dtype=torch.float32,
    )


def _ee_local_se3_state(
    right_rotvec: torch.Tensor,
    left_rotvec: torch.Tensor | None = None,
    base_xy: tuple[float, float] = (0.1, -0.2),
) -> torch.Tensor:
    return torch.cat((_ee_state(right_rotvec, left_rotvec), torch.tensor(base_xy, dtype=torch.float32)))


def test_ee_local_se3_relative_processor_roundtrips_absolute_16d_action_chunk():
    from lerobot.processor import (
        EELocalSE3AbsoluteActionsProcessorStep,
        EELocalSE3RelativeActionsProcessorStep,
    )

    state = _ee_local_se3_state(torch.tensor([0.0, 0.0, math.pi / 4]), base_xy=(0.0, 0.0)).unsqueeze(0)
    actions = torch.stack(
        [
            _ee_local_se3_state(torch.tensor([0.0, 0.0, math.pi / 3]), base_xy=(0.1, 0.2)),
            _ee_local_se3_state(torch.tensor([0.1, 0.2, -0.1]), base_xy=(0.2, 0.4)),
            _ee_local_se3_state(torch.tensor([-0.2, 0.0, 0.3]), base_xy=(0.3, 0.6)),
        ]
    ).unsqueeze(0)

    relative_step = EELocalSE3RelativeActionsProcessorStep(enabled=True)
    absolute_step = EELocalSE3AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: actions,
    }

    relative_transition = relative_step(transition)
    recovered_transition = absolute_step(relative_transition)

    torch.testing.assert_close(recovered_transition[TransitionKey.ACTION], actions, atol=1e-5, rtol=1e-5)


def test_downloaded_fold_towel_ee_checkpoint_processors_reconnect_and_output_absolute_16d(monkeypatch):
    checkpoint_dir = Path(
        "/home/chenglong/workplace/nero_teleop_ws/lerobot/outputs/train/"
        "pi05_fold_towel_ee_bryce301best/pi05_fold_towel_ee_local_se3_rel_8gpu/"
        "checkpoints/020000/pretrained_model"
    )
    assert checkpoint_dir.exists()

    class DummyTokenizer:
        padding_side = "right"

        def __call__(self, texts, **kwargs):
            batch = len(texts)
            max_length = kwargs.get("max_length", 200)
            return {
                "input_ids": torch.zeros((batch, max_length), dtype=torch.long),
                "attention_mask": torch.ones((batch, max_length), dtype=torch.long),
            }

    import lerobot.processor.tokenizer_processor as tokenizer_processor

    monkeypatch.setattr(tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=checkpoint_dir)

    from lerobot.processor import EELocalSE3AbsoluteActionsProcessorStep, EELocalSE3RelativeActionsProcessorStep

    relative_step = next(step for step in preprocessor.steps if isinstance(step, EELocalSE3RelativeActionsProcessorStep))
    absolute_step = next(
        step for step in postprocessor.steps if isinstance(step, EELocalSE3AbsoluteActionsProcessorStep)
    )
    assert absolute_step.relative_step is relative_step

    state = _ee_local_se3_state(torch.tensor([0.0, 0.0, 0.2]), base_xy=(0.1, 0.2)).unsqueeze(0)
    absolute_actions = _ee_local_se3_state(torch.tensor([0.0, 0.3, 0.2]), base_xy=(0.3, 0.1)).view(
        1, 1, 16
    )
    relative_transition = relative_step(
        {
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: absolute_actions,
        }
    )
    recovered = absolute_step({TransitionKey.ACTION: PolicyAction(relative_transition[TransitionKey.ACTION])})

    torch.testing.assert_close(recovered[TransitionKey.ACTION], absolute_actions, atol=1e-5, rtol=1e-5)


def test_pi05_default_relative_action_type_preserves_joint_processors(monkeypatch):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep

    import lerobot.policies.pi05.processor_pi05 as processor_pi05
    from lerobot.processor import ProcessorStep

    class DummyTokenizerProcessor(ProcessorStep):
        def __call__(self, transition):
            return transition

        def transform_features(self, features):
            return features

    monkeypatch.setattr(processor_pi05, "TokenizerProcessorStep", lambda **kwargs: DummyTokenizerProcessor())

    config = PI05Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_feature_names=["right_nero_joint_1", "right_gripper_width"],
        device="cpu",
    )

    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=None)

    assert any(isinstance(step, RelativeActionsProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, AbsoluteActionsProcessorStep) for step in postprocessor.steps)
