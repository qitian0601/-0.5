#!/usr/bin/env python

import math

import torch

from lerobot.processor.ee_so3_relative_action_processor import (
    EEAbsoluteActionsProcessorStep,
    EELocalSE3AbsoluteActionsProcessorStep,
    EELocalSE3RelativeActionsProcessorStep,
    EERelativeActionsProcessorStep,
    rotvec_to_matrix,
    to_absolute_ee_local_se3_actions,
    to_absolute_ee_actions,
    to_relative_ee_local_se3_actions,
    to_relative_ee_actions,
)
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


def test_relative_then_absolute_roundtrip_for_action_chunk():
    state = torch.stack(
        [
            _ee_state(torch.tensor([0.0, 0.0, math.pi / 4])),
            _ee_state(torch.tensor([0.1, -0.2, 0.3]), torch.tensor([0.0, 0.1, 0.0])),
        ]
    )
    actions = torch.stack(
        [
            torch.stack(
                [
                    _ee_state(torch.tensor([0.0, 0.0, math.pi / 3])),
                    _ee_state(torch.tensor([0.1, 0.2, -0.1])),
                    _ee_state(torch.tensor([-0.2, 0.0, 0.3])),
                ]
            ),
            torch.stack(
                [
                    _ee_state(torch.tensor([0.2, -0.1, 0.4]), torch.tensor([0.1, 0.2, 0.0])),
                    _ee_state(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([-0.2, 0.2, 0.1])),
                    _ee_state(torch.tensor([0.3, -0.3, 0.1]), torch.tensor([0.4, 0.1, 0.0])),
                ]
            ),
        ]
    )

    relative = to_relative_ee_actions(actions, state)
    recovered = to_absolute_ee_actions(relative, state)

    torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)


def test_ee_local_se3_relative_uses_end_effector_local_translation_and_base_delta():
    state = _ee_local_se3_state(torch.tensor([0.0, 0.0, math.pi / 2]), base_xy=(1.0, 2.0)).unsqueeze(0)
    target = state.clone().view(1, 1, 16)
    target[:, :, 0:3] = state[:, None, 0:3] + torch.tensor([0.0, 1.3, 0.0])
    target[:, :, 3:6] = torch.tensor([0.0, 0.0, math.pi])
    target[:, :, 14:16] = torch.tensor([1.25, 1.5])

    relative = to_relative_ee_local_se3_actions(target, state)

    # Current right EE has a +90deg yaw, so a +1.3 world-y displacement becomes +1.3 local-x.
    torch.testing.assert_close(relative[:, :, 0:3], torch.tensor([[[1.3, 0.0, 0.0]]]), atol=1e-5, rtol=1e-5)
    expected_right_delta_rot = rotvec_to_matrix(state[:, 3:6]).transpose(-1, -2) @ rotvec_to_matrix(
        target[:, 0, 3:6]
    )
    torch.testing.assert_close(
        rotvec_to_matrix(relative[:, 0, 3:6]), expected_right_delta_rot, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(relative[:, :, 14:16], torch.tensor([[[0.25, -0.5]]]))


def test_ee_local_se3_relative_then_absolute_roundtrip_for_action_chunk():
    state = torch.stack(
        [
            _ee_local_se3_state(torch.tensor([0.0, 0.0, math.pi / 4]), base_xy=(0.0, 0.0)),
            _ee_local_se3_state(
                torch.tensor([0.1, -0.2, 0.3]),
                torch.tensor([0.0, 0.1, 0.0]),
                base_xy=(0.5, -0.5),
            ),
        ]
    )
    actions = torch.stack(
        [
            torch.stack(
                [
                    _ee_local_se3_state(torch.tensor([0.0, 0.0, math.pi / 3]), base_xy=(0.1, 0.2)),
                    _ee_local_se3_state(torch.tensor([0.1, 0.2, -0.1]), base_xy=(0.2, 0.4)),
                    _ee_local_se3_state(torch.tensor([-0.2, 0.0, 0.3]), base_xy=(0.3, 0.6)),
                ]
            ),
            torch.stack(
                [
                    _ee_local_se3_state(
                        torch.tensor([0.2, -0.1, 0.4]),
                        torch.tensor([0.1, 0.2, 0.0]),
                        base_xy=(0.55, -0.45),
                    ),
                    _ee_local_se3_state(
                        torch.tensor([0.0, 0.0, 0.0]),
                        torch.tensor([-0.2, 0.2, 0.1]),
                        base_xy=(0.6, -0.4),
                    ),
                    _ee_local_se3_state(
                        torch.tensor([0.3, -0.3, 0.1]),
                        torch.tensor([0.4, 0.1, 0.0]),
                        base_xy=(0.65, -0.35),
                    ),
                ]
            ),
        ]
    )

    relative = to_relative_ee_local_se3_actions(actions, state)
    recovered = to_absolute_ee_local_se3_actions(relative, state)

    torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)


def test_relative_rotation_uses_so3_composition_not_component_subtraction():
    state = _ee_state(torch.tensor([math.pi / 2, 0.0, 0.0])).unsqueeze(0)
    target = _ee_state(torch.tensor([0.0, math.pi / 2, 0.0])).view(1, 1, 14)

    relative = to_relative_ee_actions(target, state)

    current_rot = rotvec_to_matrix(state[:, 3:6])
    target_rot = rotvec_to_matrix(target[:, 0, 3:6])
    expected_delta = target_rot @ current_rot.transpose(-1, -2)
    actual_delta = rotvec_to_matrix(relative[:, 0, 3:6])

    torch.testing.assert_close(actual_delta, expected_delta, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(relative[:, 0, 3:6], target[:, 0, 3:6] - state[:, 3:6])


def test_gripper_dimensions_stay_absolute():
    state = _ee_state(torch.zeros(3)).unsqueeze(0)
    action = _ee_state(torch.zeros(3)).view(1, 1, 14)
    action[:, :, 6] = 0.08
    action[:, :, 13] = 0.09

    relative = to_relative_ee_actions(action, state)

    torch.testing.assert_close(relative[:, :, 6], action[:, :, 6])
    torch.testing.assert_close(relative[:, :, 13], action[:, :, 13])


def test_processor_steps_cache_state_and_reverse_actions():
    state = _ee_state(torch.tensor([0.0, 0.0, 0.2])).unsqueeze(0)
    action = _ee_state(torch.tensor([0.0, 0.3, 0.2])).view(1, 1, 14)
    relative_step = EERelativeActionsProcessorStep(enabled=True)
    absolute_step = EEAbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
    }
    relative_transition = relative_step(transition)
    absolute_transition = absolute_step(relative_transition)

    torch.testing.assert_close(absolute_transition[TransitionKey.ACTION], action, atol=1e-5, rtol=1e-5)


def test_ee_local_se3_processor_steps_cache_state_and_reverse_actions():
    state = _ee_local_se3_state(torch.tensor([0.0, 0.0, 0.2]), base_xy=(0.1, 0.2)).unsqueeze(0)
    action = _ee_local_se3_state(torch.tensor([0.0, 0.3, 0.2]), base_xy=(0.3, 0.1)).view(1, 1, 16)
    relative_step = EELocalSE3RelativeActionsProcessorStep(enabled=True)
    absolute_step = EELocalSE3AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
    }
    relative_transition = relative_step(transition)
    absolute_transition = absolute_step(relative_transition)

    torch.testing.assert_close(absolute_transition[TransitionKey.ACTION], action, atol=1e-5, rtol=1e-5)


def test_pi05_processor_selects_ee_so3_relative_step():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    config = PI05Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        use_relative_actions=True,
        relative_action_type="ee_so3",
        device="cpu",
    )
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=None)

    assert any(isinstance(step, EERelativeActionsProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, EEAbsoluteActionsProcessorStep) for step in postprocessor.steps)
    relative_step = next(step for step in preprocessor.steps if isinstance(step, EERelativeActionsProcessorStep))
    absolute_step = next(step for step in postprocessor.steps if isinstance(step, EEAbsoluteActionsProcessorStep))
    assert absolute_step.relative_step is relative_step


def test_pi05_processor_selects_ee_local_se3_relative_step():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    config = PI05Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        use_relative_actions=True,
        relative_action_type="ee_local_se3",
        device="cpu",
    )
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=None)

    assert any(isinstance(step, EELocalSE3RelativeActionsProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, EELocalSE3AbsoluteActionsProcessorStep) for step in postprocessor.steps)
    relative_step = next(
        step for step in preprocessor.steps if isinstance(step, EELocalSE3RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in postprocessor.steps if isinstance(step, EELocalSE3AbsoluteActionsProcessorStep)
    )
    assert absolute_step.relative_step is relative_step


def test_pi05_drops_tail_frames_that_would_pad_action_chunks():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PI05Config(chunk_size=4, n_action_steps=4)

    assert config.drop_n_last_frames == 3


def test_pi05_ee_local_se3_requires_16d_state_and_action_features():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PI05Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        use_relative_actions=True,
        relative_action_type="ee_local_se3",
        device="cpu",
    )

    try:
        config.validate_features()
    except ValueError as exc:
        assert "requires observation.state and action to both be 16D" in str(exc)
    else:
        raise AssertionError("Expected ee_local_se3 feature validation to reject non-16D data.")
