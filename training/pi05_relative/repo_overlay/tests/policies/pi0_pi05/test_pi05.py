#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.utils.constants import ACTION_MASK, OBS_IMAGES_MASK
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_hf_token  # noqa: E402


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise


def test_pi05_action_loss_mask_indices_reduce_over_unmasked_dims():
    config = PI05Config()
    config.action_loss_mask_indices = [2, 4]

    losses = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    masked_losses, valid_dims = PI05Policy._apply_action_loss_mask(losses, config, batch={})

    expected = losses.clone()
    expected[:, :, [2, 4]] = 0
    torch.testing.assert_close(masked_losses, expected)
    torch.testing.assert_close(valid_dims, torch.tensor([9.0, 9.0]))

    per_sample_loss = masked_losses.sum(dim=(1, 2)) / valid_dims.clamp_min(1)
    torch.testing.assert_close(per_sample_loss, expected.sum(dim=(1, 2)) / 9)


def test_pi05_sample_action_mask_keeps_robot_gripper_and_masks_human_gripper():
    config = PI05Config()
    losses = torch.ones(2, 2, 14, dtype=torch.float32)
    action_mask = torch.ones(2, 14, dtype=torch.bool)
    action_mask[1, [6, 13]] = False

    masked_losses, valid_counts = PI05Policy._apply_action_loss_mask(
        losses, config, batch={ACTION_MASK: action_mask}
    )

    expected = losses.clone()
    expected[1, :, [6, 13]] = 0
    torch.testing.assert_close(masked_losses, expected)
    torch.testing.assert_close(valid_counts, torch.tensor([28.0, 24.0]))


def test_pi05_sample_action_mask_accepts_chunk_masks():
    config = PI05Config()
    losses = torch.ones(2, 3, 4, dtype=torch.float32)
    action_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    action_mask[0, :, 1] = False
    action_mask[1, 2, :] = False

    masked_losses, valid_counts = PI05Policy._apply_action_loss_mask(
        losses, config, batch={ACTION_MASK: action_mask}
    )

    assert masked_losses[0, :, 1].sum().item() == 0
    assert masked_losses[1, 2, :].sum().item() == 0
    torch.testing.assert_close(valid_counts, torch.tensor([9.0, 8.0]))


def test_pi05_preprocess_images_uses_per_sample_image_mask_for_missing_wrist_views():
    config = PI05Config()
    config.input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = config
    parameter = torch.nn.Parameter(torch.empty(0))
    policy.parameters = lambda: iter([parameter])

    batch = {
        "observation.images.front": torch.full((2, 3, 16, 16), 0.75),
        "observation.images.left_wrist": torch.full((2, 3, 16, 16), 0.25),
        "observation.images.right_wrist": torch.full((2, 3, 16, 16), 0.5),
        OBS_IMAGES_MASK: torch.tensor(
            [
                [True, True, True],
                [True, False, False],
            ]
        ),
    }

    images, img_masks = policy._preprocess_images(batch)

    assert [mask.tolist() for mask in img_masks] == [[True, True], [True, False], [True, False]]
    assert torch.all(images[1][1] == -1)
    assert torch.all(images[2][1] == -1)


def test_pi05_preprocess_images_rejects_missing_image_marked_valid():
    config = PI05Config()
    config.input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = config
    parameter = torch.nn.Parameter(torch.empty(0))
    policy.parameters = lambda: iter([parameter])

    batch = {
        "observation.images.front": torch.full((1, 3, 16, 16), 0.75),
        OBS_IMAGES_MASK: torch.tensor([[True, True]]),
    }

    with pytest.raises(ValueError, match="marked valid"):
        policy._preprocess_images(batch)
