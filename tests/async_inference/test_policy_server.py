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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()
        self.reset_calls = 0

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_maybe_enqueue_observation_uses_configured_similarity_atol(policy_server):
    """A stricter similarity threshold lets small-but-real state changes through."""
    policy_server.config.obs_similarity_atol = 0.05
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    new_obs = _make_obs(torch.ones(6) * 0.1)

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_reset_server_clears_last_observation_and_policy_state(policy_server):
    policy_server.last_processed_obs = _make_obs(torch.zeros(6), timestep=7)
    policy_server.observation_queue.put(_make_obs(torch.ones(6), timestep=8))
    with policy_server._predicted_timesteps_lock:
        policy_server._predicted_timesteps = {7, 8}

    policy_server._reset_server()

    assert policy_server.last_processed_obs is None
    assert policy_server.observation_queue.empty()
    with policy_server._predicted_timesteps_lock:
        assert policy_server._predicted_timesteps == set()
    assert policy_server.policy.reset_calls == 1


def test_policy_server_config_from_dict_restores_async_rtc_config():
    """Dict round-trips must keep async_rtc as a config object, not a raw dict."""
    from lerobot.async_inference.configs import PolicyServerConfig

    original = PolicyServerConfig(
        host="localhost",
        port=9999,
        fps=30,
        inference_latency=0.1,
        obs_queue_timeout=1.0,
        obs_similarity_atol=0.15,
    )
    config_dict = original.to_dict()
    config_dict["async_rtc"] = {"enabled": True}

    config = PolicyServerConfig.from_dict(config_dict)

    assert config.async_rtc.enabled is True
    assert config.to_dict()["async_rtc"] == {
        "enabled": True,
        "latency_quantile": 0.95,
        "debug_dump": {"enabled": False},
    }


def test_policy_server_config_from_dict_restores_async_rtc_debug_dump():
    """RTC debug dump can be enabled from CLI/dict config."""
    from lerobot.async_inference.configs import PolicyServerConfig

    config = PolicyServerConfig.from_dict(
        {
            "host": "localhost",
            "port": 9999,
            "async_rtc": {"enabled": True, "debug_dump": {"enabled": True}},
        }
    )

    assert config.async_rtc.enabled is True
    assert config.async_rtc.debug_dump.enabled is True


def test_policy_server_config_from_dict_restores_async_rtc_latency_quantile():
    """RTC latency quantile can be configured from CLI/dict config."""
    from lerobot.async_inference.configs import PolicyServerConfig

    config = PolicyServerConfig.from_dict(
        {
            "host": "localhost",
            "port": 9999,
            "async_rtc": {"enabled": True, "latency_quantile": 0.9},
        }
    )

    assert config.async_rtc.enabled is True
    assert config.async_rtc.latency_quantile == 0.9
    assert config.to_dict()["async_rtc"]["latency_quantile"] == 0.9


def test_policy_server_config_from_dict_ignores_derived_environment_dt():
    """`environment_dt` is serialized for diagnostics, but fps remains the source of truth."""
    from lerobot.async_inference.configs import PolicyServerConfig

    config = PolicyServerConfig.from_dict(
        {
            "host": "localhost",
            "port": 9999,
            "fps": 30,
            "environment_dt": 999.0,
            "inference_latency": 0.1,
            "obs_queue_timeout": 1.0,
            "obs_similarity_atol": 0.15,
            "async_rtc": {"enabled": False},
        }
    )

    assert config.environment_dt == 1 / 30


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_get_action_chunk_does_not_pass_rtc_kwargs_when_disabled(policy_server):
    """Default async inference remains compatible with policies that do not accept RTC kwargs."""

    class StrictPolicy(MockPolicy):
        def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            batch_size = len(observation[OBS_STATE])
            return torch.zeros(batch_size, 10, 6)

    policy_server.policy = StrictPolicy()
    policy_server.actions_per_chunk = 10

    chunk = policy_server._get_action_chunk({OBS_STATE: torch.zeros(1, 6)})

    assert chunk.shape == (1, 10, 6)


def test_get_action_chunk_ignores_rtc_when_policy_config_is_disabled(policy_server):
    """The server runtime switch cannot force RTC when the loaded policy disables it."""

    class StrictPolicy(MockPolicy):
        def __init__(self):
            super().__init__()
            self.config.rtc_config = SimpleNamespace(enabled=False)

        def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            batch_size = len(observation[OBS_STATE])
            return torch.zeros(batch_size, 10, 6)

    policy_server.policy = StrictPolicy()
    policy_server.actions_per_chunk = 10
    policy_server.config.async_rtc.enabled = True

    chunk = policy_server._get_action_chunk({OBS_STATE: torch.zeros(1, 6)})

    assert chunk.shape == (1, 10, 6)


def test_predict_action_chunk_passes_rtc_leftover_and_delay(policy_server):
    """RTC-enabled async inference feeds previous leftover actions back into the policy."""

    class RecordingRTCPolicy(MockPolicy):
        def __init__(self):
            super().__init__()
            self.config.rtc_config = SimpleNamespace(enabled=True, execution_horizon=10)
            self.calls = []
            self._chunks = [
                torch.arange(60, dtype=torch.float32).reshape(1, 10, 6),
                torch.arange(60, 120, dtype=torch.float32).reshape(1, 10, 6),
            ]

        def predict_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
            self.calls.append(kwargs)
            return self._chunks[len(self.calls) - 1].clone()

    policy = RecordingRTCPolicy()
    policy_server.policy = policy
    policy_server.actions_per_chunk = 10
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    policy_server.config.async_rtc.enabled = True

    first_obs = _make_obs(torch.zeros(6), timestep=0)
    policy_server._predict_action_chunk(first_obs)

    policy_server._rtc_latency_tracker.reset()
    policy_server._rtc_latency_tracker.add(2.1 * policy_server.config.environment_dt)

    second_obs = _make_obs(torch.zeros(6), timestep=3)
    policy_server._predict_action_chunk(second_obs)

    assert policy.calls[0]["prev_chunk_left_over"] is None
    assert policy.calls[0]["inference_delay"] == 0
    assert policy.calls[1]["inference_delay"] == 3
    torch.testing.assert_close(policy.calls[1]["prev_chunk_left_over"], policy._chunks[0].squeeze(0)[4:])


def test_rtc_inference_delay_uses_configured_latency_quantile(policy_server):
    """RTC inference delay should use a rolling quantile instead of the lifetime max."""
    dt = policy_server.config.environment_dt
    policy_server.config.async_rtc.latency_quantile = 0.9
    policy_server._rtc_latency_tracker.reset()

    for _ in range(99):
        policy_server._rtc_latency_tracker.add(6.1 * dt)
    policy_server._rtc_latency_tracker.add(26.0 * dt)

    assert policy_server._rtc_inference_delay() == 7


def test_predict_action_chunk_trims_stale_rtc_actions(monkeypatch, policy_server):
    """RTC async chunks must drop actions that expired while inference was running."""

    class RTCPolicy(MockPolicy):
        def __init__(self):
            super().__init__()
            self.config.rtc_config = SimpleNamespace(enabled=True, execution_horizon=10)

        def predict_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
            del kwargs
            return torch.arange(60, dtype=torch.float32).reshape(1, 10, 6)

    policy_server.policy = RTCPolicy()
    policy_server.actions_per_chunk = 10
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    policy_server.config.async_rtc.enabled = True

    start_perf = 100.0
    elapsed = 2.1 * policy_server.config.environment_dt
    perf_values = iter(
        [
            start_perf,
            start_perf,
            start_perf,
            start_perf,
            start_perf,
            start_perf,
            start_perf,
            start_perf + elapsed,
        ]
    )

    monkeypatch.setattr("lerobot.async_inference.policy_server.time.perf_counter", lambda: next(perf_values))

    obs = _make_obs(torch.zeros(6), timestep=20)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert [ta.get_timestep() for ta in timed_actions] == list(range(23, 30))
    assert [ta.get_action()[0].item() for ta in timed_actions] == [18, 24, 30, 36, 42, 48, 54]

    expected_first_ts = obs.get_timestamp() + 3 * policy_server.config.environment_dt
    assert abs(timed_actions[0].get_timestamp() - expected_first_ts) < 1e-6
    assert policy_server._rtc_chunk_start_timestep == 23
    assert policy_server._rtc_original_actions.shape == (7, 6)
    torch.testing.assert_close(policy_server._rtc_original_actions[0], torch.arange(18, 24, dtype=torch.float32))


def test_reset_server_clears_rtc_state(policy_server):
    """A fresh client connection must not inherit RTC leftovers from a previous run."""
    policy_server._rtc_original_actions = torch.ones(3, 6)
    policy_server._rtc_processed_actions = torch.ones(3, 6)
    policy_server._rtc_chunk_start_timestep = 42
    policy_server._rtc_latency_tracker.add(1.0)

    policy_server._reset_server()

    assert policy_server._rtc_original_actions is None
    assert policy_server._rtc_processed_actions is None
    assert policy_server._rtc_chunk_start_timestep is None
    assert policy_server._rtc_latency_tracker.max() == 0.0


def test_rtc_leftover_for_absolute_policy_uses_policy_device(policy_server):
    """Absolute-policy RTC prefixes must be moved back to the policy device before inference."""
    policy_server.policy.config.rtc_config = SimpleNamespace(enabled=True)
    policy_server.device = "meta"
    policy_server.config.async_rtc.enabled = True
    policy_server._rtc_original_actions = torch.ones(5, 6)
    policy_server._rtc_chunk_start_timestep = 0

    leftover = policy_server._rtc_leftover_for_observation(obs_timestep=1)

    assert leftover is not None
    assert leftover.shape == (3, 6)
    assert leftover.device.type == "meta"


def test_policy_device_prefers_runtime_device_over_checkpoint_config(policy_server):
    """The server's requested runtime device is authoritative after policy.to(device)."""
    policy_server.policy.config.device = "cuda"
    policy_server.device = "cpu"

    assert policy_server._policy_device() == "cpu"


def test_refresh_rtc_processor_steps_fills_relative_action_names_from_policy_config(policy_server):
    """Relative RTC re-anchoring needs action names when the processor was loaded without them."""
    from lerobot.processor import RelativeActionsProcessorStep

    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=None)
    policy_server.policy.config.action_feature_names = ["joint1", "joint2", "joint3"]
    policy_server.preprocessor = SimpleNamespace(steps=[relative_step])

    policy_server._refresh_rtc_processor_steps()

    assert policy_server._rtc_relative_step is relative_step
    assert relative_step.action_names == ["joint1", "joint2", "joint3"]


def test_write_rtc_debug_dump_writes_stats_jsonl(policy_server, tmp_path):
    """Server-side RTC debug data is written next to the client recording directory."""

    class DebugStep:
        step_idx = 2
        time = torch.tensor(0.5)
        guidance_weight = torch.tensor(10.0)
        inference_delay = 7
        execution_horizon = 10
        weights = torch.tensor([1.0, 1.0, 0.5, 0.0])
        correction = torch.ones(1, 4, 2)
        err = torch.ones(1, 4, 2) * 2
        x_t = torch.ones(1, 4, 2) * 3
        v_t = torch.ones(1, 4, 2) * 4
        x1_t = torch.ones(1, 4, 2) * 5

    class DebugProcessor:
        def __init__(self):
            self.reset_called = False

        def get_all_debug_steps(self):
            return [DebugStep()]

        def reset_tracker(self):
            self.reset_called = True

    processor = DebugProcessor()
    policy_server.config.async_rtc.debug_dump.enabled = True
    policy_server._configure_rtc_debug_dump(str(tmp_path))
    policy_server.policy.rtc_processor = processor

    policy_server._write_rtc_debug_dump(
        observation_timestep=100,
        chunk_start_timestep=103,
        chunk_len=47,
        total_latency_s=0.123,
        inference_delay=7,
        prev_chunk_left_over=torch.zeros(12, 6),
    )

    records = [json.loads(line) for line in (tmp_path / "rtc_debug.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["observation_timestep"] == 100
    assert record["chunk_start_timestep"] == 103
    assert record["chunk_len"] == 47
    assert record["leftover_len"] == 12
    assert record["steps"][0]["inference_delay"] == 7
    assert record["steps"][0]["execution_horizon"] == 10
    assert record["steps"][0]["weights"]["nonzero"] == 3
    assert record["steps"][0]["correction"]["norm"] == pytest.approx(torch.linalg.norm(torch.ones(1, 4, 2)).item())
    assert processor.reset_called is True


def test_enable_policy_rtc_debug_dump_reinitializes_rtc_processor(policy_server, tmp_path):
    """Enabling debug after loading a checkpoint must rebuild the RTC tracker."""

    class PolicyWithRTC(MockPolicy):
        def __init__(self):
            super().__init__()
            self.config.rtc_config = SimpleNamespace(enabled=True, debug=False, debug_maxlen=100)
            self.init_called = False

        def init_rtc_processor(self):
            self.init_called = True
            self.rtc_processor = SimpleNamespace(tracker_enabled=self.config.rtc_config.debug)

    policy = PolicyWithRTC()
    policy_server.policy = policy
    policy_server.config.async_rtc.debug_dump.enabled = True
    policy_server._configure_rtc_debug_dump(str(tmp_path))

    policy_server._enable_policy_rtc_debug_dump()

    assert policy.config.rtc_config.debug is True
    assert policy.init_called is True
    assert policy.rtc_processor.tracker_enabled is True
