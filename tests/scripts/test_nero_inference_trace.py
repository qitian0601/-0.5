import json

from lerobot_robot_nero.async_client import DEFAULT_DUAL_ACTION_NAMES
from lerobot_robot_nero.trace import NeroInferenceTraceConfig, NeroInferenceTracer, load_replay_actions


def _action(offset: float = 0.0) -> dict[str, float]:
    return {name: float(idx) + offset for idx, name in enumerate(DEFAULT_DUAL_ACTION_NAMES)}


def test_inference_tracer_writes_meta_trace_and_replay_actions(tmp_path):
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="unit"),
        meta={"task": "fold_towel", "fps": 30},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )

    tracer.record("policy_raw_action", {"action": _action()})
    tracer.record_replay_action(_action(10.0), dt_s=0.005556)
    tracer.close()

    run_dir = tmp_path / "unit"
    assert json.loads((run_dir / "meta.json").read_text())["task"] == "fold_towel"

    trace_lines = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert [event["event"] for event in trace_lines] == ["policy_raw_action"]
    assert trace_lines[0]["sequence"] == 1

    replay_lines = [json.loads(line) for line in (run_dir / "replay_actions.jsonl").read_text().splitlines()]
    assert replay_lines[0]["sequence"] == 1
    assert replay_lines[0]["dt_s"] == 0.005556
    assert replay_lines[0]["action"]["right_nero_joint_1"] == 10.0
    assert replay_lines[0]["action"]["left_gripper_width"] == 25.0


def test_disabled_inference_tracer_does_not_create_files(tmp_path):
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=False, dir=str(tmp_path)),
        meta={"task": "fold_towel"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )

    tracer.record("policy_raw_action", {"action": _action()})
    tracer.record_replay_action(_action())
    tracer.close()

    assert list(tmp_path.iterdir()) == []


def test_load_replay_actions_reads_jsonl_in_action_name_order(tmp_path):
    path = tmp_path / "replay_actions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"action": _action()}),
                json.dumps({"action": _action(20.0)}),
            ]
        )
    )

    actions = load_replay_actions(path, DEFAULT_DUAL_ACTION_NAMES)

    assert actions.shape == (2, 16)
    assert actions[0, 0] == 0.0
    assert actions[0, 7] == 7.0
    assert actions[0, 15] == 15.0
    assert actions[1, 0] == 20.0


def test_inference_tracer_buffers_writes_until_background_flush(tmp_path):
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="buffered", flush_every=1000),
        meta={"task": "fold_towel"},
        action_names=DEFAULT_DUAL_ACTION_NAMES,
    )

    for _ in range(50):
        tracer.record("executor_step", {"action": _action()})
        tracer.record_replay_action(_action())

    tracer.close()

    assert len((tmp_path / "buffered/trace.jsonl").read_text().splitlines()) == 50
    assert len((tmp_path / "buffered/replay_actions.jsonl").read_text().splitlines()) == 50
