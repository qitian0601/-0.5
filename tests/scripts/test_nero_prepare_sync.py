import numpy as np

from lerobot_robot_nero.prepare_sync import confirm_nero_enable, smooth_takeover_commands


def test_confirm_nero_enable_defaults_to_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert confirm_nero_enable()


def test_confirm_nero_enable_accepts_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")

    assert not confirm_nero_enable()


def test_smooth_takeover_commands_start_after_current_and_end_at_target():
    current = np.zeros(7)
    target = np.ones(7)

    commands = smooth_takeover_commands(current, target, steps=4)

    assert len(commands) == 4
    assert np.all(commands[0] > current)
    assert np.allclose(commands[-1], target)
