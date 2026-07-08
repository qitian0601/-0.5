from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replay_bus_table_01_move_js.sh"


def test_replay_bus_table_01_move_js_script_uses_original_dataset_and_move_js_entrypoint():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DATASET=/home/chenglong/workplace/nero_teleop_ws/data/lerobot/bus_table/bus_table_01" in text
    assert "nero-replay-dual-joint" in text
    assert "--dataset.root=\"${DATASET}\"" in text
    assert "--dataset.episode=\"${episode}\"" in text
    assert "--robot.right.connection.channel=nero_right" in text
    assert "--robot.left.connection.channel=nero_left" in text
    assert "--robot.right.connection.speed_percent=20" in text
    assert "--robot.left.connection.speed_percent=20" in text
    assert "--robot.right.command.move_method=move_js" in text
    assert "--robot.left.command.move_method=move_js" in text
    assert "--robot.right.command.max_step_rad=0.05" in text
    assert "--robot.left.command.max_step_rad=0.05" in text
    assert "--dataset.fps=20" in text
    assert "--high_rate_control=true" not in text
    assert "episode=${1:-0}" in text
