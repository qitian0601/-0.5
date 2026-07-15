from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def read_script(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_joint_inference_client_uses_stable_can_names():
    text = read_script("scripts/run_nero_infer_client.sh")

    assert "--robot.right.connection.channel=nero_right" in text
    assert "--robot.left.connection.channel=nero_left" in text
    assert "--robot.right.connection.channel=can0" not in text
    assert "--robot.left.connection.channel=can1" not in text
    assert "--robot.right.connection.channel=can1" not in text
    assert "--robot.left.connection.channel=can0" not in text


def test_pickplace_new_inference_client_uses_stable_can_names():
    text = read_script("scripts/run_nero_pickplace_new_infer_client.sh")

    assert "--robot.right.connection.channel=nero_right" in text
    assert "--robot.left.connection.channel=nero_left" in text
    assert "--robot.right.connection.channel=can0" not in text
    assert "--robot.left.connection.channel=can1" not in text
    assert "--robot.right.connection.channel=can1" not in text
    assert "--robot.left.connection.channel=can0" not in text


def test_ee_clients_default_to_stable_can_names():
    for script_name in [
        "scripts/run_nero_ee_infer_client.sh",
        "scripts/run_nero_pickplace_ee_infer_client.sh",
    ]:
        text = read_script(script_name)

        assert 'RIGHT_CAN="${NERO_RIGHT_CAN:-nero_right}"' in text
        assert 'LEFT_CAN="${NERO_LEFT_CAN:-nero_left}"' in text
        assert 'RIGHT_CAN="${NERO_RIGHT_CAN:-can0}"' not in text
        assert 'LEFT_CAN="${NERO_LEFT_CAN:-can1}"' not in text
        assert 'RIGHT_CAN="${NERO_RIGHT_CAN:-can1}"' not in text
        assert 'LEFT_CAN="${NERO_LEFT_CAN:-can0}"' not in text


def test_pickplace_ee_new_client_handeye_defaults_exist():
    text = read_script("scripts/demo/pickplace_ee_new_client.sh")

    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/demo/pickplace_ee_new_client.sh")],
        check=True,
    )
    expected_paths = {
        "right_handeye_camera_to_base_yaml": "handeye_result_right(1).yml",
        "left_handeye_camera_to_base_yaml": "handeye_result_left(1).yml",
    }
    for option in [
        "right_handeye_camera_to_base_yaml",
        "left_handeye_camera_to_base_yaml",
    ]:
        match = re.search(rf'"?--{option}=([^"\n]+)"?', text)
        assert match is not None
        path = Path(match.group(1))
        assert path.exists()
        assert path.name == expected_paths[option]


def test_pickplace_ee_new_client_defaults_to_mix_56_checkpoint_and_training_task():
    text = read_script("scripts/demo/pickplace_ee_new_client.sh")

    assert (
        'POLICY_PATH="${NERO_PICKPLACE_EE_POLICY_PATH:-/home/chenglong/workplace/'
        'nero_teleop_ws/lerobot/outputs/train/pickplace_ee_mix_56/checkpoints/014000/pretrained_model}"'
        in text
    )
    assert (
        'TASK="${NERO_PICKPLACE_EE_TASK:-Pick up the cube and place it in the target area.}"'
        in text
    )
