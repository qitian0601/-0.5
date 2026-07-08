from lerobot.scripts.lerobot_setup_motors import COMPATIBLE_DEVICES


def test_so101_8dof_leader_is_supported_by_setup_motors():
    assert "so101_8dof_leader" in COMPATIBLE_DEVICES
