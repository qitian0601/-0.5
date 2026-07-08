import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "nero_left_leader_mode.py"


def load_script():
    spec = importlib.util.spec_from_file_location("nero_left_leader_mode", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRobot:
    def __init__(self):
        self.calls = []

    def _send_msg(self, msg):
        self.calls.append(("_send_msg", msg.grag_teach_ctrl))


def test_left_leader_script_targets_stable_left_interface():
    script = load_script()

    assert script.ARM_NAME == "left"
    assert script.CHANNEL == "nero_left"
    assert str(script.FIRMWARE) == "v120"


def test_left_drag_teach_controls():
    script = load_script()
    robot = FakeRobot()

    script.start_drag_teach(robot)
    script.stop_drag_teach(robot)

    assert robot.calls == [("_send_msg", 0x01), ("_send_msg", 0x02)]
