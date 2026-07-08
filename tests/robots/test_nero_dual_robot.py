from types import SimpleNamespace

import numpy as np

from lerobot_robot_nero.config_nero import (
    NeroArmConfig,
    NeroCommandConfig,
    NeroConnectionConfig,
    NeroDualRobotConfig,
    NeroMappingConfig,
)
from lerobot_robot_nero.robot_nero_dual import NeroDualRobot, _NeroArmRuntime
from lerobot_robot_nero.trace import NeroInferenceTraceConfig, NeroInferenceTracer


class FakeEffector:
    def __init__(self):
        self.move_gripper_m_calls = []

    def move_gripper_m(self, **kwargs):
        self.move_gripper_m_calls.append(kwargs)


class FakeSdkRobot:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper"))

    def __init__(self):
        self._msg_mode = SimpleNamespace(ctrl_mode=3, move_mode=1, mit_mode=0, enable_can_push=0)
        self.calls = []
        self.enable_calls = 0
        self.enable_statuses = [False] * 7
        self.move_js_commands = []
        self.move_j_commands = []
        self.get_joint_angles_calls = 0

    def init_effector(self, effector):
        self.calls.append(("init_effector", effector))
        return FakeEffector()

    def connect(self):
        self.calls.append(("connect",))

    def _set_mode(self):
        self.calls.append(
            (
                "_set_mode",
                self._msg_mode.ctrl_mode,
                self._msg_mode.move_mode,
                self._msg_mode.mit_mode,
                self._msg_mode.enable_can_push,
            )
        )

    def set_normal_mode(self):
        self.calls.append(("set_normal_mode",))

    def set_follower_mode(self):
        self.calls.append(("set_follower_mode",))

    def set_motion_mode(self, motion_mode):
        self.calls.append(("set_motion_mode", motion_mode))

    def reset(self):
        self.calls.append(("reset",))

    def clear_joint_error(self, joint_index=255):
        self.calls.append(("clear_joint_error", joint_index))

    def enable(self):
        self.calls.append(("enable",))
        self.enable_calls += 1
        if self.enable_calls == 1:
            self.enable_statuses = [True, True, False, False, False, False, False]
            return True
        self.enable_statuses = [True] * 7
        return True

    def get_joints_enable_status_list(self):
        self.calls.append(("get_joints_enable_status_list", tuple(self.enable_statuses)))
        return list(self.enable_statuses)

    def set_speed_percent(self, speed_percent):
        self.calls.append(("set_speed_percent", speed_percent))

    def get_joint_angles(self):
        self.get_joint_angles_calls += 1
        return np.zeros(7)

    def get_flange_pose(self):
        self.calls.append(("get_flange_pose",))
        return SimpleNamespace(msg=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def fk(self, joints):
        self.calls.append(("fk", tuple(joints)))
        return [value + 1.0 for value in joints[:6]]

    def move_js(self, joints):
        self.move_js_commands.append(joints)
        self.calls.append(("move_js", tuple(joints)))

    def move_j(self, joints):
        self.move_j_commands.append(joints)
        self.calls.append(("move_j", tuple(joints)))


class FakeCamera:
    def __init__(self, failures=0):
        self.failures = failures
        self.connect_calls = 0
        self.disconnect_calls = 0

    @property
    def is_connected(self):
        return self.connect_calls > self.failures

    def connect(self):
        self.connect_calls += 1
        if self.connect_calls <= self.failures:
            raise TimeoutError("warmup failed")

    def disconnect(self):
        self.disconnect_calls += 1


def test_dual_arm_connect_takes_follower_can_control_before_enabling(monkeypatch):
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            connection=NeroConnectionConfig(speed_percent=42, enable_retry_s=0.0),
            mapping=NeroMappingConfig(arm="right"),
        )
    )
    fake_robot = FakeSdkRobot()
    monkeypatch.setattr(runtime, "_make_sdk_robot", lambda: fake_robot)

    runtime.connect()

    assert fake_robot.calls[:7] == [
        ("init_effector", "agx_gripper"),
        ("connect",),
        ("_set_mode", 1, 1, 0, 1),
        ("set_follower_mode",),
        ("set_motion_mode", "js"),
        ("reset",),
        ("clear_joint_error", 255),
    ]
    assert fake_robot.calls.count(("enable",)) == 2
    assert fake_robot.calls[-1] == ("set_speed_percent", 42)


def test_dual_arm_connect_can_skip_reset_on_connect(monkeypatch):
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            connection=NeroConnectionConfig(speed_percent=42, enable_retry_s=0.0, reset_on_connect=False),
            mapping=NeroMappingConfig(arm="right"),
        )
    )
    fake_robot = FakeSdkRobot()
    monkeypatch.setattr(runtime, "_make_sdk_robot", lambda: fake_robot)

    runtime.connect()

    assert ("reset",) not in fake_robot.calls
    assert ("clear_joint_error", 255) in fake_robot.calls
    assert fake_robot.calls[-1] == ("set_speed_percent", 42)


def test_dual_robot_config_keeps_pair_speed_when_only_reset_is_overridden():
    cfg = NeroDualRobotConfig(
        right=NeroArmConfig(connection=NeroConnectionConfig(reset_on_connect=False)),
        left=NeroArmConfig(connection=NeroConnectionConfig(channel="can1", reset_on_connect=False)),
    )

    assert cfg.right.connection.speed_percent == 100
    assert cfg.left.connection.speed_percent == 100
    assert cfg.right.connection.reset_on_connect is False
    assert cfg.left.connection.reset_on_connect is False


def test_dual_robot_connection_defaults_use_v120_firmware():
    cfg = NeroDualRobotConfig()

    assert cfg.right.connection.firmware_version == "V120"
    assert cfg.left.connection.firmware_version == "V120"


def test_dual_robot_connect_retries_camera_warmup_failures(monkeypatch):
    robot = object.__new__(NeroDualRobot)
    robot.cameras = {"left_wrist": FakeCamera(failures=1)}

    sleep_calls = []
    monkeypatch.setattr("lerobot_robot_nero.robot_nero_dual.time.sleep", lambda seconds: sleep_calls.append(seconds))

    robot._connect_cameras()

    camera = robot.cameras["left_wrist"]
    assert camera.connect_calls == 2
    assert camera.disconnect_calls == 0
    assert sleep_calls == [1.0]


def test_dual_arm_send_action_uses_move_js_by_default():
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="right"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf")),
        )
    )
    fake_robot = FakeSdkRobot()
    runtime.robot = fake_robot
    runtime.end_effector = FakeEffector()
    runtime._last_command = np.zeros(7)

    runtime.send_action({**{f"right_nero_joint_{idx}": 0.1 for idx in range(1, 8)}, "right_gripper_width": 0.03})

    assert fake_robot.move_js_commands == [[0.1] * 7]
    assert fake_robot.move_j_commands == []
    assert runtime.end_effector.move_gripper_m_calls == [{"value": 0.03, "force": 1.0}]


def test_dual_arm_send_action_uses_move_j_when_configured():
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="right"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf"), move_method="move_j"),
        )
    )
    fake_robot = FakeSdkRobot()
    runtime.robot = fake_robot
    runtime.end_effector = FakeEffector()
    runtime._last_command = np.zeros(7)

    runtime.send_action({**{f"right_nero_joint_{idx}": 0.1 for idx in range(1, 8)}, "right_gripper_width": 0.03})

    assert fake_robot.move_js_commands == []
    assert fake_robot.move_j_commands == [[0.1] * 7]


def test_dual_arm_send_action_records_command_and_feedback(tmp_path):
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="right"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf")),
        )
    )
    fake_robot = FakeSdkRobot()
    runtime.robot = fake_robot
    runtime.end_effector = FakeEffector()
    runtime._last_command = np.zeros(7)
    tracer = NeroInferenceTracer(
        NeroInferenceTraceConfig(enabled=True, dir=str(tmp_path), run_name="arm"),
        meta={"task": "trace"},
        action_names=[f"right_nero_joint_{idx}" for idx in range(1, 8)],
    )
    runtime.tracer = tracer

    runtime.send_action({**{f"right_nero_joint_{idx}": 0.1 for idx in range(1, 8)}, "right_gripper_width": 0.03})
    tracer.close()

    trace_text = (tmp_path / "arm/trace.jsonl").read_text()
    assert '"event": "arm_command"' in trace_text
    assert '"event": "arm_feedback_after_command"' in trace_text
    assert '"move_method": "move_js"' in trace_text


def test_dual_arm_high_rate_options_skip_gripper_and_feedback():
    runtime = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="right"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf")),
        )
    )
    fake_robot = FakeSdkRobot()
    runtime.robot = fake_robot
    runtime.end_effector = FakeEffector()
    runtime._last_command = np.zeros(7)

    runtime.send_action(
        {**{f"right_nero_joint_{idx}": 0.1 for idx in range(1, 8)}, "right_gripper_width": 0.03},
        send_gripper=False,
        read_feedback=False,
    )

    assert fake_robot.move_js_commands == [[0.1] * 7]
    assert runtime.end_effector.move_gripper_m_calls == []
    assert fake_robot.get_joint_angles_calls == 0


def test_dual_arm_reads_flange_pose_from_sdk_feedback():
    runtime = _NeroArmRuntime(NeroArmConfig(mapping=NeroMappingConfig(arm="right")))
    runtime.robot = FakeSdkRobot()
    runtime._gripper_width = 0.03

    pose = runtime.read_flange_pose()
    observation = runtime.flange_observation()

    assert np.allclose(pose, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert observation == {
        "right_flange_x": 0.1,
        "right_flange_y": 0.2,
        "right_flange_z": 0.3,
        "right_flange_roll": 0.4,
        "right_flange_pitch": 0.5,
        "right_flange_yaw": 0.6,
        "right_gripper_width": 0.03,
    }


def test_dual_arm_converts_command_joints_to_flange_action_with_fk():
    runtime = _NeroArmRuntime(NeroArmConfig(mapping=NeroMappingConfig(arm="left")))
    runtime.robot = FakeSdkRobot()

    action = runtime.flange_action_from_joints(
        np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 0.04
    )

    assert action == {
        "left_flange_x": 1.0,
        "left_flange_y": 1.1,
        "left_flange_z": 1.2,
        "left_flange_roll": 1.3,
        "left_flange_pitch": 1.4,
        "left_flange_yaw": 1.5,
        "left_gripper_width": 0.04,
    }


def test_dual_robot_send_action_stores_last_flange_action():
    robot = object.__new__(NeroDualRobot)
    robot.cameras = {}
    robot._is_connected = True
    robot.right = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="right"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf")),
        )
    )
    robot.left = _NeroArmRuntime(
        NeroArmConfig(
            mapping=NeroMappingConfig(arm="left"),
            command=NeroCommandConfig(alpha=1.0, max_step_rad=float("inf")),
        )
    )
    for runtime in (robot.right, robot.left):
        runtime.robot = FakeSdkRobot()
        runtime.end_effector = FakeEffector()
        runtime._last_command = np.zeros(7)

    robot.send_action(
        {
            **{f"right_nero_joint_{idx}": 0.1 for idx in range(1, 8)},
            "right_gripper_width": 0.03,
            **{f"left_nero_joint_{idx}": 0.2 for idx in range(1, 8)},
            "left_gripper_width": 0.04,
        },
        read_feedback=False,
    )

    assert robot.last_flange_action == {
        "right_flange_x": 1.1,
        "right_flange_y": 1.1,
        "right_flange_z": 1.1,
        "right_flange_roll": 1.1,
        "right_flange_pitch": 1.1,
        "right_flange_yaw": 1.1,
        "right_gripper_width": 0.03,
        "left_flange_x": 1.2,
        "left_flange_y": 1.2,
        "left_flange_z": 1.2,
        "left_flange_roll": 1.2,
        "left_flange_pitch": 1.2,
        "left_flange_yaw": 1.2,
        "left_gripper_width": 0.04,
    }
