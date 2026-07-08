import builtins

from lerobot_robot_nero import record_dual_joint_raw
from lerobot_robot_nero.config_nero import NeroDualRobotConfig
from lerobot_robot_nero.mapping import SO101ToNeroMapping
from lerobot.configs.dataset import DatasetRecordConfig


class Dataset:
    features = {"dummy": {"dtype": "float32", "shape": [1]}}

    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


class Leader:
    def get_action(self):
        return {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "joint_7.pos": 0.0,
            "gripper.pos": 50.0,
        }


class Arm:
    def __init__(self, arm: str):
        self.arm = arm
        self.mapping = SO101ToNeroMapping(so101_gripper_min_deg=0.0, so101_gripper_max_deg=100.0)


class Robot:
    last_flange_action = {"right_flange_x": 1.0, "left_flange_x": 2.0}

    def __init__(self):
        self.right = Arm("right")
        self.left = Arm("left")
        self.observation_reads = 0
        self.flange_observation_reads = 0
        self.sent = []

    def get_observation(self):
        self.observation_reads += 1
        return {"right_nero_joint_1": 0.0, "left_nero_joint_1": 0.0}

    def send_action(self, action):
        self.sent.append(action)
        return action

    def get_flange_state_observation(self):
        self.flange_observation_reads += 1
        return {"right_flange_x": 0.0, "left_flange_x": 0.0}


def test_raw_teleop_step_records_flange_pose_frame_data():
    robot = Robot()

    frame = record_dual_joint_raw.run_teleop_step(
        right_leader=Leader(),
        left_leader=Leader(),
        robot=robot,
        record=True,
    )

    assert frame is not None
    assert robot.observation_reads == 1
    assert robot.flange_observation_reads == 1
    assert frame["flange_observation"] == {"right_flange_x": 0.0, "left_flange_x": 0.0}
    assert frame["flange_action"] == {"right_flange_x": 1.0, "left_flange_x": 2.0}


def test_save_raw_episode_frames_saves_every_collected_frame(monkeypatch):
    collected = [
        {"observation": {"value": 1}, "action": {"value": 10}},
        {"observation": {"value": 2}, "action": {"value": 20}},
        {"observation": {"value": 3}, "action": {"value": 30}},
    ]
    dataset = Dataset()

    def flatten(frame, *, features, task):
        return {"frame": frame, "features": features, "task": task}

    monkeypatch.setattr(record_dual_joint_raw, "flatten_episode_frame", flatten)

    record_dual_joint_raw.save_raw_episode_frames(
        dataset=dataset,
        frames=collected,
        task="fold towel",
    )

    assert [frame["frame"] for frame in dataset.frames] == collected
    assert [frame["task"] for frame in dataset.frames] == ["fold towel"] * len(collected)
    assert dataset.saved == 1


def test_save_raw_episode_frames_can_save_joint_and_flange_datasets(monkeypatch):
    collected = [
        {
            "observation": {"joint": 1},
            "action": {"joint": 10},
            "flange": {"observation": {"pose": 1}, "action": {"pose": 10}},
        },
        {
            "observation": {"joint": 2},
            "action": {"joint": 20},
            "flange": {"observation": {"pose": 2}, "action": {"pose": 20}},
        },
    ]
    joint_dataset = Dataset()
    flange_dataset = Dataset()

    def flatten(frame, *, features, task):
        return {"frame": frame, "features": features, "task": task}

    monkeypatch.setattr(record_dual_joint_raw, "flatten_episode_frame", flatten)

    record_dual_joint_raw.save_raw_episode_frames_pair(
        joint_dataset=joint_dataset,
        flange_dataset=flange_dataset,
        frames=collected,
        task="pickplace",
    )

    assert [frame["frame"]["observation"] for frame in joint_dataset.frames] == [{"joint": 1}, {"joint": 2}]
    assert [frame["frame"]["observation"] for frame in flange_dataset.frames] == [{"pose": 1}, {"pose": 2}]
    assert joint_dataset.saved == 1
    assert flange_dataset.saved == 1


def test_save_raw_flange_episode_frames_only_saves_flange_dataset(monkeypatch):
    collected = [
        {
            "observation": {"joint": 1},
            "action": {"joint": 10},
            "flange": {"observation": {"pose": 1}, "action": {"pose": 10}},
        },
        {
            "observation": {"joint": 2},
            "action": {"joint": 20},
            "flange": {"observation": {"pose": 2}, "action": {"pose": 20}},
        },
    ]
    flange_dataset = Dataset()

    def flatten(frame, *, features, task):
        return {"frame": frame, "features": features, "task": task}

    monkeypatch.setattr(record_dual_joint_raw, "flatten_episode_frame", flatten)

    record_dual_joint_raw.save_raw_flange_episode_frames(
        flange_dataset=flange_dataset,
        frames=collected,
        task="pickplace",
    )

    assert [frame["frame"]["observation"] for frame in flange_dataset.frames] == [{"pose": 1}, {"pose": 2}]
    assert [frame["task"] for frame in flange_dataset.frames] == ["pickplace", "pickplace"]
    assert flange_dataset.saved == 1


def test_attach_flange_frames_adds_flange_data_to_each_frame(monkeypatch):
    frames = [{"observation": {"joint": 1}, "action": {"joint": 10}}]
    robot = object()

    monkeypatch.setattr(
        record_dual_joint_raw,
        "convert_joint_frame_to_flange_frame",
        lambda frame, robot_arg: {"observation": {"pose": frame["observation"]["joint"]}, "action": {"pose": 10}},
    )

    enriched = record_dual_joint_raw.attach_flange_frames(frames, robot)

    assert enriched == [
        {
            "observation": {"joint": 1},
            "action": {"joint": 10},
            "flange": {"observation": {"pose": 1}, "action": {"pose": 10}},
        }
    ]


def test_raw_recorder_smoothly_syncs_before_next_episode(monkeypatch):
    class Connected:
        def __init__(self):
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    class RawRobot:
        name = "nero_dual"
        cameras = {}

        def __init__(self):
            self.connected = False

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

    leaders = [Connected(), Connected()]
    robot = RawRobot()
    sync_calls = []

    monkeypatch.setattr(record_dual_joint_raw, "make_teleoperator_from_config", lambda cfg: leaders.pop(0))
    monkeypatch.setattr(record_dual_joint_raw, "NeroDualRobot", lambda cfg: robot)
    monkeypatch.setattr(record_dual_joint_raw, "make_dataset_features", lambda robot, *, use_videos: {})
    monkeypatch.setattr(record_dual_joint_raw, "make_flange_dataset_features", lambda robot, *, use_videos: {})
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    monkeypatch.setattr(record_dual_joint_raw, "sync_to_current_leaders", lambda **kwargs: sync_calls.append(kwargs))
    monkeypatch.setattr(
        record_dual_joint_raw,
        "run_idle_teleop_until_start",
        lambda **kwargs: record_dual_joint_raw.IdleTeleopDecision.START_RECORDING,
    )
    monkeypatch.setattr(
        record_dual_joint_raw,
        "collect_episode_buffer",
        lambda **kwargs: [
            {
                "observation": {"right_nero_joint_1": 0.0},
                "action": {"right_nero_joint_1": 0.0},
                "flange_observation": {"right_flange_x": 0.0},
                "flange_action": {"right_flange_x": 0.0},
            }
        ],
    )
    monkeypatch.setattr(
        record_dual_joint_raw,
        "prompt_raw_episode_review",
        lambda **kwargs: record_dual_joint_raw.EpisodeReviewDecision.SAVE,
    )
    monkeypatch.setattr(record_dual_joint_raw, "run_idle_teleop_until_shutdown", lambda **kwargs: None)

    cfg = record_dual_joint_raw.NeroRecordDualJointRawConfig(
        right_leader=object(),
        left_leader=object(),
        robot=NeroDualRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id="chenglong/pickplace_joint_raw",
            single_task="pickplace",
            num_episodes=2,
            video=False,
            push_to_hub=False,
        ),
        dry_run=True,
        manual_stop=False,
        takeover_time_s=1.5,
        takeover_dt_s=0.1,
    )

    record_dual_joint_raw.record_dual_joint_raw(cfg)

    assert len(sync_calls) == 2
    assert sync_calls[1]["takeover_time_s"] == 1.5
    assert sync_calls[1]["takeover_dt_s"] == 0.1
