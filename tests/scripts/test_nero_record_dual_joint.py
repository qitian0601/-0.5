from lerobot_robot_nero.mapping import SO101ToNeroMapping
from lerobot_robot_nero.record_dual_joint import (
    EpisodeReviewDecision,
    convert_joint_frame_to_flange_frame,
    make_flange_dataset_features,
    prompt_episode_review,
    _resolve_flange_dataset_config,
    review_decision_from_text,
    run_teleop_step,
    save_flange_episode_frames,
    save_episode_frames_pair,
    sync_to_current_leaders_before_next_episode,
)
from lerobot.configs.dataset import DatasetRecordConfig
import numpy as np


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
    control_dt_s = 0.006
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


def test_dual_teleop_step_does_not_read_observation_when_not_recording():
    robot = Robot()

    frame = run_teleop_step(
        right_leader=Leader(),
        left_leader=Leader(),
        robot=robot,
        record=False,
    )

    assert frame is None
    assert robot.observation_reads == 0
    assert len(robot.sent) == 1


def test_dual_teleop_step_reads_observation_when_recording():
    robot = Robot()

    frame = run_teleop_step(
        right_leader=Leader(),
        left_leader=Leader(),
        robot=robot,
        record=True,
    )

    assert frame is not None
    assert robot.observation_reads == 1
    assert robot.flange_observation_reads == 1
    assert frame["flange_action"] == {"right_flange_x": 1.0, "left_flange_x": 2.0}
    assert len(robot.sent) == 1


def test_sync_before_next_episode_smoothly_aligns_when_more_episodes_remain(monkeypatch):
    calls = []

    def sync(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.sync_to_current_leaders", sync)

    sync_to_current_leaders_before_next_episode(
        right_leader="right-leader",
        left_leader="left-leader",
        robot="robot",
        recorded_episodes=1,
        max_episodes=3,
        takeover_time_s=2.0,
        takeover_dt_s=0.02,
    )

    assert calls == [
        {
            "right_leader": "right-leader",
            "left_leader": "left-leader",
            "robot": "robot",
            "takeover_time_s": 2.0,
            "takeover_dt_s": 0.02,
        }
    ]


def test_sync_before_next_episode_skips_after_last_episode(monkeypatch):
    calls = []

    def sync(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.sync_to_current_leaders", sync)

    sync_to_current_leaders_before_next_episode(
        right_leader="right-leader",
        left_leader="left-leader",
        robot="robot",
        recorded_episodes=3,
        max_episodes=3,
        takeover_time_s=2.0,
        takeover_dt_s=0.02,
    )

    assert calls == []


def test_review_decision_defaults_to_save_on_empty_input():
    assert review_decision_from_text("") is EpisodeReviewDecision.SAVE
    assert review_decision_from_text("   ") is EpisodeReviewDecision.SAVE


def test_review_decision_accepts_rerecord_commands():
    assert review_decision_from_text("r") is EpisodeReviewDecision.RERECORD
    assert review_decision_from_text("重录") is EpisodeReviewDecision.RERECORD


def test_review_decision_accepts_finish_commands():
    assert review_decision_from_text("q") is EpisodeReviewDecision.FINISH_AND_EXIT
    assert review_decision_from_text("退出") is EpisodeReviewDecision.FINISH_AND_EXIT


def test_episode_review_defaults_to_save_when_stdin_is_not_interactive(monkeypatch):
    class NonInteractiveStdin:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NonInteractiveStdin())

    decision = prompt_episode_review(episode_number=1, kept_frames=12, raw_frames=20)

    assert decision is EpisodeReviewDecision.SAVE


def test_convert_frame_to_flange_pose_uses_flange_observation_and_last_flange_action():
    class PoseRobot(Robot):
        cameras = {"front": object()}

    frame = {
        "observation": {"front": "front-image"},
        "action": {"right_nero_joint_1": 0.0},
        "flange_observation": {
            "right_flange_x": 0.0,
            "right_flange_y": 0.1,
            "right_flange_z": 0.2,
            "right_flange_roll": 0.3,
            "right_flange_pitch": 0.4,
            "right_flange_yaw": 0.5,
            "right_gripper_width": 0.03,
            "left_flange_x": 3.0,
            "left_flange_y": 3.1,
            "left_flange_z": 3.2,
            "left_flange_roll": 3.3,
            "left_flange_pitch": 3.4,
            "left_flange_yaw": 3.5,
            "left_gripper_width": 0.04,
        },
        "flange_action": {
            "right_flange_x": 1.0,
            "right_flange_y": 1.1,
            "right_flange_z": 1.2,
            "right_flange_roll": 1.3,
            "right_flange_pitch": 1.4,
            "right_flange_yaw": 1.5,
            "right_gripper_width": 0.03,
            "left_flange_x": 2.0,
            "left_flange_y": 2.1,
            "left_flange_z": 2.2,
            "left_flange_roll": 2.3,
            "left_flange_pitch": 2.4,
            "left_flange_yaw": 2.5,
            "left_gripper_width": 0.04,
        },
    }

    converted = convert_joint_frame_to_flange_frame(frame, PoseRobot())

    assert converted["observation"]["right_flange_x"] == 0.0
    assert converted["observation"]["front"] == "front-image"
    assert converted["action"]["right_flange_x"] == 1.0
    assert converted["action"]["left_flange_yaw"] == 2.5


def test_make_flange_dataset_features_uses_robot_flange_features(monkeypatch):
    class FeatureRobot:
        flange_action_features = {"right_flange_x": float}
        flange_observation_features = {"right_flange_x": float, "front": (480, 640, 3)}

    seen = []

    def aggregate(*, pipeline, initial_features, use_videos):
        seen.append(initial_features)
        if "right_flange_x" in repr(initial_features) and "front" not in repr(initial_features):
            return {"action": {"dtype": "float32", "shape": (1,), "names": ["right_flange_x"]}}
        return {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["right_flange_x"]}}

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.aggregate_pipeline_dataset_features", aggregate)

    features = make_flange_dataset_features(FeatureRobot(), use_videos=True)

    assert features["action"]["names"] == ["right_flange_x"]
    assert features["observation.state"]["names"] == ["right_flange_x"]
    assert any("front" in repr(initial_features) for initial_features in seen)


def test_save_episode_frames_pair_saves_joint_and_flange(monkeypatch):
    class Dataset:
        features = {"dummy": {"dtype": "float32", "shape": [1]}}

        def __init__(self):
            self.frames = []
            self.saved = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

    frames = [
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
        return {"frame": frame, "task": task}

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.flatten_episode_frame", flatten)

    save_episode_frames_pair(
        joint_dataset=joint_dataset,
        flange_dataset=flange_dataset,
        frames=frames,
        task="pickplace",
    )

    assert [frame["frame"]["observation"] for frame in joint_dataset.frames] == [{"joint": 1}, {"joint": 2}]
    assert [frame["frame"]["observation"] for frame in flange_dataset.frames] == [{"pose": 1}, {"pose": 2}]
    assert joint_dataset.saved == 1
    assert flange_dataset.saved == 1


def test_save_episode_frames_pair_copies_joint_videos_to_flange_dataset(tmp_path, monkeypatch):
    class Meta:
        fps = 30
        features = {
            "action": {"dtype": "float32", "shape": [1], "names": ["pose"]},
            "observation.state": {"dtype": "float32", "shape": [1], "names": ["pose"]},
            "observation.images.front": {
                "dtype": "video",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channels"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        video_keys = ["observation.images.front"]
        total_episodes = 0
        total_frames = 0
        tasks = {}
        video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

        def __init__(self, root):
            self.root = root
            self.saved_episodes = []
            self.latest_episode = None

        def save_episode_tasks(self, tasks):
            for task in tasks:
                self.tasks.setdefault(task, len(self.tasks))

        def get_task_index(self, task):
            return self.tasks[task]

        def save_episode(self, episode_index, episode_length, episode_tasks, ep_stats, ep_metadata):
            self.latest_episode = {
                "episode_index": episode_index,
                "length": episode_length,
                "tasks": episode_tasks,
                **ep_metadata,
                **{f"stats/{key}/{stat}": value for key, stats in ep_stats.items() for stat, value in stats.items()},
            }
            self.saved_episodes.append(
                {
                    "episode_index": episode_index,
                    "length": episode_length,
                    "tasks": episode_tasks,
                    "stats": ep_stats,
                    "metadata": ep_metadata,
                }
            )
            self.total_episodes += 1
            self.total_frames += episode_length

    class Writer:
        def __init__(self, root):
            self._meta = Meta(root)
            self.saved_data_buffers = []

        def _save_episode_data(self, episode_buffer):
            self.saved_data_buffers.append(dict(episode_buffer))
            episode_length = len(episode_buffer["timestamp"])
            return {
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": episode_length,
            }

    class Dataset:
        features = Meta.features

        def __init__(self, root):
            self.root = root
            self.writer = Writer(root)
            self.saved = 0

        def add_frame(self, frame):
            del frame

        def save_episode(self):
            self.saved += 1
            video_path = self.root / Meta.video_path.format(
                video_key="observation.images.front", chunk_index=0, file_index=0
            )
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"encoded-once")
            self.writer._meta.save_episode(
                0,
                2,
                ["pickplace"],
                {},
                {
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": 2,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                    "videos/observation.images.front/from_timestamp": 0.0,
                    "videos/observation.images.front/to_timestamp": 1.0,
                },
            )

    def flatten(frame, *, features, task):
        del features
        return {
            "action": np.array([frame["action"]["pose"]], dtype=np.float32),
            "observation.state": np.array([frame["observation"]["pose"]], dtype=np.float32),
            "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
            "task": task,
        }

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.flatten_episode_frame", flatten)

    joint_dataset = Dataset(tmp_path / "joint")
    flange_dataset = Dataset(tmp_path / "flange")
    frames = [
        {
            "observation": {"pose": 1},
            "action": {"pose": 10},
            "flange": {"observation": {"pose": 2}, "action": {"pose": 20}},
        },
        {
            "observation": {"pose": 3},
            "action": {"pose": 30},
            "flange": {"observation": {"pose": 4}, "action": {"pose": 40}},
        },
    ]

    save_episode_frames_pair(
        joint_dataset=joint_dataset,
        flange_dataset=flange_dataset,
        frames=frames,
        task="pickplace",
    )

    copied_video = tmp_path / "flange/videos/observation.images.front/chunk-000/file-000.mp4"
    assert copied_video.read_bytes() == b"encoded-once"
    assert joint_dataset.saved == 1
    assert flange_dataset.saved == 0
    assert len(flange_dataset.writer.saved_data_buffers) == 1
    assert "observation.images.front" not in flange_dataset.writer.saved_data_buffers[0]
    assert flange_dataset.writer._meta.saved_episodes[0]["metadata"][
        "videos/observation.images.front/file_index"
    ] == 0


def test_save_flange_episode_frames_only_saves_flange_dataset(monkeypatch):
    class Dataset:
        features = {"dummy": {"dtype": "float32", "shape": [1]}}

        def __init__(self):
            self.frames = []
            self.saved = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

    frames = [
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

    monkeypatch.setattr("lerobot_robot_nero.record_dual_joint.flatten_episode_frame", flatten)

    save_flange_episode_frames(
        flange_dataset=flange_dataset,
        frames=frames,
        task="pickplace",
    )

    assert [frame["frame"]["observation"] for frame in flange_dataset.frames] == [{"pose": 1}, {"pose": 2}]
    assert [frame["task"] for frame in flange_dataset.frames] == ["pickplace", "pickplace"]
    assert flange_dataset.saved == 1


def test_flange_dataset_config_inherits_base_settings_with_repo_and_root_override():
    class Config:
        dataset = DatasetRecordConfig(
            repo_id="chenglong/pickplace_joint_001",
            root="/tmp/pickplace_joint_001",
            single_task="pickplace",
            fps=20,
            num_episodes=7,
            video=False,
            push_to_hub=True,
            num_image_writer_threads_per_camera=2,
        )
        flange_dataset = DatasetRecordConfig(
            repo_id="chenglong/pickplace_flange_pose_001",
            root="/tmp/pickplace_flange_pose_001",
            push_to_hub=False,
        )

    resolved = _resolve_flange_dataset_config(Config())

    assert resolved.repo_id == "chenglong/pickplace_flange_pose_001"
    assert str(resolved.root) == "/tmp/pickplace_flange_pose_001"
    assert resolved.single_task == "pickplace"
    assert resolved.fps == 20
    assert resolved.num_episodes == 7
    assert resolved.video is False
    assert resolved.push_to_hub is False
    assert resolved.num_image_writer_threads_per_camera == 2


def test_flange_dataset_config_mirrors_base_video_for_copied_videos():
    class Config:
        dataset = DatasetRecordConfig(
            repo_id="chenglong/pickplace_joint_001",
            root="/tmp/pickplace_joint_001",
            single_task="pickplace",
            fps=20,
            video=True,
        )
        flange_dataset = DatasetRecordConfig(
            repo_id="chenglong/pickplace_flange_pose_001",
            root="/tmp/pickplace_flange_pose_001",
            video=False,
        )

    resolved = _resolve_flange_dataset_config(Config())

    assert resolved.video is True
