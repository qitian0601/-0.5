from lerobot_robot_nero.config_nero import NeroTrimConfig
import numpy as np

from lerobot_robot_nero.trimming import (
    trim_static_head_tail,
    trim_static_head_tail_indices_from_action_array,
    trim_static_head_tail_multi_arm,
)


def make_frame(joint_value: float, gripper_width: float = 0.0) -> dict:
    action = {f"right_nero_joint_{idx}": joint_value for idx in range(1, 8)}
    action["right_gripper_width"] = gripper_width
    return {"observation": {}, "action": action}


def make_dual_frame(right_joint_value: float, left_joint_value: float) -> dict:
    action = {f"right_nero_joint_{idx}": right_joint_value for idx in range(1, 8)}
    action.update({f"left_nero_joint_{idx}": left_joint_value for idx in range(1, 8)})
    action["right_gripper_width"] = 0.0
    action["left_gripper_width"] = 0.0
    return {"observation": {}, "action": action}


def test_trim_static_head_and_tail_keeps_motion_with_roll():
    frames = [make_frame(0.0) for _ in range(10)]
    frames += [make_frame(float(idx) * 0.01) for idx in range(1, 6)]
    frames += [make_frame(0.05) for _ in range(10)]
    config = NeroTrimConfig(static_time_s=0.3, preroll_s=0.1, postroll_s=0.1, min_episode_frames=1)

    trimmed = trim_static_head_tail(frames, fps=10, arm="right", config=config)

    assert len(trimmed) == 8
    assert trimmed[0] is frames[8]
    assert trimmed[-1] is frames[15]


def test_trim_rejects_fully_static_episode():
    frames = [make_frame(0.0) for _ in range(20)]

    trimmed = trim_static_head_tail(frames, fps=10, arm="right", config=NeroTrimConfig(min_episode_frames=1))

    assert trimmed == []


def test_trim_rejects_too_short_episode_after_trimming():
    frames = [make_frame(0.0) for _ in range(10)]
    frames += [make_frame(0.01), make_frame(0.02)]
    frames += [make_frame(0.02) for _ in range(10)]
    config = NeroTrimConfig(static_time_s=0.3, preroll_s=0.0, postroll_s=0.0, min_episode_frames=5)

    trimmed = trim_static_head_tail(frames, fps=10, arm="right", config=config)

    assert trimmed == []


def test_multi_arm_trim_keeps_episode_when_only_left_arm_moves():
    frames = [make_dual_frame(0.0, 0.0) for _ in range(10)]
    frames += [make_dual_frame(0.0, float(idx) * 0.01) for idx in range(1, 6)]
    frames += [make_dual_frame(0.0, 0.05) for _ in range(10)]
    config = NeroTrimConfig(static_time_s=0.3, preroll_s=0.1, postroll_s=0.1, min_episode_frames=1)

    trimmed = trim_static_head_tail_multi_arm(frames, fps=10, arms=("right", "left"), config=config)

    assert len(trimmed) == 8
    assert trimmed[0] is frames[8]
    assert trimmed[-1] is frames[15]


def test_action_array_trim_matches_multi_arm_trim_indices():
    frames = [make_dual_frame(0.0, 0.0) for _ in range(10)]
    frames += [make_dual_frame(0.0, float(idx) * 0.01) for idx in range(1, 6)]
    frames += [make_dual_frame(0.0, 0.05) for _ in range(10)]
    action_values = np.array(
        [
            [frame["action"][f"right_nero_joint_{idx}"] for idx in range(1, 8)]
            + [frame["action"]["right_gripper_width"]]
            + [frame["action"][f"left_nero_joint_{idx}"] for idx in range(1, 8)]
            + [frame["action"]["left_gripper_width"]]
            for frame in frames
        ],
        dtype=float,
    )
    config = NeroTrimConfig(static_time_s=0.3, preroll_s=0.1, postroll_s=0.1, min_episode_frames=1)

    trim_indices = trim_static_head_tail_indices_from_action_array(action_values, fps=10, config=config)

    assert trim_indices == (8, 16)
