import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from .config_nero import NeroTrimConfig
from .trimming import trim_static_head_tail_indices_from_action_array

logger = logging.getLogger(__name__)

AUTO_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


@dataclass(frozen=True)
class EpisodeTrimPlan:
    source_episode_index: int
    output_episode_index: int
    source_from_index: int
    source_to_index: int
    raw_frames: int
    kept_frames: int
    task: str


@dataclass(frozen=True)
class DatasetTrimSummary:
    original_episodes: int
    original_frames: int
    processed_episodes: int
    processed_frames: int
    rejected_episodes: int


@dataclass
class NeroTrimDualJointDatasetConfig:
    repo_id: str
    root: Path
    new_repo_id: str | None = None
    new_root: Path | None = None
    trim: NeroTrimConfig = field(default_factory=NeroTrimConfig)
    overwrite: bool = False
    dry_run: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    video_encoding_batch_size: int = 1
    encoder_threads: int | None = None


def _episode_value(episodes, key: str, episode_index: int):
    values = episodes[key]
    try:
        return values.iloc[episode_index]
    except AttributeError:
        return values[episode_index]


def _episode_task(episodes, episode_index: int) -> str:
    tasks = _episode_value(episodes, "tasks", episode_index)
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError(f"Episode {episode_index} has no task in metadata.")
        return str(tasks[0])
    return str(tasks)


def _episode_bounds(episodes, episode_index: int) -> tuple[int, int]:
    start = int(_episode_value(episodes, "dataset_from_index", episode_index))
    end = int(_episode_value(episodes, "dataset_to_index", episode_index))
    return start, end


def _episode_action_values(dataset: LeRobotDataset, start: int, end: int) -> np.ndarray:
    action_rows = dataset.hf_dataset.select_columns(ACTION)[start:end][ACTION]
    return np.asarray(action_rows, dtype=float)


def make_episode_trim_plans(
    dataset: LeRobotDataset,
    *,
    config: NeroTrimConfig,
) -> tuple[list[EpisodeTrimPlan], DatasetTrimSummary]:
    plans: list[EpisodeTrimPlan] = []
    rejected = 0
    processed_frames = 0

    for episode_index in range(dataset.meta.total_episodes):
        raw_start, raw_end = _episode_bounds(dataset.meta.episodes, episode_index)
        action_values = _episode_action_values(dataset, raw_start, raw_end)
        trim_indices = trim_static_head_tail_indices_from_action_array(
            action_values,
            fps=dataset.meta.fps,
            config=config,
        )
        if trim_indices is None:
            rejected += 1
            continue

        keep_start, keep_end = trim_indices
        source_from_index = raw_start + keep_start
        source_to_index = raw_start + keep_end
        kept_frames = source_to_index - source_from_index
        processed_frames += kept_frames

        plans.append(
            EpisodeTrimPlan(
                source_episode_index=episode_index,
                output_episode_index=len(plans),
                source_from_index=source_from_index,
                source_to_index=source_to_index,
                raw_frames=raw_end - raw_start,
                kept_frames=kept_frames,
                task=_episode_task(dataset.meta.episodes, episode_index),
            )
        )

    summary = DatasetTrimSummary(
        original_episodes=dataset.meta.total_episodes,
        original_frames=dataset.meta.total_frames,
        processed_episodes=len(plans),
        processed_frames=processed_frames,
        rejected_episodes=rejected,
    )
    return plans, summary


def format_trim_summary(summary: DatasetTrimSummary) -> str:
    return "\n".join(
        [
            f"原数据条数: {summary.original_episodes}",
            f"原数据总帧数: {summary.original_frames}",
            f"处理后数据条数: {summary.processed_episodes}",
            f"处理后总帧数: {summary.processed_frames}",
            f"跳过数据条数: {summary.rejected_episodes}",
        ]
    )


def _resolve_output(cfg: NeroTrimDualJointDatasetConfig) -> tuple[str, Path]:
    output_repo_id = cfg.new_repo_id or f"{cfg.repo_id}_trimmed"
    output_root = cfg.new_root or cfg.root.with_name(f"{cfg.root.name}_trimmed")
    output_root = Path(output_root)
    if output_root.resolve() == cfg.root.resolve():
        raise ValueError("new_root must be different from root; refusing to overwrite the source dataset.")
    return output_repo_id, output_root


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if not output_root.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}. Use --overwrite=true to replace it.")
    shutil.rmtree(output_root)


def make_writer_frame(item: dict, *, features: dict[str, dict], task: str) -> dict:
    frame = {}
    for key in features:
        if key in AUTO_FEATURES:
            continue
        value = item[key]
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        frame[key] = value
    frame["task"] = task
    return frame


def write_trimmed_dataset(
    source: LeRobotDataset,
    plans: list[EpisodeTrimPlan],
    *,
    output_repo_id: str,
    output_root: Path,
    cfg: NeroTrimDualJointDatasetConfig,
) -> LeRobotDataset:
    _prepare_output_root(output_root, overwrite=cfg.overwrite)
    target = LeRobotDataset.create(
        output_repo_id,
        source.meta.fps,
        features=source.meta.features,
        root=output_root,
        robot_type=source.meta.robot_type,
        use_videos=len(source.meta.video_keys) > 0,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
        batch_encoding_size=cfg.video_encoding_batch_size,
        encoder_threads=cfg.encoder_threads,
    )

    manager = VideoEncodingManager(target)
    manager.__enter__()
    try:
        for plan in tqdm(plans, desc="Writing trimmed episodes"):
            for frame_index in range(plan.source_from_index, plan.source_to_index):
                item = source[frame_index]
                target.add_frame(make_writer_frame(item, features=target.features, task=plan.task))
            target.save_episode()
    except BaseException as exc:
        manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        manager.__exit__(None, None, None)

    target.finalize()
    return LeRobotDataset(output_repo_id, root=output_root, return_uint8=True)


def postprocess_trim_dual_joint_dataset(cfg: NeroTrimDualJointDatasetConfig) -> LeRobotDataset | None:
    init_logging()
    register_third_party_plugins()

    output_repo_id, output_root = _resolve_output(cfg)
    source = LeRobotDataset(cfg.repo_id, root=cfg.root, return_uint8=True)
    plans, summary = make_episode_trim_plans(source, config=cfg.trim)

    print(format_trim_summary(summary))
    print(f"输出数据集: {output_root}")

    if cfg.dry_run:
        print("dry_run=true，只检测不写入。")
        return None

    if not plans:
        raise ValueError("No valid episodes remain after trimming; refusing to create an empty dataset.")

    result = write_trimmed_dataset(
        source,
        plans,
        output_repo_id=output_repo_id,
        output_root=output_root,
        cfg=cfg,
    )
    print(f"写入完成: {result.meta.total_episodes} 条, {result.meta.total_frames} 帧")
    return result


@parser.wrap()
def main(cfg: NeroTrimDualJointDatasetConfig) -> None:
    postprocess_trim_dual_joint_dataset(cfg)


if __name__ == "__main__":
    main()
