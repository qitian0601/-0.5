from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class NeroInferenceTraceConfig:
    enabled: bool = False
    dir: str = "lerobot/outputs/nero_inference_records"
    run_name: str = ""
    flush_every: int = 100


def _default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class NeroInferenceTracer:
    def __init__(
        self,
        config: NeroInferenceTraceConfig,
        *,
        meta: dict[str, Any],
        action_names: tuple[str, ...] | list[str],
    ):
        self.config = config
        self.action_names = tuple(action_names)
        self.run_dir: Path | None = None
        self._trace_file = None
        self._replay_file = None
        self._sequence = 0
        self._replay_sequence = 0
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[tuple[str, dict[str, Any]] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._closed = False

        if not config.enabled:
            return
        if config.flush_every <= 0:
            raise ValueError(f"trace.flush_every must be positive, got {config.flush_every}.")

        run_name = config.run_name.strip() or _default_run_name()
        self.run_dir = Path(config.dir).expanduser() / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._trace_file = (self.run_dir / "trace.jsonl").open("w", encoding="utf-8")
        self._replay_file = (self.run_dir / "replay_actions.jsonl").open("w", encoding="utf-8")
        self._queue = queue.SimpleQueue()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        meta_payload = {
            **_jsonable(meta),
            "trace_config": asdict(config),
            "action_names": list(self.action_names),
            "created_time_s": time.time(),
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @property
    def enabled(self) -> bool:
        return self._trace_file is not None

    def _enqueue(self, stream: str, payload: dict[str, Any]) -> None:
        if self._queue is not None:
            self._queue.put((stream, payload))

    def _writer_loop(self) -> None:
        if self._queue is None:
            return
        pending = 0
        while True:
            item = self._queue.get()
            if item is None:
                break
            stream, payload = item
            if stream == "trace":
                file_obj = self._trace_file
            elif stream == "replay":
                file_obj = self._replay_file
            else:
                continue
            if file_obj is None:
                continue
            file_obj.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
            pending += 1
            if pending >= self.config.flush_every:
                self._flush_files()
                pending = 0
        self._flush_files()

    def _flush_files(self) -> None:
        if self._trace_file is not None:
            self._trace_file.flush()
        if self._replay_file is not None:
            self._replay_file.flush()

    def record(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self._trace_file is None:
            return
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            payload = {
                "sequence": self._sequence,
                "time_s": time.time(),
                "perf_counter_s": time.perf_counter(),
                "event": event,
                "data": data or {},
            }
        self._enqueue("trace", payload)

    def record_replay_action(self, action: dict[str, float], *, dt_s: float | None = None) -> None:
        if self._replay_file is None:
            return
        replay_action = {name: float(action[name]) for name in self.action_names}
        with self._lock:
            if self._closed:
                return
            self._replay_sequence += 1
            replay_payload = {
                "sequence": self._replay_sequence,
                "time_s": time.time(),
                "action": replay_action,
            }
            if dt_s is not None:
                replay_payload["dt_s"] = float(dt_s)
        self._enqueue("replay", replay_payload)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._queue is not None:
            self._queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join()
            self._writer_thread = None
        with self._lock:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None
            if self._replay_file is not None:
                self._replay_file.close()
                self._replay_file = None


def load_replay_actions(path: Path, action_names: tuple[str, ...] | list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            action = payload["action"]
            rows.append([float(action[name]) for name in action_names])
    if not rows:
        raise ValueError(f"No replay actions found in {path}")
    return np.asarray(rows, dtype=float)
