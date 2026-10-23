"""Session logging: every event to runs/sessions/<timestamp>.jsonl, plus
optional recording of the annotated video to mp4 (the owner's reels).

The logger is an EventBus subscriber — one line of wiring, no engine code.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.events import Event, EventBus


class SessionLogger:
    """Writes events as JSONL; optionally records annotated frames."""

    def __init__(self, runs_root="runs/sessions", record_video: bool = False,
                 fps: float = 15.0) -> None:
        """runs_root: directory for <timestamp>.jsonl files. record_video:
        also write <timestamp>.mp4 of annotated frames pushed via frame()."""
        self.path: Path | None = None
        self.video_path: Path | None = None
        self._fh = None
        self._writer: cv2.VideoWriter | None = None
        self._fps = fps
        self._runs_root = Path(runs_root)
        self._record = record_video
        self.events_written = 0

    def start(self, session_id: str | None = None) -> Path:
        """Open a new session file. Output: the jsonl path."""
        sid = session_id or time.strftime("%Y%m%d-%H%M%S")
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self.path = self._runs_root / f"{sid}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.video_path = self._runs_root / f"{sid}.mp4" if self._record else None
        self.events_written = 0
        return self.path

    def handle_event(self, event: Event) -> None:
        """EventBus subscriber: append one JSON line."""
        if self._fh is None:
            return
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()
        self.events_written += 1

    def frame(self, annotated: np.ndarray) -> None:
        """Add one BGR annotated frame to the recording (if enabled)."""
        if self.video_path is None or annotated is None:
            return
        h, w = annotated.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc,
                                           self._fps, (w, h))
        self._writer.write(annotated)

    def close(self) -> None:
        """Flush and close both sinks."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def attach(bus: EventBus, **kwargs) -> SessionLogger:
    """Create a SessionLogger subscribed to `bus` (one-liner for the app)."""
    logger = SessionLogger(**kwargs)
    bus.subscribe(logger.handle_event)
    return logger
