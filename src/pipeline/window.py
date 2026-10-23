"""Sliding window over the live frame stream + frame-source adapters.

SlidingWindowBuffer keeps the newest frames within `window_seconds` of the
latest timestamp. sample(T) maps the buffer's [start, stop] frame span to
T indices via the registry's frame sampler — the SAME sampler the training
Dataset uses, so live clips and training clips are sampled identically.

FrameSources adapt producers to FramePacket streams:
  CameraFrameSource — wraps the existing src.camera_worker.CameraWorker
  ClipFrameSource   — replays a video file as FramePackets at real-time
                      pace (Tier 2) or as fast as wanted (benchmarks)
  SyntheticFrameSource — procedurally generated frames (smoke tests; no
                      camera needed)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import src.registry as registry
from src.capture import CameraConfig, CameraCapture, FramePacket
from src.camera_worker import CameraWorker
from src.frame_buffer import LatestFrameBuffer


class SlidingWindowBuffer:
    """Newest-frames ring over a time span, with drop-invariant sampling."""

    def __init__(self, window_seconds: float = 2.0, max_packets: int = 120) -> None:
        """window_seconds: span kept behind the newest packet's timestamp.
        max_packets: hard cap (2 s at 30 fps = 60; cap guards fast bursts)."""
        self.window_seconds = float(window_seconds)
        self._packets: list[FramePacket] = []
        self._max = max_packets

    def push(self, packet: FramePacket) -> None:
        """Add one packet; drop packets older than the window. O(1) amortized."""
        self._packets.append(packet)
        cutoff = packet.timestamp_ns - int(self.window_seconds * 1e9)
        # Packets arrive in timestamp order (single producer); trim the head.
        while self._packets and self._packets[0].timestamp_ns < cutoff:
            self._packets.pop(0)
        if len(self._packets) > self._max:
            del self._packets[: len(self._packets) - self._max]

    def __len__(self) -> int:
        return len(self._packets)

    def newest(self) -> FramePacket | None:
        return self._packets[-1] if self._packets else None

    def sample(self, T: int) -> list[FramePacket]:
        """T packets spread across the window, oldest to newest.

        Uses the registry frame sampler on the buffer's frame-index span
        (frame_id as the index space, mirroring training's use of video
        frame numbers). Fewer packets than T are REPEATED by the sampler's
        documented behaviour — never errors mid-demo.
        Output: list of T FramePackets.
        """
        if not self._packets:
            return []
        first, last = self._packets[0], self._packets[-1]
        span = max(last.frame_id - first.frame_id, 0)
        fps = len(self._packets) / max(self.window_seconds, 1e-6)  # observed
        sampler = registry.get("frame_sampler")
        indices = sampler.sample_indices(0, span, T, max(fps, 1.0))
        base = first.frame_id
        by_id = {p.frame_id: p for p in self._packets}
        out: list[FramePacket] = []
        for idx in indices.tolist():        # (T,) int64 -> python ints
            fid = base + int(idx)
            if fid in by_id:
                out.append(by_id[fid])
            else:  # frame ids can skip on drops; fall back to nearest
                nearest = min(by_id, key=lambda k: abs(k - fid))
                out.append(by_id[nearest])
        return out


class CameraFrameSource:
    """Live camera via the existing worker/buffer layer (behavior of
    src/capture|frame_buffer|camera_worker is untouched)."""

    def __init__(self, source="/dev/video0", width: int = 640, height: int = 480,
                 fps: int = 30, pixel_format: str = "MJPG") -> None:
        self.config = CameraConfig(source=source, width=width, height=height,
                                   fps=fps, pixel_format=pixel_format)
        self.buffer = LatestFrameBuffer()
        self.worker: CameraWorker | None = None

    def start(self) -> None:
        if self.worker is not None:
            raise RuntimeError("camera source already started")
        self.worker = CameraWorker(self.config, self.buffer)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def latest(self, timeout: float = 1.0) -> FramePacket | None:
        """Newest frame (None on timeout — camera loss is survivable)."""
        return self.buffer.get(timeout=timeout)


class ClipFrameSource:
    """Replays a video file as FramePackets.

    real_time=True paces frames at the file's fps (Tier 2 demo replay);
    False returns frames as fast as decoded (benchmarks). frame_id counts
    from 1 like the camera; timestamp_ns is a monotonic real clock so the
    sliding window behaves exactly as live.
    """

    def __init__(self, path, real_time: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"clip not found: {self.path}")
        self.real_time = real_time
        self.cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._t0_ns: int | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open clip: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0  # 0 -> assume 25
        self._frame_id = 0
        self._t0_ns = None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def next_frame(self) -> FramePacket | None:
        """Next frame or None at end of file."""
        if self.cap is None:
            raise RuntimeError("clip source not started")
        if self.real_time:
            # Pace: frame n belongs at n/fps seconds after the first read.
            if self._t0_ns is None:
                self._t0_ns = time.monotonic_ns()
            target = self._t0_ns + int((self._frame_id / self.fps) * 1e9)
            now = time.monotonic_ns()
            if target > now:
                time.sleep((target - now) / 1e9)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._frame_id += 1
        return FramePacket(frame_id=self._frame_id,
                           timestamp_ns=time.monotonic_ns(), image=frame)


class SyntheticFrameSource:
    """Procedural frames for smoke tests: a moving bright box on a dark
    background, timestamped like a real camera. No hardware needed."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 15.0,
                 moving: bool = True) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.moving = moving
        self._frame_id = 0
        self._t0_ns: int | None = None

    def start(self) -> None:
        self._frame_id = 0
        self._t0_ns = None

    def stop(self) -> None:
        pass

    def next_frame(self, color_by_class: int | None = None) -> FramePacket:
        """One frame; color_by_class in [0..7] tints the box (matcher tests)."""
        if self._t0_ns is None:
            self._t0_ns = time.monotonic_ns()
        target = self._t0_ns + int((self._frame_id / self.fps) * 1e9)
        now = time.monotonic_ns()
        if target > now:
            time.sleep((target - now) / 1e9)
        self._frame_id += 1
        img = np.zeros((self.height, self.width, 3), np.uint8)
        t = self._frame_id
        x = (int(t * 8) % (self.width - 80)) if self.moving else self.width // 2 - 40
        palette = [(255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60),
                   (255, 60, 255), (60, 255, 255), (200, 120, 40), (240, 240, 240)]
        color = palette[color_by_class % 8] if color_by_class is not None else (200, 200, 200)
        img[self.height // 2 - 40:self.height // 2 + 40, x:x + 80] = color
        return FramePacket(frame_id=self._frame_id,
                           timestamp_ns=time.monotonic_ns(), image=img)
