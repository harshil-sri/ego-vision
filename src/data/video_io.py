"""Decode specific frames from a video file with OpenCV.

Two decode strategies with different cost/accuracy trade-offs:

  Sequential (default, ``fast_seek=False``):
      cap.grab() through EVERY frame from 0 up to the largest requested
      index, calling cap.retrieve() only at the T needed indices. grab()
      decodes the frame but skips the (expensive) color conversion and
      memcpy, retrieve() finishes the job for the frames we keep.
      Cost model: roughly O(max_index) grabs + O(T) retrieves, i.e. the cost
      depends on WHERE in the video the segment lives (a segment starting at
      frame 100k pays 100k grabs) and barely on T. Always frame-accurate.

  Fast seek (``fast_seek=True``):
      cap.set(CAPPROP_POS_FRAMES, idx) then read() per index, so the decoder
      jumps near the target instead of walking from 0. Cost model: O(T) seeks
      with a small constant per seek. ACCURACY RISK: for codecs with
      B-frames (typical H.264) OpenCV's frame-number seek relies on the
      container's timestamp table; on some files it can land one or more
      frames off. Fine for exploration; use the sequential mode when the
      frames must match the annotation indices exactly (training data).

Both modes return RGB (OpenCV decodes BGR internally; we convert).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _ffprobe(path: str, entries: str, stream: str = "v:0") -> str:
    """Run ffprobe and return its stripped stdout ("" on any failure).

    Inputs: path to a media file; entries like "stream=r_frame_rate".
    Output: str, first line of ffprobe output.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream,
             "-show_entries", entries,
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def decode_frames(path: str, frame_indices: np.ndarray,
                  fast_seek: bool = False) -> np.ndarray:
    """Decode specific frames of a video as RGB uint8.

    Inputs:
        path: video file path (e.g. .../P01/videos/P01_01.MP4).
        frame_indices: array-like of frame numbers, shape (T,), dtype
            int64-compatible, values >= 0, any order, duplicates allowed.
        fast_seek: see module docstring for the cost/accuracy trade-off.

    Output: np.ndarray of shape (T, H, W, 3), dtype uint8, RGB channel
        order, row i corresponding to frame_indices[i] (input order kept).

    Raises:
        FileNotFoundError: path does not exist (message lists the path and
            hints that the bulk download may still be running).
        RuntimeError: the video has fewer frames than requested (or is
            truncated); the message names the video and the first missing
            index. The function NEVER silently returns fewer rows.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"video file not found: {path}. If the bulk download is still "
            f"running, the file may not have landed yet (partials live "
            f"under .cache/ with a .incomplete suffix); check with "
            f"os.path.exists before building the dataset."
        )

    # Normalise to a flat int64 array; keep the ORIGINAL order for output.
    idx = np.asarray(frame_indices, dtype=np.int64).reshape(-1)  # (T,) int64
    if idx.size == 0:
        raise ValueError("frame_indices must contain at least one index")
    if idx.min() < 0:
        raise ValueError(f"negative frame index in {idx.tolist()}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {path}")

    try:
        if fast_seek:
            frames = _decode_fast_seek(cap, idx, path)   # (T,H,W,3) uint8
        else:
            frames = _decode_sequential(cap, idx, path)  # (T,H,W,3) uint8
    finally:
        cap.release()

    return frames


def _retrieve_rgb(cap, path: str, index: int) -> np.ndarray:
    """retrieve() the current frame and convert BGR->RGB.

    Inputs: opened VideoCapture positioned on the frame to read.
    Output: np.ndarray (H, W, 3) uint8 RGB.
    """
    ok, frame_bgr = cap.retrieve()
    if not ok or frame_bgr is None:
        raise RuntimeError(
            f"{path}: failed to retrieve() frame {index}"
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8


def _decode_sequential(cap, idx: np.ndarray, path: str) -> np.ndarray:
    """Walk frames 0..max(idx) with grab(), retrieve() only where needed.

    Inputs: cap (opened VideoCapture), idx (T,) int64 (any order, dups ok).
    Output: (T, H, W, 3) uint8 RGB aligned with idx order.
    """
    # Unique sorted targets: the walk visits each needed frame exactly once.
    uniq = np.unique(idx)                      # (U,) int64 ascending
    frames_by_index: dict[int, np.ndarray] = {}

    u = 0                # pointer into uniq
    pos = 0              # frame number the capture is currently holding
    while u < uniq.size:
        if not cap.grab():                     # EOF before a needed frame
            break
        if pos == int(uniq[u]):
            frames_by_index[int(uniq[u])] = _retrieve_rgb(cap, path, int(uniq[u]))
            u += 1
        pos += 1

    missing = [int(i) for i in uniq if int(i) not in frames_by_index]
    if missing:
        raise RuntimeError(
            f"{path}: could not decode frame(s) {missing[:5]} "
            f"({len(missing)} total); video ends before the largest "
            f"requested index or the file is truncated."
        )

    # Map back to the caller's (possibly unsorted, duplicated) order.
    return np.stack([frames_by_index[int(i)] for i in idx])  # (T,H,W,3)


def _decode_fast_seek(cap, idx: np.ndarray, path: str) -> np.ndarray:
    """Seek to each index with CAP_PROP_POS_FRAMES and read one frame.

    Inputs: cap (opened VideoCapture), idx (T,) int64 in caller order.
    Output: (T, H, W, 3) uint8 RGB. May land off-by-frames on some codecs
    (see module docstring); prefer sequential mode for annotation-exact work.
    """
    frames = []
    for i in idx.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)    # jump near frame i
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"{path}: fast seek failed to read frame {i} "
                f"(video may be shorter than the index, or the codec's "
                f"seek table is coarse)."
            )
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames)                    # (T,H,W,3) uint8


def probe_fps(path: str) -> float:
    """Return the video's frame rate in fps.

    Input: path to a video file.
    Output: float fps (> 0), e.g. 59.9400599400599 for EPIC 1080p files.

    Strategy: cv2's CAP_PROP_FPS first (fast, no subprocess); if it returns
    0/absent, fall back to ffprobe avg_frame_rate, then r_frame_rate (both
    parsed from a fraction like "30000/1001"). Raises RuntimeError if every
    route fails.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"video file not found: {path}")

    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps > 0:
        return fps

    for entries in ("stream=avg_frame_rate", "stream=r_frame_rate"):
        raw = _ffprobe(path, entries)
        if raw and raw != "N/A":
            try:
                num, _, den = raw.partition("/")
                value = float(num) / (float(den) if den else 1.0)
            except ValueError:
                continue
            if value > 0:
                return value
    raise RuntimeError(f"could not determine fps for {path}")


def probe_frame_count(path: str) -> int:
    """Return the number of video frames, 0 if it cannot be determined.

    Input: path to a video file.
    Output: int frame count from the MP4 header (nb_frames); falls back to
    counting packets (nb_read_packets) when the header lies or is absent.
    This reads metadata only — it never decodes the whole file.
    """
    path = str(path)
    # Fast path: nb_frames from the MP4 header (no decoding).
    raw = _ffprobe(path, "stream=nb_frames")
    try:
        count = int(raw)
        if count > 0:
            return count
    except ValueError:
        pass
    # Slow path: count packets (-count_packets demuxes but does not decode).
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_packets",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=600, check=False,
        )
        return int(out.stdout.strip())
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return 0
