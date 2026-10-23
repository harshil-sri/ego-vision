"""Reference frame sampler: (start_frame, stop_frame, T, fps) -> frame indices.

This is study material (see docs/worked/frame_sampler.md). Every step carries
a shape comment; the code prefers the obvious loop over clever indexing.

Contract (brief section 5.1):
    sample_indices(start_frame, stop_frame, T, fps) -> np.ndarray[T] int64

Behaviour:
  - Returns exactly T sorted indices, all inside the clip.
  - stop_inclusive=True (default): the clip covers frames
    start_frame..stop_frame, i.e. length = stop_frame - start_frame + 1.
    This matches EPIC-100 annotations (see DECISIONS_NEEDED.md S1 for the
    evidence). stop_inclusive=False switches to the half-open convention.
  - T larger than the clip length: indices repeat (np.linspace naturally
    repeats endpoints); still exactly T values, all within the clip. This is
    the documented behaviour, not an error.
  - fps is accepted for interface completeness (indices live in frame space;
    any temporal arithmetic the caller does needs fps, this function does
    not). It must be positive.

Worked example (docs/worked/frame_sampler.md):
    sample_indices(100, 180, 8, 59.94, stop_inclusive=True)
    -> [105, 115, 125, 135, 146, 156, 166, 176]
    (81 frames split into 8 equal segments of 10.125 frames; each index is
    the centre of its segment, rounded to the nearest frame).
"""

from __future__ import annotations

import numpy as np


def clip_length(start_frame: int, stop_frame: int,
                stop_inclusive: bool = True) -> int:
    """Number of frames covered by the clip.

    Inputs are scalars; output is a scalar int.
    Inclusive: stop-start+1 (e.g. 100..180 -> 81). Exclusive: stop-start (80).
    """
    if stop_inclusive:
        return stop_frame - start_frame + 1
    return stop_frame - start_frame


def sample_indices(start_frame: int, stop_frame: int, T: int, fps: float,
                   stop_inclusive: bool = True) -> np.ndarray:
    """Pick T frame indices spread evenly across the clip.

    Inputs:
        start_frame, stop_frame: int frame numbers (annotation space).
        T: number of frames to sample (>=1).
        fps: video frame rate (>0; unused for the arithmetic, kept for the
            stable interface).
        stop_inclusive: whether stop_frame is part of the clip.
    Output:
        np.ndarray of shape (T,), dtype int64, sorted ascending, every value
        within [start_frame, stop_frame] (inclusive) or
        [start_frame, stop_frame) (exclusive).

    Method: divide the clip into T equal segments and take the centre of each
    ("segment-based centre sampling", the TSN convention). For T points this
    is equivalent to linspace over positions start + (i + 0.5) * L / T for
    i in 0..T-1, rounded to nearest integer and clipped into the clip.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if stop_frame < start_frame:
        raise ValueError(
            f"stop_frame ({stop_frame}) < start_frame ({start_frame})")

    length = clip_length(start_frame, stop_frame, stop_inclusive)  # scalar int
    # Segment centres in *offset* space: (T,) floats in (0, length).
    # Example: length 81, T 8 -> [5.0625, 15.19, ..., 75.94]
    centres = (np.arange(T) + 0.5) * (length / T)          # (T,) float64
    # Round to nearest frame offset and keep inside [0, length-1].
    offsets = np.rint(centres).astype(np.int64)            # (T,) int64
    offsets = np.clip(offsets, 0, length - 1)              # (T,) int64
    indices = (start_frame + offsets).astype(np.int64)     # (T,) int64
    return indices
