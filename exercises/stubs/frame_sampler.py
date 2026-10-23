"""OWNER EXERCISE STUB — copy to core/frame_sampler.py and fill in.

Reference: reference/frame_sampler_ref.py. Worked example:
docs/worked/frame_sampler.md. Equivalence check:
python3 -m pytest tests/equivalence --impl core

The stop_frame convention (inclusive vs exclusive) is decision S1 in
DECISIONS_NEEDED.md — the reference default is INCLUSIVE (matches EPIC
annotations); keep the parameter so both conventions are testable.
"""

from __future__ import annotations

import numpy as np


def sample_indices(start_frame: int, stop_frame: int, T: int, fps: float,
                   stop_inclusive: bool = True) -> np.ndarray:
    """Pick T frame indices spread evenly across the clip.

    Input: start_frame/stop_frame int (annotation space), T >= 1, fps > 0
    (kept for interface stability; indices live in frame space).
    Output: (T,) int64, sorted ascending, every index inside the clip.

    Required behaviour:
      - exactly T indices, even when T exceeds the clip length (then
        indices repeat near the ends — never error, never leave the clip);
      - segment-centre sampling: split the clip into T equal segments,
        take each segment's centre, round to nearest frame, clip to range;
      - single-frame clips (start == stop, inclusive) return start T times.
    """
    raise NotImplementedError("owner fills this in")
