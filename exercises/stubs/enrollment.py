"""OWNER EXERCISE STUB — copy to core/enrollment.py and fill in.

Reference: reference/enrollment_ref.py. Worked example:
docs/worked/enrollment.md. Equivalence check:
python3 -m pytest tests/equivalence --impl core

The two ideas: prototype = re-normalized mean of L2-normalized embeddings;
match = dot product against prototypes (= cosine similarity on unit
vectors), argmax, "unknown" below threshold.
"""

from __future__ import annotations

import numpy as np


class IngredientMatcher:
    """Prototype matcher over ingredient embeddings."""

    def __init__(self, embed_fn, threshold: float = 0.5) -> None:
        """embed_fn: list[HxWx3 uint8] -> (N, D) float32 L2-normalized rows.
        threshold: cosine-similarity cutoff in [-1, 1]."""
        raise NotImplementedError("owner fills this in")

    def enroll(self, name: str, frames: list[np.ndarray]) -> None:
        """Store/replace one prototype from frames.

        Input: frames list of HxWx3 uint8 (any H, W). ValueError on empty.
        Effect: prototype row (D,) unit-length appended/replaced."""
        raise NotImplementedError("owner fills this in")

    def match(self, image: np.ndarray) -> tuple[str, float]:
        """Best match for one HxWx3 uint8 image.

        Output: (name, cosine_score); ("unknown", 0.0) when nothing is
        enrolled; ("unknown", best_score) when below threshold."""
        raise NotImplementedError("owner fills this in")

    def save(self, path) -> None:
        """Persist names, prototypes (K, D), threshold to `path` (.npz)."""
        raise NotImplementedError("owner fills this in")

    def load(self, path) -> None:
        """Restore state saved by save()."""
        raise NotImplementedError("owner fills this in")
