"""Shared helpers for the equivalence harness (generator + tests both use
these, which is what makes the vectors reproducible)."""

from __future__ import annotations

import numpy as np


def toy_embed(frames: list[np.ndarray]) -> np.ndarray:
    """Deterministic toy embedder standing in for DINOv2 etc.

    Input: list of HxWx3 uint8 frames (length N).
    Output: (N, 3) float32, each row = mean colour / 255, L2-normalized.
    Used ONLY for tests; the real app passes a model-backed embed_fn.
    """
    rows = np.stack([f.reshape(-1, 3).mean(axis=0) for f in frames])  # (N,3)
    rows = (rows / 255.0).astype(np.float32)
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return rows / norms                                   # (N, 3) unit rows
