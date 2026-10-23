"""Reference scan-phase matcher: enroll ingredients from a few webcam
frames, then identify the ingredient in hand by cosine similarity to the
stored prototypes.

Study material (docs/worked/enrollment.md). Two ideas only:
  prototype = mean of L2-normalized embeddings, re-normalized (unit vector);
  match     = dot product with each prototype (= cosine similarity, because
              both sides are unit vectors), argmax, threshold to "unknown".

Contract (brief section 5.1):
    IngredientMatcher(embed_fn, threshold: float)
      .enroll(name: str, frames: list[np.ndarray HxWx3]) -> None
      .match(image: np.ndarray HxWx3) -> tuple[str, float]
      .save(path) / .load(path)
    embed_fn: list[HxWx3 uint8] -> np.ndarray (N, D) float32, L2-normalized
    rows (contract of the caller; we defensively re-normalize anyway).

Edge cases (covered in tests and the worked doc):
  - match() before any enroll() -> ("unknown", 0.0).
  - score below threshold -> ("unknown", score) — the score is still the
    best similarity found, so the UI/debug drawer can show it.
  - enroll() with an empty frame list raises ValueError.
  - duplicate enroll(name) replaces the old prototype (retake flow).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale rows of (N, D) to unit length.

    Input (N, D) float; output (N, D) float32 with row norms 1. Zero rows
    stay zero (guarded division; a zero embedding is meaningless anyway and
    will simply match nothing).
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)   # (N, 1) float
    safe = np.maximum(norms, 1e-12)                    # (N, 1) avoid /0
    return (x / safe).astype(np.float32)               # (N, D) float32


class IngredientMatcher:
    """Prototype matcher over ingredient embeddings (nearest prototype)."""

    def __init__(self, embed_fn, threshold: float = 0.5) -> None:
        """
        embed_fn: callable list[HxWx3 uint8] -> (N, D) L2-normalized float32.
        threshold: float in [-1, 1]; similarities below it return "unknown".
        """
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        self.embed_fn = embed_fn
        self.threshold = float(threshold)
        self.names: list[str] = []                      # (K,) ingredient names
        self.prototypes: np.ndarray | None = None       # (K, D) unit rows

    # -- enrollment ---------------------------------------------------------

    def enroll(self, name: str, frames: list[np.ndarray]) -> None:
        """Store one prototype for `name` from `frames`.

        Input: frames list of HxWx3 uint8 images (any H, W per frame).
        Effect: names gains/updates `name`; prototypes keeps (K, D) unit rows
        where the row for `name` is the re-normalized mean embedding.
        Raises ValueError on empty frames.
        """
        if not frames:
            raise ValueError(f"enroll('{name}') got an empty frame list")
        emb = self.embed_fn(frames)                     # (N, D) float32
        if emb.ndim != 2 or emb.shape[0] != len(frames):
            raise ValueError(
                f"embed_fn returned {emb.shape}, expected ({len(frames)}, D)")
        emb = _l2_normalize(emb)                        # (N, D) unit rows
        proto = emb.mean(axis=0)                        # (D,) mean direction
        norm = float(np.linalg.norm(proto))
        proto = (proto / max(norm, 1e-12)).astype(np.float32)  # (D,) unit

        if name in self.names:
            i = self.names.index(name)                  # retake replaces
            self.prototypes[i] = proto
        else:
            self.names.append(name)
            if self.prototypes is None:
                self.prototypes = proto[None, :]        # (1, D)
            else:
                self.prototypes = np.vstack([self.prototypes,
                                             proto[None, :]])  # (K+1, D)

    # -- matching -----------------------------------------------------------

    def match(self, image: np.ndarray) -> tuple[str, float]:
        """Best-matching ingredient for one HxWx3 uint8 image.

        Output: (name, score) — score is the cosine similarity of the best
        prototype, or 0.0 with name "unknown" when nothing is enrolled or
        the best score is below threshold.
        """
        if not self.names or self.prototypes is None:
            return "unknown", 0.0
        emb = self.embed_fn([image])                    # (1, D) float32
        emb = _l2_normalize(emb)                        # (1, D)
        # Dot products of unit vectors = cosine similarities: (1, D) @ (D, K)
        sims = emb @ self.prototypes.T                  # (1, K)
        best = int(np.argmax(sims[0]))                  # scalar index
        score = float(sims[0, best])
        if score < self.threshold:
            return "unknown", score
        return self.names[best], score

    # -- persistence ----------------------------------------------------------

    def save(self, path) -> None:
        """Save names + prototypes + threshold to `path` as .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(self.names, dtype=object),
                 prototypes=self.prototypes,
                 threshold=np.array(self.threshold))

    def load(self, path) -> None:
        """Restore matcher state saved by save()."""
        data = np.load(Path(path), allow_pickle=True)
        self.names = list(data["names"].tolist())
        self.prototypes = data["prototypes"]            # (K, D) float32
        self.threshold = float(data["threshold"])
