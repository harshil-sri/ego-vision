"""OWNER EXERCISE STUB — copy to core/metrics.py and fill in the bodies.

The reference implementation (study it first, then close it):
reference/metrics_ref.py. Worked example with hand numbers:
docs/worked/metrics.md. Your version is checked against the same shared
vectors by:  python3 -m pytest tests/equivalence --impl core

Conventions: scores (N, C) float, higher = more confident; pred/y (N,) int
class ids. All functions pure (no side effects).

NOTE for your implementation: keep the exact signatures below.
"""

from __future__ import annotations

import numpy as np


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of samples whose true class is in the top-k scores.

    Input: scores (N, C) float; y (N,) int; k >= 1.
    Output: float in [0, 1].
    Hint: np.argsort(scores, axis=1)[:, -k:][:, ::-1] gives (N, k) class
    ids in descending confidence; compare each row against y.
    """
    raise NotImplementedError("owner fills this in")


def per_class_accuracy(pred: np.ndarray, y: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Accuracy restricted to samples of each true class.

    Input: pred (N,), y (N,), num_classes C.
    Output: (C,) float; classes absent from y report 0.0 (never NaN).
    """
    raise NotImplementedError("owner fills this in")


def mean_class_accuracy(pred: np.ndarray, y: np.ndarray,
                        num_classes: int) -> float:
    """Mean of per-class accuracy over classes PRESENT in y only.

    Output: scalar float; 0.0 for empty y.
    """
    raise NotImplementedError("owner fills this in")


def confusion_matrix(pred: np.ndarray, y: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """(C, C) int64; rows = true class, columns = predicted class."""
    raise NotImplementedError("owner fills this in")


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class in y.

    Output: float; 0.0 for empty y. (Why it exists: on long-tailed data
    like EPIC, majority accuracy can look 'good' while the model is
    useless — every result table shows it next to real accuracy.)
    """
    raise NotImplementedError("owner fills this in")
