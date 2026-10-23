"""Reference metrics: top-k, per-class / mean-class accuracy, confusion
matrix, majority baseline.

Study material (docs/worked/metrics.md). Conventions used everywhere:

  scores: np.ndarray of shape (N, C), higher = more confident, one row per
          sample, one column per class. Softmax probabilities fit; raw
          logits also fit (argmax is invariant to softmax).
  pred:   np.ndarray of shape (N,), predicted class ids 0..C-1.
  y:      np.ndarray of shape (N,), true class ids 0..C-1.

All functions are pure: no globals, no in-place edits.
"""

from __future__ import annotations

import numpy as np


def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of samples whose true class is among the top-k scores.

    Inputs: scores (N, C) float; y (N,) integer; k >= 1.
    Output: float in [0, 1].
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be (N, C), got {scores.shape}")
    if y.shape != (scores.shape[0],):
        raise ValueError(f"y {y.shape} does not match scores rows {scores.shape[0]}")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k={k} out of range for {scores.shape[1]} classes")

    # argsort ascending along classes, take last k columns, flip -> descending
    # (N, k) class ids: column 0 is the argmax prediction.
    topk = np.argsort(scores, axis=1)[:, -k:][:, ::-1]     # (N, k) int64
    # Is the true class anywhere in this sample's top-k row?
    hits = (topk == y[:, None]).any(axis=1)                # (N,) bool
    return float(hits.mean())                              # scalar float


def per_class_accuracy(pred: np.ndarray, y: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Accuracy restricted to samples of each true class.

    Inputs: pred (N,), y (N,), num_classes C.
    Output: (C,) float; entry c = correct_c / count_c, or 0.0 where class c
    never appears in y (documented: absent classes report 0.0, never NaN,
    so downstream means/tables stay well-defined).
    """
    correct = np.zeros(num_classes, dtype=np.int64)        # (C,) counts right
    total = np.zeros(num_classes, dtype=np.int64)          # (C,) counts seen
    for c in range(num_classes):                           # plain loop first
        mask = y == c                                     # (N,) bool
        total[c] = mask.sum()                              # scalar
        correct[c] = (pred[mask] == c).sum()               # scalar
    acc = np.zeros(num_classes, dtype=np.float64)          # (C,)
    nonzero = total > 0
    acc[nonzero] = correct[nonzero] / total[nonzero]       # only seen classes
    return acc


def mean_class_accuracy(pred: np.ndarray, y: np.ndarray,
                        num_classes: int) -> float:
    """Mean of per-class accuracy over classes PRESENT in y.

    Classes absent from y are ignored (not counted as 0): a fair mean over
    what was actually evaluated. If y is empty, returns 0.0.
    Output: scalar float.
    """
    acc = per_class_accuracy(pred, y, num_classes)         # (C,) float
    present = np.zeros(num_classes, dtype=bool)            # (C,) bool
    for c in range(num_classes):
        present[c] = np.any(y == c)                        # scalar bool
    if not present.any():
        return 0.0
    return float(acc[present].mean())                      # scalar


def confusion_matrix(pred: np.ndarray, y: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """C x C integer matrix; rows = true class, columns = predicted class.

    Entry [c_true, c_pred] counts samples with y == c_true and pred == c_pred.
    Row sums therefore equal per-class sample counts.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # (C, C)
    for t, p in zip(y.tolist(), pred.tolist()):                # plain loop
        if not (0 <= t < num_classes and 0 <= p < num_classes):
            raise ValueError(f"class id out of range: y={t}, pred={p}")
        cm[t, p] += 1
    return cm


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class in y.

    Output: scalar float = max class count / N. With ties, any majority
    choice gives the same accuracy. Empty y -> 0.0.
    """
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)           # (U,), (U,) sorted
    return float(counts.max() / y.size)
