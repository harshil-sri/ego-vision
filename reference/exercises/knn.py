"""kNN exercise (CS231n A1 style): L2 distances three ways + predict_labels.

Study material (docs/worked/knn.md). The single most important line for
the unblocking rule: the output is (num_test, num_train), so
dists[i, j] pairs TEST row i with TRAIN row j — fix the output shape
first, everything else follows.

Every version must agree: tests/test_exercises.py checks np.allclose
between the two-loop, one-loop and no-loop versions on random data.

Shapes: X (N, D) float64. Dists (num_test, num_train) float64.
"""

from __future__ import annotations

import numpy as np


def compute_distances_two_loops(X_test: np.ndarray,
                                X_train: np.ndarray) -> np.ndarray:
    """L2 distances, explicit double loop (the paper-and-pencil version).

    X_test: (num_test, D); X_train: (num_train, D).
    Output: (num_test, num_train); dists[i, j] = ||X_test[i] - X_train[j]||_2.
    """
    num_test = X_test.shape[0]                # scalar
    num_train = X_train.shape[0]              # scalar
    dists = np.zeros((num_test, num_train), dtype=np.float64)  # (T, N)
    for i in range(num_test):                 # one row per TEST point
        for j in range(num_train):            # one column per TRAIN point
            diff = X_test[i] - X_train[j]     # (D,) difference vector
            dists[i, j] = np.sqrt(np.sum(diff * diff))  # scalar L2 norm
    return dists


def compute_distances_one_loop(X_test: np.ndarray,
                               X_train: np.ndarray) -> np.ndarray:
    """L2 distances, loop over test points only, vectorized over trains.

    Same contract as the two-loop version. For each test row we broadcast
    one subtraction: (num_train, D) -> square -> sum over D -> sqrt.
    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train), dtype=np.float64)
    for i in range(num_test):
        diff = X_train - X_test[i]            # (num_train, D) broadcast
        dists[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))   # (num_train,)
    return dists


def compute_distances_no_loops(X_test: np.ndarray,
                               X_train: np.ndarray) -> np.ndarray:
    """L2 distances with zero loops via the expansion identity.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b

    Same contract. The middle term is the only matmul:
      test_sq  (num_test, 1)   row norms squared of test points
      train_sq (1, num_train)  row norms squared of train points
      cross    (num_test, num_train) = X_test @ X_train.T
    """
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)    # (T, 1)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True)  # (N, 1)
    cross = X_test @ X_train.T                              # (T, N)
    sq = test_sq + train_sq.T - 2.0 * cross                 # (T, N) broadcast
    # Numerical safety: expansion can go a hair negative on near-identical
    # rows; np.maximum clips before the sqrt.
    return np.sqrt(np.maximum(sq, 0.0))


def predict_labels(dists: np.ndarray, y_train: np.ndarray, k: int = 1) -> np.ndarray:
    """Majority vote among the k nearest training points per test point.

    dists: (num_test, num_train); y_train: (num_train,) int labels; k >= 1.
    Output: (num_test,) int64 predicted labels.
    Ties: np.bincount + argmax resolves them by SMALLEST label id —
    deterministic, and documented so the tests can rely on it.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=np.int64)             # (T,)
    for i in range(num_test):
        # argsort ascending: first k entries = indices of k nearest trains.
        nearest = np.argsort(dists[i])[:k]                  # (k,) train idxs
        votes = y_train[nearest]                            # (k,) labels
        y_pred[i] = np.bincount(votes).argmax()             # scalar
    return y_pred


def knn_accuracy(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Fraction correct. y_pred/y_test: (N,) int."""
    return float(np.mean(y_pred == y_test))
