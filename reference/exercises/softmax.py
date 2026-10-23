"""Softmax loss + gradient exercise (CS231n A1 style): naive with loops,
then vectorized, then a numerical gradient check.

Study material (docs/worked/softmax.md). Why we subtract the max before
exp: exp(1000) overflows float64 (max ~1.8e308); shifting scores by their
row max leaves softmax UNCHANGED (it cancels in the normalizer) but keeps
every exponent <= 0, so no overflow. This is the answer to "why subtract
the max" — the math is invariant, the float range is not.

Shapes: X (N, D), W (D, C), y (N,) int in [0, C).
Loss: scalar float64. dW: (D, C) float64, same as W.
"""

from __future__ import annotations

import numpy as np


def softmax_loss_naive(X: np.ndarray, W: np.ndarray, y: np.ndarray,
                       reg: float = 0.0) -> tuple[float, np.ndarray]:
    """Loop version — trace it by hand on the 2x2 example in the worked doc.

    Returns (loss scalar, dW (D, C)).
    """
    N, D = X.shape                     # scalars
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)              # (D, C)
    for i in range(N):
        scores = X[i] @ W              # (C,) class scores for sample i
        scores = scores - scores.max() # subtract max: see module docstring
        exp = np.exp(scores)           # (C,) all in (0, 1]
        probs = exp / exp.sum()        # (C,) softmax; sums to 1
        loss += -np.log(probs[y[i]])   # scalar: -log of the correct class
        # Gradient of -log(probs[y]) wrt scores is (probs - onehot_y):
        dscores = probs.copy()         # (C,)
        dscores[y[i]] -= 1.0           # subtract 1 at the correct class
        # Chain rule to W: dL/dW += x_i outer dscores -> (D, C)
        dW += np.outer(X[i], dscores)
    loss = loss / N + reg * np.sum(W * W)      # mean + L2 regularization
    dW = dW / N + 2 * reg * W
    return float(loss), dW


def softmax_loss_vectorized(X: np.ndarray, W: np.ndarray, y: np.ndarray,
                            reg: float = 0.0) -> tuple[float, np.ndarray]:
    """Same contract, no Python loops over N.

    Key line: scores[np.arange(N), y] is fancy indexing that picks each
    row's correct-class score — shape (N,).
    """
    N = X.shape[0]
    scores = X @ W                                  # (N, D) @ (D, C) -> (N, C)
    scores = scores - scores.max(axis=1, keepdims=True)  # (N, 1) per-row max
    exp = np.exp(scores)                            # (N, C)
    probs = exp / exp.sum(axis=1, keepdims=True)    # (N, C) rows sum to 1
    loss = float(-np.mean(np.log(probs[np.arange(N), y])))   # scalar
    dscores = probs.copy()                          # (N, C)
    dscores[np.arange(N), y] -= 1.0                 # -1 at correct classes
    dW = X.T @ dscores / N                          # (D, N) @ (N, C) -> (D, C)
    if reg > 0:
        loss += reg * float(np.sum(W * W))
        dW = dW + 2 * reg * W
    return loss, dW


def numerical_gradient(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar f at x.

    f: callable x -> float. x: any-shape array. Output: same shape as x.
    Central difference: (f(x+h e_ij) - f(x-h e_ij)) / (2h) per element.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + h
        fp = f(x)
        x[ix] = old - h
        fm = f(x)
        x[ix] = old
        grad[ix] = (fp - fm) / (2 * h)
        it.iternext()
    return grad
