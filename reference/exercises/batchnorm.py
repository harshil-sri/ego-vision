"""BatchNorm forward exercise: train mode (batch statistics, update running
stats) and eval mode (running statistics) — compared against
nn.BatchNorm2d in tests/test_exercises.py.

Study material (docs/worked/batchnorm.md). The demo runs at batch size 1;
this file is why model.eval() matters there:

  TRAIN mode normalizes by THIS batch's mean/var. At batch size 1 the
  batch mean IS the sample, so every normalized value becomes ~0 and the
  BN output is just bias (garbage) — and the variance estimate is
  meaningless.

  EVAL mode normalizes by the RUNNING mean/var accumulated during
  training — fixed constants, safe at any batch size including 1.

Shapes: x (N, C, H, W) float32 (2D input (N, C) also supported — set
spatial dims to []). gamma (C,), beta (C,). Output same shape as x.
"""

from __future__ import annotations

import numpy as np


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      running_mean: np.ndarray, running_var: np.ndarray,
                      momentum: float = 0.1, eps: float = 1e-5,
                      train: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BatchNorm forward in numpy.

    Inputs:
        x: (N, C) or (N, C, H, W) float32.
        gamma, beta: (C,) scale and shift.
        running_mean, running_var: (C,) — UPDATED IN PLACE when train=True
        (returns the updated arrays; caller stores them, mirroring PyTorch).
        momentum: weight of NEW stats in the running update
        (PyTorch semantics: running = (1 - momentum) * running + momentum * batch).
        eps: numerical floor inside the sqrt.
        train: True -> batch stats + update running; False -> running stats.
    Output: (out same shape as x, running_mean (C,), running_var (C,)).
    """
    orig_shape = x.shape                          # e.g. (N, C, H, W)
    if x.ndim == 2:
        xf = x                                    # (N, C)
        reduce_axes = (0,)                        # normalize over N
    elif x.ndim == 4:
        xf = x.transpose(0, 2, 3, 1).reshape(-1, orig_shape[1])
        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C): one ROW per spatial
        # position, one COLUMN per channel — statistics go down columns.
        reduce_axes = (0,)
    else:
        raise ValueError(f"expected (N, C) or (N, C, H, W), got {orig_shape}")

    if train:
        mean = xf.mean(axis=reduce_axes)          # (C,) per-channel mean
        var = xf.var(axis=reduce_axes)            # (C,) biased variance
        m = xf.shape[0]                           # samples per channel (N*H*W)
        # PyTorch subtlety: normalization uses the BIASED variance above,
        # but the RUNNING variance is updated with the UNBIASED one
        # (multiplied by m/(m-1)). Forgetting this makes eval-mode outputs
        # drift from nn.BatchNorm2d — the test catches it. At m == 1 the
        # unbiased estimate is undefined (PyTorch instead REFUSES to train
        # on 1 value per channel; we keep the biased one so the demo can
        # show the degeneracy instead of crashing).
        unbiased_var = var * (m / (m - 1)) if m > 1 else var
        # Update running stats in place (PyTorch momentum convention).
        running_mean *= (1 - momentum)
        running_mean += momentum * mean
        running_var *= (1 - momentum)
        running_var += momentum * unbiased_var
    else:
        mean = running_mean.copy()                # (C,) fixed constants
        var = running_var.copy()

    xhat = (xf - mean) / np.sqrt(var + eps)       # (N*H*W, C) standardized
    out = gamma * xhat + beta                     # (N*H*W, C) scale+shift

    if x.ndim == 4:
        # Undo the reshape/transpose: (N*H*W, C) -> (N, H, W, C) -> (N, C, H, W)
        out = out.reshape(orig_shape[0], orig_shape[2], orig_shape[3],
                          orig_shape[1]).transpose(0, 3, 1, 2)
    return out.astype(x.dtype), running_mean, running_var


def batchnorm_train_vs_eval_demo(x: np.ndarray) -> dict[str, np.ndarray]:
    """One-call demo of the batch-size-1 trap (used in the worked doc).

    Input: x (N, C) — use N >= 2 rows; the demo takes row 0 as the
    single-sample batch.

    Output dict:
      out_train_bs1: BN of ONE (1, C) sample in train mode — the batch mean
        IS the sample, so xhat = 0 exactly and the output is just beta
        (here 0): the layer erases the input entirely.
      out_eval_bs1: eval mode on the same sample with running stats
        (0, 1): output = x itself (scaled) — information preserved.
    Note: for (N, C, H, W) conv inputs at N=1 the stats pool over the H*W
    positions, so the collapse is partial rather than exact — but the
    statistics still come from a single image, which is exactly the
    unrepresentative-statistics problem eval mode avoids.
    """
    C = x.shape[1]
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    rm = np.zeros(C, dtype=np.float32)
    rv = np.ones(C, dtype=np.float32)
    single = x[:1]                                 # (1, C) batch of ONE
    out_train, _, _ = batchnorm_forward(single, gamma, beta, rm.copy(),
                                        rv.copy(), train=True)
    out_eval, _, _ = batchnorm_forward(single, gamma, beta, rm.copy(),
                                       rv.copy(), train=False)
    return {"out_train_bs1": out_train, "out_eval_bs1": out_eval}
