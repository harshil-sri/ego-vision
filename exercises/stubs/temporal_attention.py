"""OWNER EXERCISE STUB — copy to core/temporal_attention.py and fill in.

Reference: reference/temporal_attention_ref.py. Worked example:
docs/worked/temporal_attention.md. Equivalence check:
python3 -m pytest tests/equivalence --impl core

Architecture the harness expects (so weights can be injected for the
equivalence test): nn.Linear D->D with bias named W, nn.Linear D->1 without
bias named v; scores = v(tanh(W(x))); softmax over time; weighted sum.
You may add parameters beyond these, but keep W and v with these shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Attention pooling over the time axis of (B, T, D) frame features."""

    def __init__(self, D: int) -> None:
        """D: feature dimension (>= 1). Learnables: W (D->D, bias), v (D->1,
        no bias)."""
        raise NotImplementedError("owner fills this in")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool (B, T, D) float -> (pooled (B, D), weights (B, T)).

        weights = softmax over dim=1 (time); each row sums to 1.
        pooled = sum_t weights_t * x_t.
        Raise ValueError for wrong ndim or last-dim != D.
        """
        raise NotImplementedError("owner fills this in")
