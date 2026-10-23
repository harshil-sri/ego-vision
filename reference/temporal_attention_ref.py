"""Reference temporal attention: pool a sequence of per-frame features into
one clip vector using learned attention weights.

Study material (docs/worked/temporal_attention.md). Shapes on every step.

Contract (brief section 5.1):
    TemporalAttention(D).forward(x) with x of shape (B, T, D)
    -> (pooled (B, D), weights (B, T))   weights sum to 1 over T.

Design (additive "Bahdanau-style" attention over T positions):
    score_t = v^T tanh(W x_t + b)            one scalar per frame
    weights = softmax over t of score        (B, T)
    pooled  = sum_t weights_t * x_t          (B, D)
With a residual-style gated option used by TSM's online models is NOT
included; this is the minimal attention the owner can defend.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Attention pooling over the time axis of (B, T, D) frame features.

    Parameters:
        D: feature dimension (must match x's last dimension).
    Learnables:
        W: Linear(D -> D, bias) projecting each frame feature.
        v: Linear(D -> 1, no bias) turning each projection into one score.
    """

    def __init__(self, D: int) -> None:
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        self.D = D
        self.W = nn.Linear(D, D, bias=True)   # weights (D, D), bias (D,)
        self.v = nn.Linear(D, 1, bias=False)  # weights (1, D), no bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool (B, T, D) -> ((B, D), (B, T)).

        x: float tensor (B, T, D), any batch B >= 1, sequence T >= 1.
        Returns:
            pooled: (B, D) attention-weighted average of the T frames.
            weights: (B, T) softmax attention weights; each row sums to 1.
        """
        if x.ndim != 3 or x.shape[-1] != self.D:
            raise ValueError(f"x must be (B, T, {self.D}), got {tuple(x.shape)}")

        B, T, D = x.shape                       # scalars
        proj = torch.tanh(self.W(x))            # (B, T, D) one projection/frame
        scores = self.v(proj)                   # (B, T, 1) one score/frame
        scores = scores.squeeze(-1)             # (B, T)    drop trailing 1
        # Softmax over the time axis only (dim=1); rows of the result sum to 1.
        weights = torch.softmax(scores, dim=1)  # (B, T)
        # Weighted sum over time: broadcasting (B, T, 1) * (B, T, D) -> (B, T, D)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, D)
        return pooled, weights
