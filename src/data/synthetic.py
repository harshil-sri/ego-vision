"""Synthetic clip dataset for tests and overfit smoke-runs (no disk needed).

Each item is a short clip whose frames are uniform noise plus a per-class
mean shift on channel 0, so the classes are linearly separable enough that a
small model can drive training accuracy to ~100% (the WP3 --overfit-batch
gate) without any real data.

Determinism: item i uses rng = np.random.default_rng([seed, i]) — a stream
seeded from the pair (seed, i), so item i is identical across runs and
independent of which other items were fetched (safe with DataLoader workers
shuffling). NumPy's Generator streams are reproducible for a fixed version;
they may change across MAJOR numpy upgrades (fine for tests/overfitting).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data


class SyntheticClipDataset(torch.utils.data.Dataset):
    """Deterministic synthetic (clip, label) pairs.

    __init__ inputs:
        num_classes: int, number of classes (default 8, the demo verbs).
        length: int, dataset size; item i has label i % num_classes.
        T, C, H, W: clip tensor dims (default 4, 3, 48, 48 — small so a
            CPU overfit run takes seconds, not minutes).
        seed: int, base seed; item i's stream is default_rng([seed, i]).

    __getitem__(i) outputs:
        clip: torch.float32 (T, C, H, W).
            frame = U[0,1) noise on all channels, PLUS label * 0.11 added to
            channel 0 only. Values are NOT clipped to [0,1] (class 7 shifts
            channel 0 to a mean of ~1.27); separability beats staying in
            range for a synthetic overfit target.
        label: python int = i % num_classes.

    Class means on channel 0: class k -> 0.5 + k * 0.11, so classes 0 and 7
    differ by ~0.77 in expectation — far above the noise (sd ~0.29/sqrt(T*H*W)
    of a clip mean), making a linear classifier on the channel-0 mean enough.
    """

    #: mean shift added to channel 0 per class index (constructor-constant)
    CHANNEL0_SHIFT = 0.11

    def __init__(self, num_classes: int = 8, length: int = 64,
                 T: int = 4, C: int = 3, H: int = 48, W: int = 48,
                 seed: int = 0):
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.length = int(length)
        self.T, self.C, self.H, self.W = int(T), int(C), int(H), int(W)
        self.seed = int(seed)

    def __len__(self) -> int:
        """Number of items. Scalar int."""
        return self.length

    def __getitem__(self, i: int):
        """Return (clip float32 (T,C,H,W), label int) for item i."""
        label = i % self.num_classes

        # Independent per-item stream; re-fetching item i always matches.
        rng = np.random.default_rng([self.seed, i])
        frames = rng.random((self.T, self.H, self.W, self.C),
                            dtype=np.float32)          # (T,H,W,C) in [0,1)
        frames[..., 0] += label * self.CHANNEL0_SHIFT  # class shift, ch 0 only

        clip = torch.from_numpy(frames)                # (T,H,W,C) float32
        clip = clip.permute(0, 3, 1, 2).contiguous()   # (T,C,H,W) float32
        return clip, label
