"""Tests for src/data/synthetic.py (SyntheticClipDataset)."""

from __future__ import annotations

import numpy as np
import torch

from src.data.synthetic import SyntheticClipDataset


def test_shapes_and_dtypes():
    """Default constructor -> (T,C,H,W) = (4,3,48,48) float32."""
    ds = SyntheticClipDataset()
    clip, label = ds[0]
    assert clip.shape == (4, 3, 48, 48)
    assert clip.dtype == torch.float32
    assert isinstance(label, int)


def test_label_is_index_mod_classes():
    """label = i % num_classes for every item."""
    ds = SyntheticClipDataset(num_classes=8, length=64)
    for i in [0, 1, 7, 8, 9, 63]:
        _, label = ds[i]
        assert label == i % 8


def test_determinism():
    """Same index + same seed -> identical clips, across dataset instances."""
    ds_a = SyntheticClipDataset(seed=0)
    ds_b = SyntheticClipDataset(seed=0)
    for i in [0, 17, 63]:
        clip_a, _ = ds_a[i]
        clip_b, _ = ds_b[i]
        assert torch.allclose(clip_a, clip_b)
    # same index, DIFFERENT seed -> different noise
    ds_c = SyntheticClipDataset(seed=1)
    assert not torch.allclose(ds_a[5][0], ds_c[5][0])


def test_class_separability():
    """Channel-0 mean of class 7 clips exceeds class 0 by > 0.5.

    Classes are shifted by label * 0.11 on channel 0, so the expected gap
    is 7 * 0.11 = 0.77; averaging over many frames/pixels keeps the noise
    contribution tiny. This is the property the WP3 overfit gate relies on.
    """
    ds = SyntheticClipDataset(num_classes=8, length=64, seed=0)
    ch0 = {label: [] for label in range(8)}
    for i in range(len(ds)):
        clip, label = ds[i]
        ch0[label].append(float(clip[:, 0].mean()))  # mean over T,H,W
    mean0 = float(np.mean(ch0[0]))
    mean7 = float(np.mean(ch0[7]))
    assert mean7 - mean0 > 0.5
    # and neighbouring classes are also shifted apart on average
    assert float(np.mean(ch0[1])) - mean0 > 0.05


def test_dataloader_batches():
    """Items stack into (B, T, C, H, W) via a plain DataLoader."""
    from torch.utils.data import DataLoader
    ds = SyntheticClipDataset(num_classes=4, length=8, T=2, H=16, W=16)
    loader = DataLoader(ds, batch_size=4)
    clips, labels = next(iter(loader))
    assert clips.shape == (4, 2, 3, 16, 16)
    assert labels.tolist() == [0, 1, 2, 3]
