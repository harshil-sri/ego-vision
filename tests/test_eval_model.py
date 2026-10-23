"""Tests for src/train/eval_model.py with HAND-COMPUTED expected numbers.

Case (5 samples, 3 classes), all values verifiable on paper:

    scores (5,3)          y    argmax(pred)
    [3, 1, 0]             0    0   correct
    [2, 2, 1]             1    0   wrong
    [0, 4, 1]             2    1   wrong
    [1, 0, 5]             2    2   correct
    [2, 3, 0]             1    1   correct

    top1            = 3/5            = 0.6
    per_class       = [1/1, 1/2, 1/2] = [1.0, 0.5, 0.5]
    mean_class      = (1.0+0.5+0.5)/3 = 0.666667 (2/3)
    majority        = 2/5            = 0.4   (classes 1 and 2 tie at 2)
    confusion rows=true, cols=pred:
        [[1,0,0], [1,1,0], [0,1,1]]
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data

from src.train.eval_model import evaluate, save_confusion_png

SCORES = torch.tensor([[3.0, 1.0, 0.0],
                       [2.0, 2.0, 1.0],
                       [0.0, 4.0, 1.0],
                       [1.0, 0.0, 5.0],
                       [2.0, 3.0, 0.0]])           # (5, 3) float32
Y = torch.tensor([0, 1, 2, 2, 1])                  # (5,) int64


class FixedScoresModel(nn.Module):
    """Model ignoring its input and returning fixed SCORES rows.

    forward input: x (B, 1) float tensor of ROW INDICES.
    forward output: (B, 3) float32 logits = SCORES[rows].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.long().squeeze(-1)      # (B,) int64 row ids
        return SCORES[rows]              # (B, 3) float32


def _loader(batch_size: int) -> torch.utils.data.DataLoader:
    """Loader over (row-index (1,) float, label) pairs.

    Output: DataLoader yielding ((B,1) float32, (B,) int64).
    """
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # (5, 1) float32
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, Y),
                                       batch_size=batch_size)


def test_evaluate_exact_numbers():
    """evaluate() reproduces every hand-computed value above."""
    model = FixedScoresModel()
    out = evaluate(model, _loader(batch_size=2), "cpu", num_verbs=3)

    assert out["top1"] == pytest.approx(0.6)
    assert out["top5"] == pytest.approx(0.6)          # duplicated: C=3 <= 5
    assert "note" in out and "num_verbs" in out["note"]
    assert out["per_class"] == pytest.approx([1.0, 0.5, 0.5])
    assert out["mean_class"] == pytest.approx(2.0 / 3.0)
    assert out["majority_baseline"] == pytest.approx(0.4)
    assert out["confusion"] == [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
    assert out["n_samples"] == 5


def test_evaluate_batching_invariant():
    """Batch size 1 vs 5 gives identical metrics (concatenation is sound)."""
    m1 = evaluate(FixedScoresModel(), _loader(batch_size=1), "cpu", 3)
    m5 = evaluate(FixedScoresModel(), _loader(batch_size=5), "cpu", 3)
    for key in ("top1", "top5", "mean_class", "majority_baseline",
                "per_class", "confusion"):
        assert m1[key] == m5[key]


def test_evaluate_top5_distinct_when_many_classes():
    """With C > 5, top5 is a real top-5 (here: truth in ranks 1-5 for all
    but the sample whose true class is 6th) — checked on a fixed case."""
    scores = torch.eye(7)[torch.arange(7)].float()   # (7,7): row i -> class i
    # shift sample 0's mass so its true class falls OUTSIDE its top-5
    scores[0] = torch.tensor([0.0, 9, 8, 7, 6, 5, 0.0])  # class 0 rank 6

    class ByIndex(nn.Module):
        def forward(self, x):                       # x (B,1) row indices
            return scores[x.long().squeeze(-1)]     # (B, 7)

    x = torch.arange(7, dtype=torch.float32).unsqueeze(1)  # (7,1)
    y = torch.arange(7)                             # (7,) truth = class id
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=7)
    out = evaluate(ByIndex(), loader, "cpu", num_verbs=7)
    assert "note" not in out                        # real top-5, no dupe
    # top1: only sample 0 wrong (its argmax is class 1) -> 6/7
    assert out["top1"] == pytest.approx(6 / 7)
    # top5: sample 0's true class 0 sits at rank 6 -> still wrong -> 6/7
    assert out["top5"] == pytest.approx(6 / 7)
    # per-class: everyone correct except class 0 -> 0/1
    assert out["per_class"][0] == pytest.approx(0.0)
    assert all(a == pytest.approx(1.0) for a in out["per_class"][1:])


def test_evaluate_leaves_model_in_eval_mode():
    """Hard rule 7: even after .train(), evaluate() ends in eval mode."""
    model = FixedScoresModel()
    model.train()
    evaluate(model, _loader(batch_size=5), "cpu", 3)
    assert not model.training


def test_evaluate_empty_loader_is_well_defined():
    """Empty loader returns zeros instead of crashing (missing-data case)."""
    empty = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)),
        batch_size=4)
    out = evaluate(FixedScoresModel(), empty, "cpu", 3)
    assert out["n_samples"] == 0
    assert out["top1"] == 0.0 and out["mean_class"] == 0.0
    assert out["confusion"] == [[0] * 3] * 3


def test_save_confusion_png_writes_file(tmp_path: Path):
    """PNG is created, non-empty, and starts with the PNG magic bytes."""
    cm = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]           # (3,3) ints
    out = tmp_path / "sub" / "confusion.png"         # nested dir gets created
    save_confusion_png(cm, ["close", "cut", "mix"], out)
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
