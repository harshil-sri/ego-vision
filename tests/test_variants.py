"""Tests for src/models/variants.py: all three variants, registry wiring,
the predict_probs eval-mode contract, and the BatchNorm train/eval trap."""

from __future__ import annotations

import pytest
import torch

from src import registry
from src.models.variants import VARIANTS, VerbModel


def _clip_batch(B: int = 2, T: int = 4, S: int = 64, seed: int = 0) -> torch.Tensor:
    """Deterministic random clip batch.

    Output: torch.float32 (B, T, 3, S, S) in a seeded generator.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 3, S, S, generator=g)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes(variant):
    """Every variant: (2,4,3,64,64) float32 -> (2,8) logits."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", num_verbs=8, variant=variant,
                      pretrained=False)
    clips = _clip_batch(2, 4, 64)
    logits = model(clips)
    assert logits.shape == (2, 8)
    assert logits.dtype == torch.float32


def test_invalid_variant_raises():
    """Unknown variant -> ValueError listing the valid names."""
    with pytest.raises(ValueError, match="single_frame"):
        VerbModel("resnet18", 8, "slowfast", pretrained=False)


def test_temporal_attn_uses_registry_class():
    """temporal_attn variant builds its attn module via the registry (the
    owner's core/ TemporalAttention must drop in unchanged)."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "temporal_attn", pretrained=False)
    attn_cls = registry.get_class("temporal_attention")
    assert type(model.attn) is attn_cls


@pytest.mark.parametrize("variant", VARIANTS)
def test_backward_reaches_backbone(variant):
    """Loss.backward() leaves a grad on the backbone's first conv.

    single_frame intentionally trains through the MIDDLE frame only — the
    backbone still receives gradients through that single-frame path, which
    is what this asserts (conv1 is applied to the middle frame)."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, variant, pretrained=False)
    clips = _clip_batch(2, 4, 64)
    model(clips).sum().backward()          # scalar loss
    assert model.backbone.conv1.weight.grad is not None
    assert model.backbone.conv1.weight.grad.abs().sum() > 0


def test_single_frame_uses_middle_frame_only():
    """single_frame output depends ONLY on frame index T//2.

    Eval mode + no_grad so BatchNorm running stats never update between
    calls (an update would make the 'identical logits' assertions flaky).
    """
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "single_frame", pretrained=False)
    model.eval()
    T = 4
    middle = T // 2  # == 2 for T=4
    with torch.no_grad():
        base = _clip_batch(1, T, 64, seed=1)

        changed_elsewhere = base.clone()
        # overwrite every NON-middle frame (indices 0, 1, 3 for T=4):
        for t in range(T):
            if t != middle:
                changed_elsewhere[:, t] = torch.randn(3, 64, 64)
        # non-middle changes must NOT move the logits
        assert torch.allclose(model(base), model(changed_elsewhere))

        changed_middle = base.clone()
        changed_middle[:, middle] = torch.randn(3, 64, 64)
        # a middle-frame change MUST move the logits
        assert not torch.allclose(model(base), model(changed_middle))


def test_predict_probs_is_softmax_and_sums_to_one():
    """predict_probs: (T,C,H,W) -> (num_verbs,) float32 summing to 1."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "tsn_avg", pretrained=False)
    clip = _clip_batch(1, 4, 64, seed=2).squeeze(0)  # (T,C,H,W)
    probs = model.predict_probs(clip)
    assert probs.shape == (8,)
    assert probs.dtype == torch.float32
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_predict_probs_leaves_model_in_eval_mode():
    """Hard rule: the inference helper must leave the model in eval().

    Even after an explicit model.train(), predict_probs flips to eval and
    STAYS there — this is the documented BatchNorm batch-size-1 trap fix
    (a train-mode BN at batch 1 classifies with the single sample's own
    statistics, silently wrecking predictions)."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "tsn_avg", pretrained=False)
    model.train()
    assert model.training
    _ = model.predict_probs(_clip_batch(1, 4, 64, seed=3).squeeze(0))
    assert not model.training, "predict_probs must leave the model in eval()"
    # and the backbone itself, not just the wrapper:
    assert not model.backbone.training


def test_batch1_eval_differs_from_train_when_bn_stats_differ():
    """BatchNorm trap (hard rule 7): with running stats != batch stats, a
    batch-1 forward in train mode differs from the same forward in eval.

    single_frame is NOT exempt: its middle frame still passes through the
    backbone's BN layers, so the same trap applies and .eval() is required
    for its inference path too (predict_probs does this)."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "single_frame", pretrained=False)

    clip = _clip_batch(1, 4, 64, seed=4)  # (1, T, C, H, W): batch of ONE

    model.train()
    out_train = model(clip)      # uses the single clip's own batch statistics
    model.eval()
    out_eval = model(clip)       # uses running stats (still ~init: mean0/var1)
    # Freshly initialised running stats differ from this clip's batch stats,
    # so the two logits must NOT match. If they did, BN eval-mode wiring
    # would be broken (or BN was silently removed).
    assert not torch.allclose(out_train, out_eval), (
        "train- and eval-mode batch-1 forwards agree; BatchNorm running "
        "statistics are not being used in eval mode")


def test_eval_forward_deterministic_batch1():
    """In eval mode, repeated batch-1 forwards are identical (no batch-stat
    leakage between calls)."""
    torch.manual_seed(0)
    model = VerbModel("resnet18", 8, "temporal_attn", pretrained=False)
    model.eval()
    clip = _clip_batch(1, 4, 64, seed=5)
    with torch.no_grad():
        a = model(clip)
        b = model(clip)
    assert torch.allclose(a, b)
