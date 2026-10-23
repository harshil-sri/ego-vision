"""Equivalence tests: replay the shared vectors against whichever
implementation the registry currently selects.

Run against the reference (default, must always pass):
    python3 -m pytest tests/equivalence -q
Run against the owner's core/ versions (once he has written them):
    python3 -m pytest tests/equivalence --impl core -q

Vector files live in tests/equivalence/vectors/ and are generated from the
REFERENCE implementations by scripts/gen_equivalence_vectors.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.registry as registry
from tests.equivalence.common import toy_embed

VECTORS = Path(__file__).parent / "vectors"


def active_impl() -> str:
    return registry.active_impls()["metrics"]


# ---------------------------------------------------------------- metrics

def test_metrics():
    v = np.load(VECTORS / "metrics.npz")
    mod = registry.get("metrics")
    scores, y = v["scores"], v["y"]
    pred = scores.argmax(1)
    assert mod.topk_accuracy(scores, y, 1) == pytest.approx(float(v["top1"]))
    assert mod.topk_accuracy(scores, y, 3) == pytest.approx(float(v["top3"]))
    assert mod.per_class_accuracy(pred, y, 8) == pytest.approx(v["per_class"])
    assert mod.mean_class_accuracy(pred, y, 8) == pytest.approx(float(v["mean_class"]))
    assert mod.confusion_matrix(pred, y, 8) == pytest.approx(v["confusion"])
    assert mod.majority_baseline(y) == pytest.approx(float(v["majority"]))


# ---------------------------------------------------------- frame sampler

def test_frame_sampler():
    v = np.load(VECTORS / "frame_sampler.npz", allow_pickle=False)
    mod = registry.get("frame_sampler")
    fn = getattr(mod, "sample_indices")
    for name in ["typical", "first_segment", "T_gt_len", "single_frame_clip",
                 "boundary", "exclusive"]:
        a, b, T, fps, inc = v[name + "_args"]
        got = fn(int(a), int(b), int(T), float(fps), stop_inclusive=bool(inc))
        assert isinstance(got, np.ndarray) and got.shape == (T,), (name, got)
        assert got.dtype == np.int64, (name, got.dtype)
        assert np.array_equal(got, v[name]), (name, got, v[name])
        assert bool((np.diff(got) >= 0).all()), f"{name}: not sorted"


# ------------------------------------------------------------ enrollment

def test_enrollment():
    v = np.load(VECTORS / "enrollment.npz", allow_pickle=True)
    cls = registry.get_class("enrollment")
    m = cls(toy_embed, threshold=float(v["threshold"]))
    m.enroll("red", [v["red0"], v["red3"]])
    m.enroll("green", [v["green0"], v["green1"]])
    m.enroll("blue", [v["blue0"], v["blue2"]])
    for q in ["q_red", "q_green", "q_grey"]:
        name, score = m.match(v[q])
        exp_name = str(v["match_" + q.split("_")[1] + "_name"])
        exp_score = float(v["match_" + q.split("_")[1] + "_score"])
        assert name == exp_name, (q, name, exp_name)
        assert score == pytest.approx(exp_score, abs=1e-5), (q, score)


# ----------------------------------------------------- temporal attention

def test_temporal_attention():
    v = np.load(VECTORS / "temporal_attention.npz")
    cls = registry.get_class("temporal_attention")
    att = cls(16)
    # Inject the vector's weights (documented attribute names; the stub
    # tells the owner which names the harness sets).
    with torch.no_grad():
        att.W.weight.copy_(torch.from_numpy(v["W_weight"]))
        att.W.bias.copy_(torch.from_numpy(v["W_bias"]))
        att.v.weight.copy_(torch.from_numpy(v["v_weight"]))
    pooled, weights = att(torch.from_numpy(v["x"]))
    assert pooled.shape == (4, 16) and weights.shape == (4, 8)
    assert np.allclose(pooled.detach().numpy(), v["pooled"], atol=1e-5)
    assert np.allclose(weights.detach().numpy(), v["weights"], atol=1e-5)
    assert torch.allclose(weights.sum(1), torch.ones(4), atol=1e-5)


# --------------------------------------------------------- state machine

def test_state_machine():
    payload = json.loads((VECTORS / "state_machine.json").read_text())
    from reference.state_machine_ref import Recipe, Step  # types only
    r = payload["recipe"]
    recipe = Recipe(r["name"], r["ingredients"], r["verbs"],
                    [Step(s["verb"], s["ingredient"], s["note"])
                     for s in r["steps"]])
    cls = registry.get_class("state_machine")
    sm = cls(recipe)
    for i, ev in enumerate(payload["events"]):
        out = sm.update(ev["verb"], ev["ingredient"], t_ns=i)
        assert out.kind == ev["kind"], (i, out.kind, ev["kind"])
        assert out.step_index == ev["step_index"], (i, out.step_index)


# ---------------------------------------------------------------- report

def test_active_impl_is_known():
    assert active_impl() in ("reference", "core")
