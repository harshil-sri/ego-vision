"""Recipe-aware verb prior: math contract + engine wiring.

apply_recipe_prior is pure and deterministic — every property below is a
hand-checkable invariant, not a tolerance test.
"""

import numpy as np
import pytest

from src.pipeline.engine import COOKING, CookingEngine, EngineConfig
from src.pipeline.events import EventBus
from src.pipeline.prior import apply_recipe_prior, legal_verbs
from src.pipeline.window import SyntheticFrameSource

RECIPE = "recipes/example_salad.yaml"


class _Steps:
    def __init__(self, verbs):
        self.verb = verbs


# ---------------------------------------------------------------- math

def test_legal_verbs_remaining_steps_only():
    steps = [_Steps(v) for v in ("take", "wash", "cut", "put", "pour")]
    assert legal_verbs(steps, 0) == {"take", "wash", "cut", "put", "pour"}
    assert legal_verbs(steps, 2) == {"cut", "put", "pour"}
    assert legal_verbs(steps, 4) == {"pour"}
    assert legal_verbs(steps, 5) == set()      # recipe done


def test_prior_masks_illegal_and_renormalizes():
    verbs = ["close", "cut", "mix", "take", "wash"]
    probs = np.array([0.10, 0.20, 0.10, 0.30, 0.30])
    out = apply_recipe_prior(probs, verbs, {"take", "wash"}, leak=0.0)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == 0.0 and out[2] == 0.0          # hard-masked
    # take:wash ratio preserved: 0.30:0.30 -> equal
    assert out[3] == pytest.approx(out[4])


def test_prior_leak_keeps_small_mass_for_illegal():
    verbs = ["cut", "take"]
    probs = np.array([0.9, 0.1])
    out = apply_recipe_prior(probs, verbs, {"take"}, leak=0.10)
    assert out[1] > out[0]          # confident illegal verb still wins
    assert out.sum() == pytest.approx(1.0)
    # strong model confidence can beat the mask: 0.9*0.1 vs 0.1*1 -> 0.09 vs 0.10
    assert out[0] == pytest.approx(0.09 / 0.19)


def test_prior_degenerate_inputs_safe():
    for bad in (np.zeros(3), np.array([])):
        out = apply_recipe_prior(bad, ["a", "b", "c"][:len(bad)], {"a"})
        assert np.isfinite(out).all() and out.sum() <= 1.0 + 1e-9
    out = apply_recipe_prior(np.array([0.5, 0.5]), ["a", "b"], set())
    assert out.tolist() == [0.5, 0.5]               # no legal set -> untouched
    with pytest.raises(ValueError):
        apply_recipe_prior(np.array([-0.1, 1.1]), ["a", "b"], {"a"})


# ---------------------------------------------------------- engine wiring

class _FakeVerbSource:
    """Emits a fixed probability vector; mimics ModelVerbSource.verbs."""

    name = "fake"
    verbs = ["close", "cut", "mix", "open", "pour", "put", "take", "wash"]

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def predict(self, window):
        return self.probs.copy()


def _engine(prior_on: bool, probs) -> CookingEngine:
    cfg = EngineConfig(mode="clip", embedder="toy", matcher_threshold=0.3,
                       scan_seconds=0.4, scan_fps=6.0, tick_hz=10.0,
                       matcher_hz=4.0, metrics_every_s=0.5,
                       recipe_prior=prior_on, min_conf=0.5)
    eng = CookingEngine(SyntheticFrameSource(fps=20), cfg, RECIPE,
                        verb_source=_FakeVerbSource(probs), bus=EventBus())
    return eng


def test_prior_flips_prediction_to_legal_verb():
    # model (weakly) prefers 'close' — a verb that never appears in the
    # salad recipe. Without the prior it wins; with it, 'take' (step 1)
    # must win. Ids: close=0 cut=1 mix=2 open=3 pour=4 put=5 take=6 wash=7
    probs = [0.52, 0.05, 0.03, 0.05, 0.03, 0.04, 0.25, 0.03]
    fired = []
    for use_prior in (False, True):
        eng = _engine(use_prior, probs)
        eng._ingredient = ("tomato", 0.9)
        eng.phase = COOKING
        # drive one recognition tick directly (no thread needed)
        from types import SimpleNamespace
        eng._recognize_verb([SimpleNamespace(image=np.zeros((8, 8, 3), np.uint8))] * 8)
        fired.append((use_prior, eng._last_verb[0]))
    without, with_prior = fired
    assert without == (False, "close")
    assert with_prior == (True, "take")   # prior rescued the first step
