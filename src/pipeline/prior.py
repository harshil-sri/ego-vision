"""Recipe-aware verb prior: constrain recognition to plausible next steps.

Guided cooking is not open-world verb recognition: at any moment the recipe
constrains what can legally happen next. This module turns the state
machine's progress into a probability mask over verbs:

  legal verbs  = verbs of the REMAINING steps (current step first)
  illegal ones = verbs whose every step is already done (they can only
                 produce repeat/wrong_order noise), or verbs that never
                 appear in the recipe at all

Masking is soft: illegal verbs keep `leak` of their mass (renormalized), so
a genuinely performed-but-unexpected verb can still win when the model is
confident enough. With leak=0 the layer is a hard constraint.

Contract:
    apply_recipe_prior(probs (C,) float, verbs list[str], legal set[str],
                       leak float) -> (C,) float, sums to 1

All math is local and testable; the engine wires it in front of the smoother
(config knob `recipe_prior`, default off).
"""

from __future__ import annotations

import numpy as np


def legal_verbs(steps, current_index: int) -> set[str]:
    """Verbs of the remaining steps: current step and everything after it.

    Inputs: steps = list of step objects with .verb (recipe.steps or
    machine steps); current_index = the state machine's current step.
    Output: set of verb names. At recipe end (current_index >= len) the set
    is empty — callers should treat that as "no prior" (nothing left to do).
    """
    return {s.verb for s in steps[current_index:]} if current_index < len(steps) else set()


def apply_recipe_prior(probs: np.ndarray, verbs: list[str],
                       legal: set[str], leak: float = 0.05) -> np.ndarray:
    """Soft-mask `probs` to `legal` verbs.

    Inputs: probs (C,) float >= 0; verbs aligned with probs (the model's
    verb list, any order); legal = set of legal verb names; leak in [0, 1] —
    the fraction of mass illegal verbs keep (0 = hard constraint).
    Output: (C,) float64 summing to 1 (uniform input stays uniform when the
    legal set is empty or covers everything — degenerate-safe).
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return probs.copy()
    if probs.min() < 0 or not np.isfinite(probs).all():
        raise ValueError("probs must be finite and non-negative")
    leak = float(min(max(leak, 0.0), 1.0))
    if not legal or probs.sum() <= 0:
        return probs / max(probs.sum(), 1e-12)
    mask = np.array([1.0 if v in legal else leak for v in verbs])
    out = probs * mask
    return out / out.sum()
