"""State machine policies: strict vs optimistic tracking.

Measured motivation in reports/loop_eval.md: with ORACLE verbs, strict
advances on only ~1/3 of real EPIC action ticks (real cooking interleaves
and revisits); optimistic lifts the oracle to 0.75 advance / 6-of-10
completed. These tests pin the rule semantics.
"""

import pytest

from reference.state_machine_ref import Recipe, RecipeStateMachine, Step


def _machine(steps, policy="strict"):
    recipe = Recipe(name="t", steps=[Step(v, n) for v, n in steps],
                    ingredients=[n for _, n in steps],
                    verbs=sorted({v for v, _ in steps}))
    return RecipeStateMachine(recipe, policy=policy)


def test_strict_flags_later_step_wrong_order():
    m = _machine([("take", "tomato"), ("cut", "tomato"), ("put", "tomato")])
    ev = m.update("cut", "tomato", 0)          # later step, early
    assert ev.kind == "wrong_order"
    assert m.current_index == 0                # strict: no movement


def test_optimistic_jumps_to_later_step():
    m = _machine([("take", "tomato"), ("cut", "tomato"), ("put", "tomato")],
                 policy="optimistic")
    ev = m.update("cut", "tomato", 0)          # matches later step 1
    assert ev.kind == "advance"
    assert m.current_index == 2                # skipped to after 'cut'
    ev = m.update("put", "tomato", 0)
    assert ev.kind == "done"


def test_optimistic_final_step_completes_recipe():
    m = _machine([("take", "x"), ("put", "x")], policy="optimistic")
    ev = m.update("put", "x", 0)               # jumps to last step -> done
    assert ev.kind == "done"


def test_both_policies_agree_on_exact_repeat_and_unknown():
    for policy in ("strict", "optimistic"):
        m = _machine([("take", "a"), ("cut", "a")], policy=policy)
        assert m.update("take", "a", 0).kind == "advance"
        assert m.update("take", "a", 0).kind == "repeat"   # just-completed
        assert m.update("wash", "a", 0).kind == "idle"     # not in recipe
        assert m.current_index == 1


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        _machine([("take", "a")], policy="sloppy")
