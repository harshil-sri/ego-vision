"""OWNER EXERCISE STUB — copy to core/state_machine.py and fill in.

Reference: reference/state_machine_ref.py (Step/Recipe/Event dataclasses
live there; import or copy them). Worked example: docs/worked/state_machine.md
— it contains the full decision table to trace by hand.
Equivalence check: python3 -m pytest tests/equivalence --impl core

The four rules (trace them on paper before writing code):
  1. pair == current step   -> advance (or "done" if it was the last)
  2. pair == just-done step -> repeat (no state change)
  3. pair == some LATER step-> wrong_order (message names expected step)
  4. anything else          -> idle (no state change)
"""

from __future__ import annotations

# from reference.state_machine_ref import Recipe, Step, Event  # or copy them


class RecipeStateMachine:
    """Walks a Recipe from step 0 to done, one update() per event."""

    def __init__(self, recipe) -> None:
        """recipe: Recipe with a non-empty steps list (ValueError if empty)."""
        raise NotImplementedError("owner fills this in")

    def update(self, verb: str, ingredient: str, t_ns: int):
        """Apply one (verb, ingredient) event at time t_ns (ns monotonic).

        Output: Event(kind in {"advance","wrong_order","repeat","idle",
        "done"}, step_index int (-1 for pure idle), next_step Step|None,
        message str). Never raises on unknown verbs/ingredients."""
        raise NotImplementedError("owner fills this in")
