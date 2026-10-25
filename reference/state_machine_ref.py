"""Reference recipe state machine: consumes (verb, ingredient, t_ns) events
and walks a hard-coded recipe, flagging wrong order.

Study material (docs/worked/state_machine.md). Plain control flow, no
frameworks, no classes beyond the data containers — deliberately.

Contract (brief section 5.1):
    RecipeStateMachine(recipe: Recipe)
    .update(verb: str, ingredient: str, t_ns: int) -> Event

    Event(kind, step_index, next_step, message)
    kind in {"advance", "wrong_order", "repeat", "idle", "done"}

Semantics (the table the owner must be able to defend):
  Let cur = the step we are waiting for, prev = the step just completed.
  1. Event (verb, ingredient) equals cur's (verb, ingredient):
       advance. cur becomes cur+1; if that was the LAST step -> "done".
  2. Event equals prev's (verb, ingredient) exactly:
       "repeat" — already done, no state change (calm message).
  3. Event equals some FUTURE step's (verb, ingredient) (skipping ahead):
       "wrong_order" — message names the step we expected instead.
  4. Anything else (wrong ingredient, unknown, unrelated verb):
       "idle" — no state change; message restates what we wait for.
  Notes:
  - "unknown" verb or ingredient never advances (treated by rule 4).
  - After "done", every event is "idle" (recipe finished message).
  - Steps may repeat the same (verb, ingredient) pair elsewhere in the
    recipe (e.g. take tomato ... take cucumber): matching compares pairs,
    and rule 3 only triggers for a LATER occurrence, never an earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """One recipe step: a (verb, ingredient) pair plus optional note.

    verb/ingredient: str, must appear in the recipe's verbs/ingredients lists
    (validated by the loader in src/pipeline/recipe.py, not here).
    """
    verb: str
    ingredient: str
    note: str = ""


@dataclass
class Recipe:
    """A named recipe: ingredient list, allowed verbs, ordered steps."""
    name: str
    ingredients: list[str]
    verbs: list[str]
    steps: list[Step]


@dataclass
class Event:
    """Result of one update() call.

    kind: str as above; step_index: int index the event referred to (the
    step advanced to / expected / repeated; -1 for pure idle);
    next_step: Step or None (the step to show the user as "upcoming");
    message: human-readable, template-only text (no LLM anywhere).
    """
    kind: str
    step_index: int
    next_step: Step | None
    message: str


@dataclass
class _State:
    """Mutable machine state. Kept explicit so update() stays a pure
    function of (state, event) — easy to test and to trace by hand."""
    current: int = 0                    # index of the step we wait for
    finished: bool = False
    history: list[tuple] = field(default_factory=list)  # (t_ns, verb, ing, kind)


class RecipeStateMachine:
    """Walks a Recipe from step 0 to done, one update() per event.

    policy:
      'strict'    — Rule 3 fires wrong_order for any later-step attempt
                    (linear recipes, human corrects via the UI warning).
      'optimistic'— a later-step attempt ADVANCES the pointer to that step
                    (interleaved steps are assumed done off-screen). Measured
                    motivation: real EPIC action revisits/interleaves so much
                    that even ORACLE perception advances on only ~1/3 of
                    true ticks under strict (reports/loop_eval.md).
    """

    def __init__(self, recipe: Recipe, policy: str = "strict") -> None:
        if not recipe.steps:
            raise ValueError(f"recipe '{recipe.name}' has no steps")
        if policy not in ("strict", "optimistic"):
            raise ValueError(f"unknown policy '{policy}'")
        self.recipe = recipe
        self.policy = policy
        self._state = _State()

    # -- helpers -----------------------------------------------------------

    def _pair(self, i: int) -> tuple[str, str]:
        """(verb, ingredient) of step i."""
        step = self.recipe.steps[i]
        return step.verb, step.ingredient

    def _wait_message(self) -> str:
        """What the UI should tell the user we are waiting for."""
        if self._state.finished:
            return f"Recipe '{self.recipe.name}' finished."
        step = self.recipe.steps[self._state.current]
        note = f" ({step.note})" if step.note else ""
        return f"Next: {step.verb} {step.ingredient}{note}"

    def _future_index(self, verb: str, ingredient: str) -> int | None:
        """First LATER step equal to this pair, or None."""
        pair = (verb, ingredient)
        for i in range(self._state.current, len(self.recipe.steps)):
            if self._pair(i) == pair:
                return i
        return None

    # -- main --------------------------------------------------------------

    @property
    def current_index(self) -> int:
        """Index of the step currently waited on (last index when done)."""
        return min(self._state.current, len(self.recipe.steps) - 1)

    @property
    def history(self) -> list[tuple]:
        """All events so far: (t_ns, verb, ingredient, kind)."""
        return list(self._state.history)

    def update(self, verb: str, ingredient: str, t_ns: int) -> Event:
        """Apply one (verb, ingredient) event at time t_ns; return an Event."""
        ev: Event

        if self._state.finished:
            ev = Event("idle", -1, None, self._wait_message())
        elif (verb, ingredient) == self._pair(self._state.current):
            # Rule 1: correct step done.
            done_step = self.recipe.steps[self._state.current]
            self._state.current += 1
            if self._state.current >= len(self.recipe.steps):
                self._state.finished = True
                ev = Event("done", len(self.recipe.steps) - 1, None,
                           f"Recipe '{self.recipe.name}' complete.")
            else:
                nxt = self.recipe.steps[self._state.current]
                ev = Event("advance", self._state.current - 1, nxt,
                           f"Done: {done_step.verb} {done_step.ingredient}. "
                           + self._wait_message())
        elif (self._state.current > 0
              and (verb, ingredient) == self._pair(self._state.current - 1)):
            # Rule 2: exactly the step we just completed again.
            prev = self.recipe.steps[self._state.current - 1]
            ev = Event("repeat", self._state.current - 1, None,
                       f"Already done: {prev.verb} {prev.ingredient}. "
                       + self._wait_message())
        elif (future := self._future_index(verb, ingredient)) is not None:
            if self.policy == "optimistic":
                # Rule 3a (optimistic): jump to the matched later step,
                # treating the steps in between as done off-screen.
                self._state.current = future + 1
                if self._state.current >= len(self.recipe.steps):
                    self._state.finished = True
                    ev = Event("done", future, None,
                               f"Recipe '{self.recipe.name}' complete.")
                else:
                    nxt = self.recipe.steps[self._state.current]
                    ev = Event("advance", future, nxt,
                               f"Jumped ahead to: {verb} {ingredient}. "
                               + self._wait_message())
            else:
                # Rule 3 (strict): a later step attempted early.
                want = self.recipe.steps[self._state.current]
                ev = Event("wrong_order", self._state.current, want,
                           f"Out of order: expected '{want.verb} "
                           f"{want.ingredient}' before '{verb} {ingredient}'. "
                           + self._wait_message())
        else:
            # Rule 4: anything else — no state change.
            ev = Event("idle", -1, None, self._wait_message())

        self._state.history.append((t_ns, verb, ingredient, ev.kind))
        return ev
