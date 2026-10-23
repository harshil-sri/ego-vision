# Worked example: recipe state machine (`reference/state_machine_ref.py`)

A 4-step recipe and 8 events, traced to completion. This is the same
trace the equivalence vector replays (tests/equivalence/vectors/
state_machine.json), so the code's answers are tested, not narrated.

## The recipe

```
steps: 0 take a   1 cut a   2 take b   3 put b        verbs: take, cut, put
                                                       ingredients: a, b
```

## The decision table (derive every case from these four rules)

```
pair = (verb, ingredient);  cur = step we wait for;  prev = step just done
1. pair == cur          -> advance (done if cur was last)
2. pair == prev         -> repeat  (no state change)
3. pair in LATER steps  -> wrong_order (message names cur)
4. else                 -> idle    (no state change)
```

## The trace, step by step (current step in [brackets])

| # | event | cur before | rule | kind | state after |
|---|---|---|---|---|---|
| 1 | take b | [0 take a] | 3 (pair == step 2, later) | **wrong_order** | unchanged |
| 2 | take a | [0 take a] | 1 | **advance** | cur=1 |
| 3 | take a | [1 cut a] | 2 (prev == pair) | **repeat** | unchanged |
| 4 | open a | [1 cut a] | 4 | **idle** | unchanged |
| 5 | cut a | [1 cut a] | 1 | **advance** | cur=2 |
| 6 | take b | [2 take b] | 1 | **advance** | cur=3 |
| 7 | put b | [3 put b] | 1 (last step) | **done** | finished |
| 8 | put b | finished | — (post-done rule) | **idle** | finished |

Read the interesting ones:
- Event 1: "take b" IS in the recipe (step 2) but comes early →
  wrong_order, not idle — the demo's flag-worthy moment. The message
  names the expected pair: "expected 'take a'".
- Event 3: repeating the JUST-completed step is a "repeat", not an error
  — recognizers double-fire constantly; punishing them would spam.
- Event 4: "open" isn't in this recipe at all → idle, state intact.
- Event 8: after done, everything is idle (the recipe is over; a RESTART
  is a new machine, not an update).

Why wrong_order only fires for LATER steps: repeating an EARLIER
completed pair (say "take a" at cur=2 — pair == step 0, earlier) is rule
4 → idle. Already-done-early steps shouldn't nag; the checklist already
shows them checked.

## What each row and column means

There is no matrix here — the shape to internalize is the STATE:
`current` (int index into steps) + `finished` (bool) + `history`
(list of (t_ns, verb, ingredient, kind) tuples — the session log the app
writes to runs/sessions/*.jsonl). Event.step_index: the step the event
REFERRED to (advanced-from / expected / repeated); −1 for pure idle;
next_step: what the UI should highlight next (None on done/idle).

## Three common mistakes

1. **Matching on verb only** (ignoring ingredient): "cut cucumber"
   satisfies "cut tomato" — the whole point of the (verb, ingredient)
   pairing dies. Always compare PAIRS.
2. **advance off-by-one**: advancing THEN checking done must use the
   NEW index vs len(steps); checking the OLD index reports done one step
   early or late.
3. **wrong_order for ANY mismatch** (rule 4 folded into rule 3): every
   unrecognized action spams warnings — the demo becomes unusable.
   wrong_order is reserved for "a real recipe step, attempted early".

## Edge cases

- Empty recipe: ValueError at construction (a machine with no steps
  cannot even state "next:").
- Same pair twice in the recipe ("put a" at steps 3 and 7): rule 3 finds
  the FIRST later occurrence; once cur passes step 3, "put a" only
  matches at step 7.
- unknown verb/ingredient from the matcher: rule 4 by construction —
  ("unknown", x) never equals a step pair. The scan-phase threshold and
  the verb confidence threshold (D3a) thus compose safely.
- t_ns unused in the decision logic: it only timestamps history. Idle
  detection over TIME (no event for N seconds → nudge) is the app's job,
  not the machine's — see src/pipeline for the idle timer.
