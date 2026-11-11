"""Pilot part 6: does hand state matter when the NEED to speak is held fixed?

The problem with parts 1 to 5: instructors speak less during fine-motor
actions (screw, unscrew, insert), but mistakes are also rare there. "The
hands are busy" and "nothing needs saying" arrive together. This script
looks only at moments where something DOES need saying, in two ways.
Labels only, all HoloAssist videos.

Test A: a pending next-step instruction.
    Episode: starts at a boundary of a coarse action (a step of the task
    ends or begins) and lasts until the instructor gives the next
    high-level instruction, the next coarse boundary, or 20 s.
    Unit: 0.5 s ticks inside episodes that DO end with an instruction
    within 20 s (so an instruction was pending all the time).
    Question: at equal time since the boundary, is the instruction less
    likely to start while the hands are in a fine-motor action than while
    they are idle?

Test B: a mistake in progress that the instructor will correct.
    Episode: from the start of a wrong action (label "corrected by
    instructor verbally") to the correction, at most 10 s after the action
    ends.
    Question: at equal time since the mistake started, is the correction
    less likely to start WHILE the wrong action is still going on than
    after it ended, and is that gap larger for fine-motor actions?
    The number to read is the interaction "fine x ongoing": below 1 means
    instructors hold a correction back until a fine-motor action is over,
    more than they do for other actions.

Model: logistic regression per tick (a discrete-time hazard model), time
since the episode start as a spline. 95% intervals: refit on videos
resampled with replacement.

What this cannot remove: the instructor may need more time to SEE a
mistake inside a fine-motor action, and a performer who starts fine work
after a step boundary may already know the next step.

Usage:
    python -m scripts.holoassist_need_controlled
Outputs: reports/holoassist_need_controlled.json, reports/holoassist_need_controlled.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

FINE, FREEVERB = {"screw", "unscrew", "insert"}, {"hold", "place", "inspect"}
TICK = 0.5
GROUPS = ["idle", "fine", "hold/place/inspect", "other action"]      # idle is the reference


def group_ids(top: list[str]) -> np.ndarray:
    """state id (20 verbs, other, idle) -> group id in GROUPS."""
    g = [1 if v in FINE else 2 if v in FREEVERB else 3 for v in top] + [3, 0]
    return np.array(g)


def state_at(V: dict, t: np.ndarray) -> np.ndarray:
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
    return V["state"][k]


def episodes_a(V: dict, gid: np.ndarray, max_s: float = 20.0):
    """-> rows (time since boundary, group id, onset in next tick) for test A."""
    hl = np.sort(V["onsets"]["high_level"])
    coarse = V["coarse"][np.isfinite(V["coarse"])]
    rows = []
    for i, c in enumerate(coarse):
        j = np.searchsorted(hl, c, side="right")
        if j >= len(hl):
            continue
        h = hl[j]
        nxt = coarse[i + 1] if i + 1 < len(coarse) else np.inf
        if h - c > max_s or h > nxt or not (V["lo"] <= c < V["hi"]):
            continue                                     # no instruction pending inside this episode
        t = np.arange(c, h, TICK)                        # last tick: the one with h in (t, t + TICK]
        y = np.zeros(len(t)); y[-1] = 1
        rows.append(np.c_[t - c, gid[state_at(V, t)], y])
    return np.concatenate(rows) if rows else np.zeros((0, 3))


def episodes_b(V: dict, max_after: float = 10.0):
    """-> rows (time since mistake start, ongoing 0/1, fine 0/1, correction in next tick) for test B."""
    corr = V["corr"]
    rows = []
    for s, e, verb in V["wrong"]:
        j = np.searchsorted(corr, s, side="left")
        if j >= len(corr) or corr[j] > e + max_after:
            continue
        t = np.arange(s, corr[j], TICK)
        if len(t) == 0:
            continue
        y = np.zeros(len(t)); y[-1] = 1
        rows.append(np.c_[t - s, t < e, np.full(len(t), float(verb in FINE)), y])
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    gid = group_ids(top)
    A, B = [], []                                        # one array per video
    for v in data:
        V = build_video(v, top)
        if V is None:
            continue
        A.append(episodes_a(V, gid)); B.append(episodes_b(V))
    rng = np.random.default_rng(0)

    def fit(parts, design):
        """Fit on the given per-video arrays; return the coefficients of interest."""
        X = np.concatenate(parts)
        Z, y = design(X)
        m = LogisticRegression(C=10.0, max_iter=500).fit(Z, y)
        return m.coef_[0]

    def with_ci(parts, design, idx):
        parts = [p for p in parts if len(p)]
        point = fit(parts, design)[idx]
        draws = np.array([fit([parts[k] for k in rng.integers(0, len(parts), len(parts))], design)[idx]
                          for _ in range(args.boot)])
        return [[round(float(np.exp(point[i])), 3), round(float(np.exp(np.percentile(draws[:, i], 2.5))), 3),
                 round(float(np.exp(np.percentile(draws[:, i], 97.5))), 3)] for i in range(len(idx))]

    # ---- test A ---------------------------------------------------------
    XA = np.concatenate(A)
    splA = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(XA[:, :1])

    def design_a(X):
        onehot = np.stack([(X[:, 1] == k).astype(float) for k in (1, 2, 3)], 1)     # fine, hold.., other
        return np.c_[onehot, splA.transform(X[:, :1])], X[:, 2]

    ra = with_ci(A, design_a, [0, 1, 2])
    share_a = {GROUPS[k]: round(float((XA[:, 1] == k).mean()), 3) for k in range(4)}
    raw_a = {GROUPS[k]: round(float(XA[XA[:, 1] == k, 2].mean()), 4) for k in range(4)}

    # ---- test B ---------------------------------------------------------
    XB = np.concatenate(B)
    splB = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(XB[:, :1])

    def design_b(X):
        ongoing, fine = X[:, 1], X[:, 2]
        return np.c_[ongoing, fine, ongoing * fine, splB.transform(X[:, :1])], X[:, 3]

    rb = with_ci(B, design_b, [0, 1, 2])
    eps_b = int(XB[:, 3].sum())
    before_end = {name: round(float(XB[(XB[:, 2] == f) & (XB[:, 3] == 1), 1].mean()), 3)
                  for name, f in (("fine", 1), ("not fine", 0))}
    n_b = {name: int(((XB[:, 2] == f) & (XB[:, 3] == 1)).sum()) for name, f in (("fine", 1), ("not fine", 0))}

    res = {"videos": len(A), "boot": args.boot,
           "A": {"episodes": int(XA[:, 2].sum()), "ticks": int(len(XA)), "share_of_ticks": share_a,
                 "raw_onset_rate_per_tick": raw_a,
                 "odds_ratio_vs_idle": dict(zip(GROUPS[1:], ra))},
           "B": {"episodes": eps_b, "ticks": int(len(XB)), "episodes_by_group": n_b,
                 "share_of_corrections_before_action_end": before_end,
                 "odds_ratio": dict(zip(["ongoing (not fine)", "fine (after the action)", "fine x ongoing"], rb))}}
    Path("reports/holoassist_need_controlled.json").write_text(json.dumps(res, indent=2))

    def cell(v):
        return f"{v[0]:.2f} ({v[1]:.2f} to {v[2]:.2f})"
    L = ["# HoloAssist pilot part 6: hand state at fixed need", "",
         "Generated by `python -m scripts.holoassist_need_controlled`. Labels only, "
         f"{res['videos']} videos. Odds ratios with 95% intervals from {args.boot} refits on resampled videos. "
         "See the script docstring for the design and its limits.", "",
         "## Test A: a next-step instruction is pending", "",
         f"{res['A']['episodes']} episodes (coarse boundary, then a high-level instruction within 20 s), "
         f"{res['A']['ticks']} ticks of {TICK} s.", "",
         "| hand state | share of ticks | onset rate per tick (raw) | odds of the instruction starting, against idle hands, at equal time since the boundary |",
         "|---|---|---|---|",
         f"| idle | {share_a['idle']:.0%} | {raw_a['idle']:.3f} | 1 (reference) |"]
    for g in GROUPS[1:]:
        L.append(f"| {g} | {share_a[g]:.0%} | {raw_a[g]:.3f} | {cell(res['A']['odds_ratio_vs_idle'][g])} |")
    L += ["", "## Test B: a mistake is in progress and will be corrected", "",
          f"{eps_b} corrected mistakes ({n_b['fine']} in fine-motor actions, {n_b['not fine']} in other actions). "
          f"Corrections that start before the wrong action ends: {before_end['fine']:.0%} for fine-motor actions, "
          f"{before_end['not fine']:.0%} for other actions (raw; fine-motor actions are longer).", "",
          "| term | odds ratio for the correction starting in the next tick, at equal time since the mistake started |",
          "|---|---|"]
    for k, v in res["B"]["odds_ratio"].items():
        L.append(f"| {k} | {cell(v)} |")
    Path("reports/holoassist_need_controlled.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
