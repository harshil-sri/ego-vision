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

--video repeats both tests on the videos with VideoMAE features, with the
hand state that the model predicts (cross-fitted, from
scripts.holoassist_shift_control: fine / free / other, one per second).
Differences: ticks are 1 s; the reference group is "free" (idle, hold,
place, inspect together, because the model predicts them as one group);
in test B "fine" is the predicted state at the tick, and "ongoing" still
comes from the labels. The same model on the same ticks with the labelled
group is given next to it.

Usage:
    python -m scripts.holoassist_need_controlled
    python -m scripts.holoassist_shift_control ; python -m scripts.holoassist_need_controlled --video
Outputs: reports/holoassist_need_controlled.{json,md}, reports/holoassist_need_controlled_video.{json,md}
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


def main_video(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer

    z = np.load(Path(resolve_data_path("/mnt/ssd/holoassist")) / "features_w2.0" / "oof_groups.npz", allow_pickle=True)
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    A, B = [], []            # per video: A rows (dt, predicted, labelled, y); B rows (dt, ongoing, predicted, labelled, y)
    for name in np.unique(z["video"]):
        m = z["video"] == name
        t, pred, lab = z["t_end"][m], z["predicted"][m], z["labelled"][m]     # groups: 0 fine, 1 free, 2 other
        V = build_video(data[name], top)
        ra, rb = [], []
        hl = np.sort(V["onsets"]["high_level"])
        coarse = V["coarse"][np.isfinite(V["coarse"])]
        for i, c in enumerate(coarse):
            j = np.searchsorted(hl, c, side="right")
            if j >= len(hl):
                continue
            h = hl[j]
            nxt = coarse[i + 1] if i + 1 < len(coarse) else np.inf
            k = np.flatnonzero((t >= c) & (t < h))
            if h - c > 20.0 or h > nxt or len(k) == 0:
                continue
            y = np.zeros(len(k)); y[-1] = 1
            ra.append(np.c_[t[k] - c, pred[k], lab[k], y])
        for s_, e, _ in V["wrong"]:
            j = np.searchsorted(V["corr"], s_, side="left")
            if j >= len(V["corr"]) or V["corr"][j] > e + 10.0:
                continue
            k = np.flatnonzero((t >= s_) & (t < V["corr"][j]))
            if len(k) == 0:
                continue
            y = np.zeros(len(k)); y[-1] = 1
            rb.append(np.c_[t[k] - s_, t[k] < e, pred[k], lab[k], y])
        A.append(np.concatenate(ra) if ra else np.zeros((0, 4)))
        B.append(np.concatenate(rb) if rb else np.zeros((0, 5)))
    A, B = [a for a in A if len(a)], [b for b in B if len(b)]
    rng = np.random.default_rng(0)
    XA, XB = np.concatenate(A), np.concatenate(B)
    splA = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(XA[:, :1])
    splB = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(XB[:, :1])

    def coefs(parts, col, spl, extra):
        X = np.concatenate(parts)
        Z = np.c_[(X[:, col] == 0).astype(float), (X[:, col] == 2).astype(float),
                  X[:, extra] if extra is not None else np.zeros((len(X), 0)), spl.transform(X[:, :1])]
        return LogisticRegression(C=10.0, max_iter=500).fit(Z, X[:, -1]).coef_[0][:2]      # fine, other vs free

    def ci(parts, col, spl, extra=None):
        point = coefs(parts, col, spl, extra)
        d = np.array([coefs([parts[k] for k in rng.integers(0, len(parts), len(parts))], col, spl, extra)
                      for _ in range(args.boot)])
        return {n: [round(float(np.exp(point[i])), 3), round(float(np.exp(np.percentile(d[:, i], 2.5))), 3),
                    round(float(np.exp(np.percentile(d[:, i], 97.5))), 3)] for i, n in enumerate(["fine", "other"])}

    res = {"videos": int(len(np.unique(z["video"]))), "boot": args.boot,
           "A": {"episodes": int(XA[:, -1].sum()), "ticks": int(len(XA)),
                 "predicted": ci(A, 1, splA), "labelled": ci(A, 2, splA)},
           "B": {"episodes": int(XB[:, -1].sum()), "ticks": int(len(XB)),
                 "predicted": ci(B, 2, splB, [1]), "labelled": ci(B, 3, splB, [1])}}
    Path("reports/holoassist_need_controlled_video.json").write_text(json.dumps(res, indent=2))

    def cell(v):
        return f"{v[0]:.2f} ({v[1]:.2f} to {v[2]:.2f})"
    L = ["# HoloAssist pilot part 6 from video: hand state at fixed need", "",
         "Generated by `python -m scripts.holoassist_need_controlled --video`. "
         f"{res['videos']} videos with VideoMAE features, ticks of 1 s. Hand state group (fine / free / other) predicted "
         "from video, cross-fitted by recording session; the labelled group on the same ticks for comparison. "
         f"Odds ratios against the free group, 95% intervals from {args.boot} refits on resampled videos.", "",
         f"## Test A: a next-step instruction is pending ({res['A']['episodes']} episodes, {res['A']['ticks']} ticks)", "",
         "| hand state from | fine against free | other against free |", "|---|---|---|"]
    for k in ("predicted", "labelled"):
        L.append(f"| {k} | {cell(res['A'][k]['fine'])} | {cell(res['A'][k]['other'])} |")
    L += ["", f"## Test B: a mistake is in progress and will be corrected ({res['B']['episodes']} mistakes, {res['B']['ticks']} ticks)", "",
          "Odds of the correction starting in the next second, at equal time since the mistake started and "
          "with 'wrong action still going on' in the model.", "",
          "| hand state from | fine against free | other against free |", "|---|---|---|"]
    for k in ("predicted", "labelled"):
        L.append(f"| {k} | {cell(res['B'][k]['fine'])} | {cell(res['B'][k]['other'])} |")
    Path("reports/holoassist_need_controlled_video.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--video", action="store_true", help="use video-predicted hand states (see docstring)")
    args = ap.parse_args()
    if args.video:
        return main_video(args)
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
