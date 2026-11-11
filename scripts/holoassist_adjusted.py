"""Pilot part 2b: hand-state effect on instructor speech, adjusted for need.

Part 1 showed that routine instructor speech is rare while the performer
screws or inserts. One explanation needs no interruptibility at all:
people who are screwing do not NEED help at that moment. This script asks
how much of the effect is left after "need" is accounted for, as far as
the labels allow.

Unit: one 0.5 s tick of a task (same ticks as scripts.holoassist_predict).
Outcome: a routine instructor utterance (follow-up, high-level instruction,
confirmation) starts within the next 0.5 s.

Model: logistic regression
    logit P(onset) = f(need proxies) + b[hand state]
    need proxies   seconds since the instructor last spoke, seconds since
                   the last coarse step boundary, seconds since task start,
                   fraction of the task elapsed. Each enters as a smooth
                   curve (cubic splines), not a straight line.
    hand state     one indicator per state; "idle" is the reference.
exp(b) is the ADJUSTED ODDS RATIO of that state against idle hands: how
many times more or less likely an onset is, at equal values of the need
proxies. The unadjusted odds ratio (hand state only) is printed beside it.

A third model adds each video's own overall onset rate. Sessions differ a
lot in how much the instructor talks, and quiet sessions tend to be the
ones with long screwing. Without this term the hand-state effect is
overstated.

Uncertainty: videos are resampled with replacement (cluster bootstrap) and
the model is refitted; the interval is the central 95% of the refits.

Limit: the need proxies are crude. "Adjusted" means adjusted for these
four numbers, not for everything an instructor knows.

Usage:
    python -m scripts.holoassist_adjusted --boot 40
Outputs: reports/holoassist_adjusted.json, reports/holoassist_adjusted.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_predict import tick_features
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    names = top + ["other_verb", "idle"]
    IDLE = len(names) - 1

    XT, ST, Y, G = [], [], [], []
    for i, v in enumerate(data):
        V = build_video(v, top)
        if V is None:
            continue
        f = tick_features(V, v)
        if f is None:
            continue
        XT.append(f[0]); ST.append(f[1][:, 0].astype(int)); Y.append(f[2]); G.append(np.full(len(f[2]), i))
    XT, ST, Y, G = map(np.concatenate, (XT, ST, Y, G))
    print(f"ticks {len(Y)}  onsets {int(Y.sum())}  videos {len(set(G))}")

    spl = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(XT)
    need = StandardScaler().fit_transform(spl.transform(XT))                  # (N, 4*7)
    onehot = np.zeros((len(Y), len(names) - 1))                               # idle = reference (all zeros)
    busy = ST != IDLE
    onehot[np.flatnonzero(busy), ST[busy]] = 1.0
    # Talkativeness of each video: log of its own onset rate. Without it the
    # model compares across videos (tasks with long screwing have quiet
    # instructors); with it the comparison is within a video, like the
    # circular-shift test of part 1.
    rate = {v: (Y[G == v].sum() + 0.5) / ((G == v).sum() + 1.0) for v in np.unique(G)}
    talk = np.log(np.array([rate[v] for v in G]))[:, None]
    talk = (talk - talk.mean()) / talk.std()
    X_adj = np.concatenate([need, onehot], axis=1)
    X_within = np.concatenate([need, talk, onehot], axis=1)
    k0, k1 = need.shape[1], need.shape[1] + 1

    def fit(X, y, cols):
        m = LogisticRegression(C=10.0, max_iter=300).fit(X, y)
        return m.coef_[0][cols]

    full_adj = fit(X_adj, Y, slice(k0, None))
    full_within = fit(X_within, Y, slice(k1, None))
    full_raw = fit(onehot, Y, slice(None))

    rng = np.random.default_rng(args.seed)
    vids = np.unique(G)
    rows_of = {v: np.flatnonzero(G == v) for v in vids}
    boots = []
    for b in range(args.boot):
        pick = rng.choice(vids, len(vids), replace=True)
        idx = np.concatenate([rows_of[v] for v in pick])
        boots.append(fit(X_within[idx], Y[idx], slice(k1, None)))
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{args.boot}", flush=True)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    seconds = np.bincount(ST, minlength=len(names)) * 0.5
    out = {"ticks": int(len(Y)), "onsets": int(Y.sum()), "videos": int(len(vids)), "bootstrap": args.boot,
           "reference": "idle", "states": {}}
    for j, n in enumerate(names[:-1]):
        out["states"][n] = {"hours": round(float(seconds[j] / 3600), 2),
                            "odds_ratio_unadjusted": round(float(np.exp(full_raw[j])), 3),
                            "odds_ratio_adjusted": round(float(np.exp(full_adj[j])), 3),
                            "odds_ratio_within_video": round(float(np.exp(full_within[j])), 3),
                            "ci95": [round(float(np.exp(lo[j])), 3), round(float(np.exp(hi[j])), 3)]}
    Path("reports/holoassist_adjusted.json").write_text(json.dumps(out, indent=2))
    order = sorted(out["states"], key=lambda n: out["states"][n]["odds_ratio_within_video"])
    L = ["# HoloAssist pilot part 2b: hand-state effect adjusted for need proxies", "",
         "Generated by `python -m scripts.holoassist_adjusted`. "
         f"{out['ticks']} ticks of 0.5 s from {out['videos']} videos, {out['onsets']} routine instructor onsets. "
         "Odds ratio = odds of a routine onset in the next 0.5 s in this hand state, divided by the odds with "
         "idle hands. Adjusted = at equal values of four need proxies (time since the instructor last spoke, "
         "time since the last coarse step boundary, time in task, fraction of task elapsed; spline terms). "
         "Within video = the same, plus each video's own overall onset rate as a covariate, so quiet and "
         "talkative sessions are not compared with each other. This is the column to trust. "
         f"Interval (for the within-video column) = central 95% of {args.boot} refits on videos resampled "
         "with replacement.", "",
         "| hand state | hours observed | odds ratio, unadjusted | adjusted for need | "
         "adjusted for need, within video | 95% interval |",
         "|---|---|---|---|---|---|"]
    for n in order:
        r = out["states"][n]
        L.append(f"| {n} | {r['hours']} | {r['odds_ratio_unadjusted']} | {r['odds_ratio_adjusted']} | "
                 f"{r['odds_ratio_within_video']} | {r['ci95'][0]} to {r['ci95'][1]} |")
    Path("reports/holoassist_adjusted.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
