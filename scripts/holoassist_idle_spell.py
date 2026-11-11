"""Pilot part 6c: inside a pending instruction, WHEN in an idle spell does it come?

Part 6 test A found that a pending next-step instruction starts mostly
while the performer's hands are idle. Two stories fit:

    waiting   the instructor has the instruction ready and waits for the
              hands to become free. Then it should come RIGHT AFTER the
              action ends: the chance per tick is highest in the first
              second of the idle spell.
    need      the performer stops because they do not know the next step.
              Idle hands are the request. Then the chance per tick should
              not peak at the start of the spell; it stays level or grows
              as the performer keeps waiting.

Data: the same episodes as test A (a coarse step boundary, then a
high-level instruction within 20 s; scripts.holoassist_need_controlled),
only ticks with idle hands. For each tick: how long the hands have been
idle, and which kind of action ended before the spell.

Outputs the raw chance per 0.5 s tick by idle time, and odds ratios from a
logistic model that also has the time since the step boundary (spline).
Reference: idle for 4 s or more. 95% intervals: videos resampled.

Usage:
    python -m scripts.holoassist_idle_spell
Outputs: reports/holoassist_idle_spell.json, reports/holoassist_idle_spell.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_need_controlled import TICK, group_ids
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

BINS = [0.0, 0.5, 1.0, 2.0, 4.0, np.inf]
BIN_NAMES = ["0 to 0.5 s", "0.5 to 1 s", "1 to 2 s", "2 to 4 s", "4 s or more"]
PREV = ["fine-motor action", "other action"]


def idle_rows(V: dict, gid: np.ndarray, max_s: float = 20.0) -> np.ndarray:
    """-> rows (time since boundary, idle-time bin, previous action is fine 0/1, onset in next tick)
    for the idle ticks of every pending-instruction episode of one video."""
    idle_id = len(gid) - 1
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
            continue
        t = np.arange(c, h, TICK)
        k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
        for n, (ti, ki) in enumerate(zip(t, k)):
            if V["state"][ki] != idle_id:
                continue
            k0 = ki
            while k0 > 0 and V["state"][k0 - 1] == idle_id:           # start of this idle spell
                k0 -= 1
            if k0 == 0:
                continue                                              # no action before the spell
            b = np.searchsorted(BINS, ti - V["edges"][k0], side="right") - 1
            rows.append((ti - c, b, float(gid[V["state"][k0 - 1]] == 1), float(n == len(t) - 1)))
    return np.array(rows).reshape(-1, 4)


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
    parts = []
    for v in data:
        V = build_video(v, top)
        if V is not None:
            r = idle_rows(V, gid)
            if len(r):
                parts.append(r)
    X = np.concatenate(parts)
    spl = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(X[:, :1])
    rng = np.random.default_rng(0)

    def coefs(ps):
        Z_ = np.concatenate(ps)
        onehot = np.stack([(Z_[:, 1] == b).astype(float) for b in range(4)], 1)       # reference: bin 4
        return LogisticRegression(C=10.0, max_iter=500).fit(np.c_[onehot, spl.transform(Z_[:, :1])], Z_[:, 3]).coef_[0][:4]

    def table(ps):
        Z_ = np.concatenate(ps)
        point = coefs(ps)
        d = np.array([coefs([ps[k] for k in rng.integers(0, len(ps), len(ps))]) for _ in range(args.boot)])
        out = {}
        for b, name in enumerate(BIN_NAMES):
            m = Z_[:, 1] == b
            row = {"ticks": int(m.sum()), "raw_chance_per_tick": round(float(Z_[m, 3].mean()), 4) if m.any() else None}
            if b < 4:
                row["odds_ratio_vs_4s"] = [round(float(np.exp(point[b])), 3),
                                           round(float(np.exp(np.percentile(d[:, b], 2.5))), 3),
                                           round(float(np.exp(np.percentile(d[:, b], 97.5))), 3)]
            out[name] = row
        return out

    res = {"videos": len(parts), "idle_ticks": int(len(X)), "instructions": int(X[:, 3].sum()), "boot": args.boot,
           "all": table(parts)}
    for f, name in ((1.0, PREV[0]), (0.0, PREV[1])):
        sub = [p[p[:, 2] == f] for p in parts]
        res[name] = table([p for p in sub if len(p)])
    Path("reports/holoassist_idle_spell.json").write_text(json.dumps(res, indent=2))

    L = ["# HoloAssist pilot part 6c: when in an idle spell does a pending instruction come?", "",
         "Generated by `python -m scripts.holoassist_idle_spell`. Labels only. "
         f"{res['videos']} videos, {res['idle_ticks']} idle ticks of {TICK} s inside pending-instruction episodes, "
         f"{res['instructions']} instructions that start in an idle tick. "
         "Odds ratio: against hands idle for 4 s or more, at equal time since the step boundary; "
         f"95% interval from {args.boot} refits on resampled videos. See the script docstring for the two stories.", ""]
    for key, title in (("all", "All idle spells"), (PREV[0], "Idle spells after a fine-motor action (screw, unscrew, insert)"),
                       (PREV[1], "Idle spells after any other action")):
        L += [f"## {title}", "", "| hands idle for | ticks | chance per tick that the instruction starts (raw) | odds ratio |",
              "|---|---|---|---|"]
        for name, r in res[key].items():
            orr = r.get("odds_ratio_vs_4s")
            L.append(f"| {name} | {r['ticks']} | {r['raw_chance_per_tick']} | "
                     + (f"{orr[0]:.2f} ({orr[1]:.2f} to {orr[2]:.2f})" if orr else "1 (reference)") + " |")
        L.append("")
    Path("reports/holoassist_idle_spell.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
