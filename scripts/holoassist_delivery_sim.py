"""Simulation of the app's message-hold rule on HoloAssist (sets its two numbers).

The app holds a routine message while the hands do fine-motor work
(src/pipeline/delivery.py: `fine_threshold`, `max_hold_s`). This script
replays that rule, with the real class, on the HoloAssist videos and counts
what it does. It does NOT show that the rule helps a user.

Hand state: fine-motor probability per second from the video head, cross-
fitted by recording session (the head never saw the session it scores),
smoothed like the app (running average, weight 0.5 on the newest tick) and,
for comparison, not smoothed (weight 1).

Message moments (three sets)
    step boundary   every start or end of a labelled coarse step (boundaries
                    closer than 1 s count once). This is where an assistant
                    like the app has a "next step" message.
    random moment   one moment in every 30 s of video, at a random second. A
                    message that does not depend on the task (a timer, a tip).
    human routine   every moment a human instructor started routine speech
                    (high-level instruction, follow-up, confirmation).

Scores for each (fine_threshold, max_hold_s)
    held            share of messages that wait
    wait            mean wait of a held message, seconds
    timeout         share of held messages released by max_hold_s
    replaced        share of held messages replaced by a newer message
    in fine work    share of delivered messages that arrive while the LABELLED
                    hand state is screw, unscrew or insert
    mean delay      wait averaged over all messages
"label rule" uses the labelled hand state instead of the video head (ceiling).

Usage:
    python -m scripts.holoassist_delivery_sim
Outputs: reports/holoassist_delivery_sim.json, reports/holoassist_delivery_sim.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.holoassist_from_video import fit_head, predict_head
from scripts.holoassist_need_controlled import FINE, state_at
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path
from src.pipeline.delivery import HandAwareDelivery

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MAX_HOLDS = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 1e9]
ROUTINE = ["high_level", "follow_up", "confirm"]


def replay(t: np.ndarray, fine: np.ndarray, msgs: np.ndarray, thr: float, max_hold: float) -> np.ndarray:
    """Run the delivery rule on one video.

    t (n,) tick times, fine (n,) fine-motor probability at each tick, msgs (m,)
    message times; all sorted. -> (m, 3): delivery time (nan if replaced),
    wait in seconds, reason (0 at once, 1 hands free, 2 held too long, 3 replaced).
    """
    d = HandAwareDelivery(thr, max_hold)
    out = np.full((len(msgs), 3), np.nan)
    out[:, 2] = 3
    ev = sorted([(x, 0, i) for i, x in enumerate(t)] + [(x, 1, i) for i, x in enumerate(msgs)])
    for x, kind, i in ev:
        if kind == 0:
            for m in d.tick(float(fine[i]), x):
                out[int(m["text"])] = (x, m["held_s"], 1 if m["reason"] == "hands free" else 2)
        else:
            for m in d.submit("advance", str(i), False, x):
                out[i] = (x, 0.0, 0)
    return out


def score(parts: list) -> dict:
    """parts: per video (replay output, labelled-fine at the message time, labelled-fine lookup)."""
    R = np.concatenate([p[0] for p in parts])
    held = R[:, 2] > 0
    deliv = R[:, 2] < 3
    fine_at = np.concatenate([p[2](p[0][:, 0]) for p in parts])          # nan delivery time -> False
    return {"messages": int(len(R)), "held": float(held.mean()),
            "wait": float(R[held & deliv, 1].mean()) if (held & deliv).any() else 0.0,
            "timeout": float((R[held, 2] == 2).mean()) if held.any() else 0.0,
            "replaced": float((R[held, 2] == 3).mean()) if held.any() else 0.0,
            "in_fine_work": float(fine_at[deliv].mean()),
            "mean_delay": float(np.where(deliv, R[:, 1], 0.0).sum() / max(deliv.sum(), 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--alphas", default="0.5,1", help="smoothing weights of the newest tick; the first is the app's")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold

    fd = Path(resolve_data_path("/mnt/ssd/holoassist")) / f"features_w{args.window}"
    meta, feats = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    S = len(top) + 2
    is_fine = np.array([v in FINE for v in top] + [False, False])

    vids = []                                              # (name, V, rows, t)
    for name, g in meta.groupby("video_name", sort=False):
        v = data.get(name)
        V = build_video(v, top) if v is not None else None
        if V is None:
            continue
        t = g["t_end"].to_numpy()
        keep = (t >= V["lo"]) & (t < V["hi"])
        if keep.sum() >= 4:
            vids.append((name, V, g.index.to_numpy()[keep], t[keep]))
    rows = np.concatenate([x[2] for x in vids])
    st = np.concatenate([state_at(x[1], x[3]) for x in vids])
    sess = np.concatenate([[x[0].split("-")[0]] * len(x[2]) for x in vids])
    F = feats[rows]
    p_fine = np.zeros(len(rows), dtype=np.float32)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, st, sess)):
        p_fine[te] = predict_head(fit_head(F[tr], st[tr], S, args.device), F[te], args.device)[:, is_fine].sum(1)
        print(f"  head fold {k + 1}/5", flush=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    rng = np.random.default_rng(0)
    per, o = [], 0                                         # per video: t, smoothed p, label fine 0/1, messages
    for name, V, r, t in vids:
        raw = p_fine[o:o + len(r)].astype(float); o += len(r)
        ps = {}
        for a in alphas:
            p = raw.copy()
            for i in range(1, len(p)):
                p[i] = a * p[i] + (1 - a) * p[i - 1]
            ps[f"p{a:g}"] = p
        lab = is_fine[state_at(V, t)].astype(float)

        def fine_at(x, V=V):
            ok = np.isfinite(x)
            out = np.zeros(len(x), dtype=bool)
            inside = ok & (x >= V["lo"]) & (x < V["hi"])
            out[inside] = is_fine[state_at(V, x[inside])]
            return out

        def inside(x, t=t):
            x = np.asarray(x, dtype=float)
            return np.sort(x[np.isfinite(x) & (x >= t[0]) & (x < t[-1])])

        c = inside(V["coarse"])
        c = c[np.r_[True, np.diff(c) >= 1.0]] if len(c) else c
        starts = np.arange(t[0], t[-1] - 30.0, 30.0)       # one random second in every 30 s
        rnd = inside(starts + rng.integers(0, 30, len(starts)) + 0.5)
        msgs = {"step boundary": c, "random moment": rnd,
                "human routine": inside(np.concatenate([V["onsets"][g] for g in ROUTINE]))}
        per.append({"session": name.split("-")[0], "t": t, "lab": lab, "fine_at": fine_at, "msgs": msgs, **ps})

    def run(kind, thr, hold, source, sel=None):
        parts = []
        for v in (per if sel is None else [per[i] for i in sel]):
            m = v["msgs"][kind]
            parts.append((replay(v["t"], v[source], m, thr, hold), None, v["fine_at"]))
        return score(parts)

    res = {"videos": len(per), "sessions": len({v["session"] for v in per}), "alphas": alphas,
           "fine_share_of_ticks": float(np.concatenate([v["lab"] for v in per]).mean()), "sets": {}}
    A0 = f"p{alphas[0]:g}"
    for kind in ("step boundary", "random moment", "human routine"):
        r = {"no rule": run(kind, 2.0, 0.0, A0), "label rule, no time limit": run(kind, 0.5, 1e9, "lab"),
             "label rule, 4 s": run(kind, 0.5, 4.0, "lab"), "label rule, 8 s": run(kind, 0.5, 8.0, "lab"), "grid": {}}
        for a in alphas:
            for thr in THRESHOLDS:
                for hold in MAX_HOLDS:
                    r["grid"][f"{a:g}|{thr:g}|{'none' if hold > 1e8 else f'{hold:g}'}"] = run(kind, thr, hold, f"p{a:g}")
        res["sets"][kind] = r
        print(kind, "done", flush=True)

    # how long does labelled fine-motor work continue after a step boundary that falls inside it?
    rem = []
    for v in per:
        for c in v["msgs"]["step boundary"]:
            i = np.searchsorted(v["t"], c)
            if i < len(v["t"]) and v["lab"][i]:
                j = i
                while j < len(v["t"]) and v["lab"][j]:
                    j += 1
                rem.append(v["t"][j - 1] - c + 1.0)
    res["remaining_fine_work_s"] = {f"p{q}": float(np.percentile(rem, q)) for q in (25, 50, 75, 90)}
    res["remaining_fine_work_n"] = len(rem)

    # intervals for the app defaults on random moments, resampling sessions
    D = HandAwareDelivery()
    by = {}
    for i, v in enumerate(per):
        by.setdefault(v["session"], []).append(i)
    keys = list(by)
    cache = {i: (replay(v["t"], v[A0], v["msgs"]["random moment"], D.fine_threshold, D.max_hold_s),
                 replay(v["t"], v[A0], v["msgs"]["random moment"], 2.0, 0.0)) for i, v in enumerate(per)}
    diffs = []
    for _ in range(args.boot):
        sel = [i for j in rng.integers(0, len(keys), len(keys)) for i in by[keys[j]]]
        a = score([(cache[i][0], None, per[i]["fine_at"]) for i in sel])["in_fine_work"]
        b = score([(cache[i][1], None, per[i]["fine_at"]) for i in sel])["in_fine_work"]
        diffs.append((a, b, a / b if b else np.nan))
    d = np.array(diffs)
    res["default_interval"] = {"rule": [float(np.percentile(d[:, 0], q)) for q in (2.5, 97.5)],
                               "no_rule": [float(np.percentile(d[:, 1], q)) for q in (2.5, 97.5)],
                               "ratio": [float(np.nanpercentile(d[:, 2], q)) for q in (2.5, 97.5)]}
    Path("reports/holoassist_delivery_sim.json").write_text(json.dumps(res, indent=2))

    L = ["# Simulation of the app's message-hold rule on HoloAssist", "",
         "Generated by `python -m scripts.holoassist_delivery_sim`. "
         f"{res['videos']} videos, {res['sessions']} sessions. The rule (`src/pipeline/delivery.py`) holds a routine "
         "message while the fine-motor probability is at or above a threshold, for at most a time limit. "
         "The probability comes from the video head, cross-fitted by session. Smoothing: weight of the newest "
         f"second in a running average ({alphas[0]:g} in the app, 1 = none). "
         f"Labelled fine-motor work (screw, unscrew, insert) covers {res['fine_share_of_ticks']:.1%} of the seconds. "
         "This counts what the rule does. It does not show that the rule helps a user.", ""]
    for kind, r in res["sets"].items():
        n = r["no rule"]
        L += [f"## Messages at: {kind} ({n['messages']} messages)", "",
              f"Without the rule, {n['in_fine_work']:.1%} of the messages arrive during labelled fine-motor work.", "",
              "| smoothing | threshold | time limit | held | mean wait of a held message | released by the limit | replaced | "
              "arrive in fine work | mean delay, all messages |", "|---|---|---|---|---|---|---|---|---|"]

        def row(a, b, s, sm="-"):
            return (f"| {sm} | {a} | {b} | {s['held']:.1%} | {s['wait']:.1f} s | {s['timeout']:.0%} | {s['replaced']:.0%} | "
                    f"{s['in_fine_work']:.1%} | {s['mean_delay']:.2f} s |")

        for key, s in r["grid"].items():
            sm, a, b = key.split("|")
            if b in ("4", "8", "16", "none"):              # the other limits are in the JSON
                L.append(row(a, b + (" s" if b != "none" else ""), s, sm))
        L += [row("labels", "none", r["label rule, no time limit"]), row("labels", "8 s", r["label rule, 8 s"]),
              row("labels", "4 s", r["label rule, 4 s"]), ""]
    q = res["remaining_fine_work_s"]
    di = res["default_interval"]
    L += ["## How long fine-motor work continues", "",
          f"For the {res['remaining_fine_work_n']} step boundaries that fall inside labelled fine-motor work, the work "
          f"continues for {q['p25']:.0f} s (25th percentile), {q['p50']:.0f} s (median), {q['p75']:.0f} s (75th) and "
          f"{q['p90']:.0f} s (90th) after the boundary (1 s resolution).", "",
          f"## Interval for the app defaults: threshold {D.fine_threshold:g}, limit {D.max_hold_s:g} s (random moments)", "",
          f"Share that arrives in fine work, 95% interval over {args.boot} resamples of sessions: with the rule "
          f"{di['rule'][0]:.1%} to {di['rule'][1]:.1%}, without {di['no_rule'][0]:.1%} to {di['no_rule'][1]:.1%}, "
          f"ratio {di['ratio'][0]:.2f} to {di['ratio'][1]:.2f}.", ""]
    Path("reports/holoassist_delivery_sim.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
