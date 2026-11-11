"""Ego-EXTRA: is expert speech timed around the hand state that a VIDEO model sees?

scripts.egoextra_hands found no relation between unasked expert speech and
raw hand motion (wrist speed, finger motion). Motion is not the same as
"busy with a fine-motor action". This script uses the measure that worked
on HoloAssist (pilot part 5): a classifier on frozen VideoMAE vectors.

Two models are trained on HoloAssist only (305 videos, human labels) and
applied to Ego-EXTRA unchanged. Ego-EXTRA has no labels of this kind, so
the models cannot be checked there; other glasses, other tasks.

  hand-state model   22 states (20 verbs, other verb, idle). The states are
                     grouped by what HoloAssist part 2b found:
                         fine   screw, unscrew, insert   (instructors avoid these most)
                         free   idle, hold, place, inspect (close to idle)
                         other  the rest
                     A tick's group = the group with the highest summed
                     probability.
  onset model        probability that a HoloAssist instructor starts
                     routine speech in the next second, from the video
                     vector alone (32 principal components). Ticks of each
                     Ego-EXTRA video are cut into thirds by this score:
                     low, mid, high.

Features of each dataset are standardised with that dataset's own mean and
standard deviation (a crude correction for the different cameras).

Check without labels: the glasses track the hands themselves
(scripts.egoextra_hands). If the hand-state model means anything on this
video, it should say "free" more often when no hand is tracked, and "fine"
more often when the fingers move but the wrist does not.

Test (as in scripts.egoextra_hands): share of speech onsets in each
group, observed / expected by chance; chance = circular shifts inside each
video. The tick used for an onset is the last one that ends at or before
the onset, so the model sees nothing after the speech starts.

Usage:
    python -m scripts.egoextra_from_video
Outputs: reports/egoextra_from_video.json, reports/egoextra_from_video.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.egoextra_hands import DT, hand_states
from scripts.egoextra_speech import expert_turns, load_transcript
from scripts.holoassist_from_video import fit_head, predict_head
from scripts.holoassist_predict import tick_features
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

HOLO, EGO = "/mnt/ssd/holoassist", "/mnt/ssd/ego-extra"
FINE, FREE = ["screw", "unscrew", "insert"], ["idle", "hold", "place", "inspect"]
GROUPS = ["fine", "free", "other"]
THIRDS = ["low", "mid", "high"]


def holoassist_ticks(window: float, with_types: bool = False):
    """-> features (n, 768), hand state id (n,), onset in next second (n,), state names, video name (n,).
    with_types: also a dict {utterance type: (n,) bool}, true where an instructor utterance of that
    type (follow_up, high_level, confirm, correct) starts within one tick after the tick."""
    fd = Path(resolve_data_path(HOLO)) / f"features_w{window}"
    meta, feats = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    tick = float(json.loads((fd / "info.json").read_text())["tick_s"])
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    rows, ST, Y = [], [], []
    types = {}
    for name, g in meta.groupby("video_name", sort=False):
        v = data.get(name)
        V = build_video(v, top) if v is not None else None
        if V is None:
            continue
        t = g["t_end"].to_numpy()
        keep = (t >= V["lo"]) & (t < V["hi"] - tick)
        if keep.sum() < 4:
            continue
        f = tick_features(V, v, t[keep], horizon=tick)
        rows.append(g.index.to_numpy()[keep]); ST.append(f[1][:, 0].astype(int)); Y.append(f[2])
        for gname, on in V["onsets"].items():
            on = np.sort(np.asarray(on, dtype=float))
            n_in = np.searchsorted(on, t[keep] + tick, side="right") - np.searchsorted(on, t[keep], side="right")
            types.setdefault(gname, []).append((name, n_in > 0))
    rows = np.concatenate(rows)
    out = (feats[rows], np.concatenate(ST), np.concatenate(Y), top + ["other_verb", "idle"],
           meta["video_name"].to_numpy()[rows])
    if with_types:
        vids = out[4]
        flags = {}
        for gname, parts in types.items():                 # a type can be absent in some videos
            fl = np.zeros(len(vids), dtype=bool)
            for name, m in parts:
                fl[vids == name] = m
            flags[gname] = fl
        return out + (flags,)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--turn-gap", type=float, default=2.0)
    ap.add_argument("--ask-window", type=float, default=5.0)
    ap.add_argument("--long", type=float, default=1.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)

    # ---- train on HoloAssist -------------------------------------------
    Fh, ST, Y, names, _ = holoassist_ticks(args.window)
    Fh = (Fh - Fh.mean(0)) / (Fh.std(0) + 1e-6)
    head = fit_head(Fh, ST, len(names), args.device)
    pca = PCA(32, random_state=0).fit(Fh)
    onset = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(pca.transform(Fh), Y)
    gid = np.array([0 if n in FINE else 1 if n in FREE else 2 for n in names])
    print(f"HoloAssist: {len(Y)} ticks, {int(Y.sum())} onsets", flush=True)

    # ---- apply to Ego-EXTRA --------------------------------------------
    root = Path(resolve_data_path(EGO))
    fd = root / f"features_w{args.window}"
    meta, Fe = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    Fe = (Fe - Fe.mean(0)) / (Fe.std(0) + 1e-6)
    P = predict_head(head, Fe, args.device)                            # (n, 22)
    G = np.stack([P[:, gid == k].sum(1) for k in range(3)], 1)         # (n, 3) group probabilities
    score = onset.predict_proba(pca.transform(Fe))[:, 1]
    proto = {v["ID"]: v["Protocol"] for v in json.loads((root / "protocol_split_wacv.json").read_text())["videos"]}
    tr_files = {p.name.replace("_transcript.txt", ""): p for p in (root / "transcript").rglob("*_transcript.txt")}

    videos = []
    for name, g in meta.groupby("video_name", sort=False):
        vid = next((k for k in tr_files if k in name), None)           # the file name contains the video id
        if vid is None or vid not in proto:
            continue
        u = load_transcript(tr_files[vid])
        idx, t = g.index.to_numpy(), g["t_end"].to_numpy()
        if len(u) < 5 or len(t) < 60 or abs(t[-1] - u[:, 1].max()) > 60:
            continue
        group = G[idx].argmax(1)
        lo, hi = np.percentile(score[idx], [100 / 3, 200 / 3])
        third = (score[idx] > lo).astype(int) + (score[idx] > hi)
        starts, unasked, speech = expert_turns(u, args.turn_gap, args.ask_window)
        long_ = speech >= args.long
        a = u[u[:, 2] == 0]
        tr = a[np.r_[True, a[1:, 0] - np.maximum.accumulate(a[:-1, 1]) >= args.turn_gap], 0]
        ev = {"expert, unasked, long": starts[unasked & long_],
              "expert, unasked, short": starts[unasked & ~long_],
              "expert, after trainee speech, long": starts[~unasked & long_],
              "trainee turn": tr}
        # last tick that ends at or before the onset; onsets before the first tick are dropped
        ev = {k: (np.searchsorted(t, x, side="right") - 1) for k, x in ev.items()}
        ev = {k: x[x >= 0] for k, x in ev.items()}
        # glasses' own hand tracking at each tick (0 none, 1 still, 2 mid, 3 moving), -1 if unusable
        track = {}
        for sig, pat in (("wrist", f"{vid}_*.csv"), ("fingers", f"{vid}_hand_tracking_results_*.csv")):
            track[sig] = np.full(len(t), -1)
            for hp in sorted((root / "ht" / "hand_tracking").glob(pat)):
                n_rows = sum(1 for _ in open(hp)) - 1
                if n_rows < 100:
                    continue
                hs = hand_states(hp, sig)
                if hs is not None and abs(len(hs[0]) * DT - u[:, 1].max()) <= 30:
                    track[sig] = hs[0][np.minimum((t / DT).astype(int), len(hs[0]) - 1)]
                    break
        videos.append({"vid": vid, "protocol": proto[vid], "group": group, "third": third, "ev": ev,
                       "track": track})
    print(f"Ego-EXTRA: {len(videos)} videos used of {meta['video_name'].nunique()}", flush=True)

    def ratios(vs, key, field, k_states):
        obs, null = np.zeros(k_states), np.zeros((args.shifts, k_states))
        for v in vs:
            e, s = v["ev"][key], v[field]
            if len(e) == 0:
                continue
            obs += np.bincount(s[e], minlength=k_states)
            st = s[(e[None, :] + rng.integers(0, len(s), size=args.shifts)[:, None]) % len(s)]
            for k in range(k_states):
                null[:, k] += (st == k).sum(1)
        exp = np.maximum(null.mean(0), 1e-9)
        return obs, obs / exp, np.percentile(null, 2.5, 0) / exp, np.percentile(null, 97.5, 0) / exp

    keys = list(videos[0]["ev"])
    sets = {"all videos": videos,
            "protocol 1 (proactive)": [v for v in videos if v["protocol"] == 1],
            "protocol 2 (only when asked or necessary)": [v for v in videos if v["protocol"] == 2]}
    for sc in sorted({v["vid"].split("_")[0] for v in videos}):       # scenario = first part of the video id
        sets[f"scenario {sc}"] = [v for v in videos if v["vid"].startswith(sc + "_")]
    sets = {k: v for k, v in sets.items() if v}
    allg = np.concatenate([v["group"] for v in videos])
    res = {"videos": len(videos), "hours": round(len(allg) / 3600, 1), "shifts": args.shifts,
           "holoassist_ticks": int(len(Y)),
           "group_share_of_time": {g: round(float((allg == i).mean()), 3) for i, g in enumerate(GROUPS)},
           "top_predicted_states": {names[j]: round(float(c / len(P)), 3)
                                    for j, c in Counter(P.argmax(1)).most_common(8)},
           "sets": {}}
    # label-free check: predicted group against the glasses' hand tracking
    check = {}
    for sig in ("wrist", "fingers"):
        tr_ = np.concatenate([v["track"][sig] for v in videos])
        ok = tr_ >= 0
        check[sig] = {"ticks": int(ok.sum()),
                      "share_predicted": {st: {g: round(float((allg[ok & (tr_ == k)] == i).mean()), 3)
                                               for i, g in enumerate(GROUPS)}
                                          for k, st in enumerate(["none", "still", "mid", "moving"])
                                          if (ok & (tr_ == k)).sum() > 0}}
    res["tracking_check"] = check
    for sname, vs in sets.items():
        res["sets"][sname] = {"videos": len(vs)}
        for k in keys:
            cell = {}
            for field, labels in (("group", GROUPS), ("third", THIRDS)):
                obs, r, lo, hi = ratios(vs, k, field, 3)
                cell["n"] = int(obs.sum())
                cell[field] = {l: [round(float(r[i]), 3), round(float(lo[i]), 3), round(float(hi[i]), 3)]
                               for i, l in enumerate(labels)}
            res["sets"][sname][k] = cell
    Path("reports/egoextra_from_video.json").write_text(json.dumps(res, indent=2))

    L = ["# Ego-EXTRA: speech onsets against hand state predicted from video", "",
         "Generated by `python -m scripts.egoextra_from_video`. Models trained on HoloAssist "
         f"({res['holoassist_ticks']} ticks) and applied unchanged to {res['videos']} Ego-EXTRA videos ({res['hours']} h). "
         "The models were not checked on Ego-EXTRA (no labels). See the script docstring.", "",
         "Predicted group, share of time: " + ", ".join(f"{g} {v:.0%}" for g, v in res["group_share_of_time"].items())
         + ". Most frequent predicted states: " + ", ".join(f"{k} {v:.0%}" for k, v in res["top_predicted_states"].items()) + ".", "",
         f"Each cell: observed / expected by chance [range of 95% of {args.shifts} random shifts].", ""]
    L += ["## Check without labels: predicted group against the glasses' hand tracking", "",
          "Share of ticks with each predicted group, by tracked hand state at the same tick.", "",
          "| tracking signal | tracked state | predicted fine | predicted free | predicted other |", "|---|---|---|---|---|"]
    for sig, c in res["tracking_check"].items():
        for st, sh in c["share_predicted"].items():
            L.append(f"| {sig} | {st} | " + " | ".join(f"{sh[g]:.2f}" for g in GROUPS) + " |")
    L.append("")
    for sname, r in res["sets"].items():
        L += [f"## {sname} ({r['videos']} videos)", "",
              "| onsets | n | hands: fine | hands: free | hands: other | onset score: low | mid | high |",
              "|---|---|---|---|---|---|---|---|"]
        for k in keys:
            c = r[k]
            cells = [f"{c['group'][g][0]:.2f} [{c['group'][g][1]:.2f}, {c['group'][g][2]:.2f}]" for g in GROUPS] + \
                    [f"{c['third'][g][0]:.2f} [{c['third'][g][1]:.2f}, {c['third'][g][2]:.2f}]" for g in THIRDS]
            L.append(f"| {k} | {c['n']} | " + " | ".join(cells) + " |")
        L.append("")
    Path("reports/egoextra_from_video.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
