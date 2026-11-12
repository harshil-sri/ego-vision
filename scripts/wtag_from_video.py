"""WTaG: is instructor speech timed around the hand state that a VIDEO model sees?

WTaG (Bao et al., EMNLP Findings 2023, arXiv 2311.00738) is a third
dataset: 56 recordings (9.9 h) of a user who makes pour-over coffee, a mug
cake or pinwheels with a HoloLens on the head, and a human instructor who
watches the camera and talks. Every instructor utterance has a start time,
an intent, and for instructions a type. The types give the split that
HoloAssist finding 3 needs, as a human label:

    routine      Instruction: current step, next step, detailed
    correction   Instruction: wrong action, wrong state, wrong object, wrong tool

WTaG has no hand-action labels. As on Ego-EXTRA
(scripts.egoextra_from_video), the hand state comes from the head trained
on HoloAssist video and applied unchanged; it cannot be checked here.
Hand state groups: fine (screw, unscrew, insert), free (idle, hold, place,
inspect), other. Because kitchen recipes may have little fine-motor work,
each video's ticks are also cut into thirds by the fine-motor probability
and by the HoloAssist onset score (see scripts.egoextra_from_video).

Test: share of speech onsets in each group, observed / expected by chance;
chance = circular shifts inside each video. The tick used for an onset is
the last one that ends at or before the onset.

Timing: annotation times are 100 ns ticks on one clock;
Video/VideoMpegTiming.txt gives the clock time of the first and last
frame. Video time = frame / average frame rate. A recording whose video
length differs from the timing file by more than 5 s is dropped.

Before: python -m scripts.holoassist_video_features --root /mnt/ssd/wtag
        (videos/<recording>.mp4 are hard links to extract/<recording>/Video/Video.mpeg)
Usage:  python -m scripts.wtag_from_video
Outputs: reports/wtag_from_video.json, reports/wtag_from_video.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.egoextra_from_video import FINE, FREE, GROUPS, HOLO, THIRDS, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head
from src.paths import resolve_data_path

WTAG = "/mnt/ssd/wtag"
TICKS_PER_S = 1e7
ROUTINE = {"current step", "next step", "detailed", "details"}
RECIPES = {"Boil water": "coffee", "Assemble filter cone": "coffee", "Place cake liner": "mug cake",
           "Dry ingredients": "mug cake", "Combine": "mug cake", "Place tortilla": "pinwheels",
           "Spread nut butter": "pinwheels"}


def read_rows(path: Path) -> list[list[str]]:
    return [l.rstrip("\r\n").split("\t") for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]


def load_recording(d: Path):
    """-> dict: t0 (clock ticks), duration s, recipe, events {name: onset times in s}, user (n, 2) start/end s."""
    t0, t1 = (int(x) for x in (d / "Video" / "VideoMpegTiming.txt").read_text().split()[:2])
    sec = lambda x: (int(x) - t0) / TICKS_PER_S
    steps = read_rows(d / "StepDetection" / "StepDetection.txt")
    recipe = next((RECIPES[s[2].strip()] for s in steps if s[2].strip() in RECIPES), "unknown")
    user = np.array([(sec(r[0]), sec(r[1])) for r in read_rows(d / "TextASR" / "UserAnnotations_intent.txt")
                     if len(r) >= 3], dtype=float).reshape(-1, 2)
    uq = [sec(r[0]) for r in read_rows(d / "TextASR" / "UserAnnotations_intent.txt")
          if len(r) >= 4 and r[3].strip().lower().startswith("question")]
    ev = {"routine instruction": [], "correction": [], "answer": [], "confirmation": []}
    for r in read_rows(d / "TextASR" / "InstructorAnnotations_intent.txt"):
        if len(r) < 4:
            continue
        intent, sub = r[3].strip().lower(), (r[4].strip().lower() if len(r) > 4 else "")
        if intent.startswith("instruction"):
            if sub.startswith("wrong"):
                ev["correction"].append(sec(r[0]))
            elif sub in ROUTINE or intent.endswith("current step"):
                ev["routine instruction"].append(sec(r[0]))
        elif intent in ev:
            ev[intent].append(sec(r[0]))
    ev = {k: np.array(sorted(v), dtype=float) for k, v in ev.items()}
    ev["user, any utterance"] = np.sort(user[:, 0]) if len(user) else np.array([])
    ev["user question"] = np.array(sorted(uq), dtype=float)
    return {"duration": (t1 - t0) / TICKS_PER_S, "recipe": recipe, "ev": ev, "user": user}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--ask-window", type=float, default=5.0)
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

    # ---- apply to WTaG ---------------------------------------------------
    root = Path(resolve_data_path(WTAG))
    fd = root / f"features_w{args.window}"
    meta, Fw = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    Fw = (Fw - Fw.mean(0)) / (Fw.std(0) + 1e-6)
    P = predict_head(head, Fw, args.device)
    G = np.stack([P[:, gid == k].sum(1) for k in range(3)], 1)
    score = onset.predict_proba(pca.transform(Fw))[:, 1]

    def thirds(x):
        lo, hi = np.percentile(x, [100 / 3, 200 / 3])
        return (x > lo).astype(int) + (x > hi)

    videos, dropped = [], []
    for name, g in meta.groupby("video_name", sort=False):
        rec = load_recording(root / "extract" / name)
        idx, t = g.index.to_numpy(), g["t_end"].to_numpy()
        if len(t) < 60 or abs(t[-1] - rec["duration"]) > 5:
            dropped.append(f"{name} (video {t[-1]:.0f} s, timing file {rec['duration']:.0f} s)")
            continue
        ev = dict(rec["ev"])
        r, u = ev["routine instruction"], rec["user"]
        quiet = np.array([not np.any((u[:, 0] <= s) & (u[:, 1] >= s - args.ask_window)) for s in r], dtype=bool)
        ev["routine instruction, unasked"] = r[quiet] if len(r) else r
        ev = {k: np.searchsorted(t, x, side="right") - 1 for k, x in ev.items()}
        ev = {k: x[(x >= 0)] for k, x in ev.items()}
        videos.append({"name": name, "recipe": rec["recipe"], "fps": float(g["fps"].iloc[0]),
                       "group": G[idx].argmax(1), "fine_third": thirds(G[idx, 0]), "third": thirds(score[idx]),
                       "ev": ev})
    print(f"WTaG: {len(videos)} recordings used, dropped: {dropped}", flush=True)

    def ratios(vs, key, field):
        obs, null = np.zeros(3), np.zeros((args.shifts, 3))
        for v in vs:
            e, s = v["ev"][key], v[field]
            if len(e) == 0:
                continue
            obs += np.bincount(s[e], minlength=3)
            st = s[(e[None, :] + rng.integers(0, len(s), size=args.shifts)[:, None]) % len(s)]
            for k in range(3):
                null[:, k] += (st == k).sum(1)
        exp = np.maximum(null.mean(0), 1e-9)
        return obs, obs / exp, np.percentile(null, 2.5, 0) / exp, np.percentile(null, 97.5, 0) / exp

    keys = ["routine instruction", "routine instruction, unasked", "correction", "answer", "confirmation",
            "user, any utterance", "user question"]
    sets = {"all recordings": videos}
    for rc in sorted({v["recipe"] for v in videos}):
        sets[f"recipe: {rc}"] = [v for v in videos if v["recipe"] == rc]
    sets["frame rate 20 or more"] = [v for v in videos if v["fps"] >= 20]
    allg = np.concatenate([v["group"] for v in videos])
    res = {"recordings": len(videos), "dropped": dropped, "hours": round(len(allg) / 3600, 2),
           "shifts": args.shifts, "holoassist_ticks": int(len(Y)),
           "group_share_of_time": {g: round(float((allg == i).mean()), 3) for i, g in enumerate(GROUPS)},
           "group_share_by_recipe": {rc: {g: round(float((np.concatenate([v["group"] for v in videos
                                                                         if v["recipe"] == rc]) == i).mean()), 3)
                                          for i, g in enumerate(GROUPS)}
                                     for rc in sorted({v["recipe"] for v in videos})},
           "top_predicted_states": {names[j]: round(float(c / len(P)), 3)
                                    for j, c in Counter(P.argmax(1)).most_common(8)},
           "sets": {}}
    fields = (("group", GROUPS), ("fine_third", THIRDS), ("third", THIRDS))
    for sname, vs in sets.items():
        res["sets"][sname] = {"recordings": len(vs)}
        for k in keys:
            cell = {}
            for field, labels in fields:
                obs, r, lo, hi = ratios(vs, k, field)
                cell["n"] = int(obs.sum())
                cell[field] = {l: [round(float(r[i]), 3), round(float(lo[i]), 3), round(float(hi[i]), 3)]
                               for i, l in enumerate(labels)}
            res["sets"][sname][k] = cell
    Path("reports/wtag_from_video.json").write_text(json.dumps(res, indent=2))

    L = ["# WTaG: speech onsets against hand state predicted from video", "",
         "Generated by `python -m scripts.wtag_from_video`. Models trained on HoloAssist "
         f"({res['holoassist_ticks']} ticks) and applied unchanged to {res['recordings']} WTaG recordings "
         f"({res['hours']} h). The models were not checked on WTaG (no hand labels). See the script docstring.", "",
         "Dropped: " + ("; ".join(dropped) if dropped else "none") + ".", "",
         "Predicted group, share of time: " + ", ".join(f"{g} {v:.0%}" for g, v in res["group_share_of_time"].items())
         + ". By recipe: " + "; ".join(f"{rc}: " + ", ".join(f"{g} {v:.0%}" for g, v in sh.items())
                                       for rc, sh in res["group_share_by_recipe"].items())
         + ". Most frequent predicted states: "
         + ", ".join(f"{k} {v:.0%}" for k, v in res["top_predicted_states"].items()) + ".", "",
         f"Each cell: observed / expected by chance [range of 95% of {args.shifts} random shifts]. "
         "'p(fine)' = thirds of each recording's ticks by the fine-motor probability; "
         "'onset score' = thirds by the HoloAssist onset model.", ""]
    for sname, r in res["sets"].items():
        L += [f"## {sname} ({r['recordings']} recordings)", "",
              "| onsets | n | hands: fine | hands: free | hands: other | p(fine): low | mid | high "
              "| onset score: low | mid | high |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            c = r[k]
            cells = [f"{c[f][g][0]:.2f} [{c[f][g][1]:.2f}, {c[f][g][2]:.2f}]" for f, labels in fields for g in labels]
            L.append(f"| {k} | {c['n']} | " + " | ".join(cells) + " |")
        L.append("")
    Path("reports/wtag_from_video.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
