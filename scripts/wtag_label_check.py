"""WTaG: a labelled sample to separate two explanations.

scripts.wtag_from_video found that instructor speech on WTaG does not
avoid the fine-motor seconds predicted by the HoloAssist hand-state head.
Either instructors in a kitchen do not hold routine speech during such
work, or the head does not see such work in kitchen video. WTaG has no
hand labels, so 240 moments were labelled by eye:

    120 onsets of a routine instruction (random sample, recordings with a
        frame rate of 20 or more) and, for each, one random second of the
        same recording.
    Each moment was shown as 4 frames of the 2 s that end at the moment
    (0.5 s apart, 384 x 216). The 240 items were shuffled and the key
    (onset or random) was not looked at before all labels were written.

    S  screw, unscrew or insert in progress (the "fine" group of the
       head): twisting a lid, putting the filter into the cone, toothpicks
       into the wrap
    B  other precise work in progress: pour, spread, cut, scoop or
       measure, stir, fold, squeeze, wipe
    F  everything else: idle, hold, reach, carry, place, look, hands not
       in view

Correction, 2026-11-14. The first labels were made from wrong pictures:
the WTaG videos have packets without a time stamp, and a seek with
ffmpeg -ss returned the last key frame, up to some seconds off, so most
rows showed the same picture two to four times. All 417 moments were
labelled again, blind and in a new order, from frames read in order from
the start of each file. 73% of the labels stayed the same. The csv holds
the new labels.

The labeller was the AI assistant that wrote this project's code, not a
trained human annotator, with one pass and no second rater. The labels
are in reports/wtag_hand_labels.csv so that a person can check them.

Outputs
    1. Share of S and of S or B at onsets and at random seconds, and the
       ratio (bootstrap over the 120 pairs).
    2. The head against the labels on the same 240 moments.

Usage:  python -m scripts.wtag_label_check
Outputs: reports/wtag_label_check.json, reports/wtag_label_check.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.egoextra_from_video import FINE, FREE, GROUPS, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head
from scripts.wtag_from_video import WTAG
from src.paths import resolve_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    lab = pd.read_csv("reports/wtag_hand_labels.csv")
    rd = lab[lab.kind == "random"]
    KINDS = (("onset", "instructor: routine instruction"), ("user_question", "user: question"))

    # ---- 1. labels at speech onsets against random seconds ----------------
    # Second batch (ids 241 and up): 119 user-question onsets, drawn per recording like the first
    # batch's random seconds, and 58 new random seconds, shuffled together and labelled blind.
    # All random seconds of both batches are the reference; the two samples are resampled apart.
    res = {"items": int(len(lab)),
           "label_counts": {k: {l: int((lab[lab.kind == k].label == l).sum()) for l in "SBF"}
                            for k in ("onset", "user_question", "random")},
           "ratio": {}}
    for kind, kname in KINDS:
        on = lab[lab.kind == kind]
        for name, cls in (("S (screw, unscrew, insert)", "S"), ("S or B (any precise work)", "SB")):
            a, b = on.label.isin(list(cls)).to_numpy(), rd.label.isin(list(cls)).to_numpy()
            i = rng.integers(0, len(a), size=(args.boot, len(a)))
            j = rng.integers(0, len(b), size=(args.boot, len(b)))
            r = a[i].mean(1) / np.maximum(b[j].mean(1), 1e-9)
            res["ratio"][f"{kname} | {name}"] = {
                "onset": round(float(a.mean()), 3), "random": round(float(b.mean()), 3),
                "ratio": round(float(a.mean() / b.mean()), 2),
                "ci95": [round(float(np.percentile(r, 2.5)), 2), round(float(np.percentile(r, 97.5)), 2)]}

    # ---- 2. the head against the labels -----------------------------------
    Fh, ST, _, names, _ = holoassist_ticks(args.window)
    Fh = (Fh - Fh.mean(0)) / (Fh.std(0) + 1e-6)
    head = fit_head(Fh, ST, len(names), args.device)
    gid = np.array([0 if n in FINE else 1 if n in FREE else 2 for n in names])
    fd = Path(resolve_data_path(WTAG)) / f"features_w{args.window}"
    meta, Fw = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    Fw = (Fw - Fw.mean(0)) / (Fw.std(0) + 1e-6)
    P = predict_head(head, Fw, args.device)
    G = np.stack([P[:, gid == k].sum(1) for k in range(3)], 1)
    by = {n: g for n, g in meta.groupby("video_name", sort=False)}
    rows = []
    for r in lab.itertuples():
        g = by[r.recording]
        j = np.searchsorted(g["t_end"].to_numpy(), r.t_s, side="right") - 1    # last tick ending at or before
        rows.append(g.index[max(j, 0)])
    Gs = G[rows]
    pred = Gs.argmax(1)
    y = lab.label.to_numpy()
    res["head"] = {
        "predicted_share": {g: round(float((pred == i).mean()), 3) for i, g in enumerate(GROUPS)},
        "predicted_by_label": {l: {g: int(((y == l) & (pred == i)).sum()) for i, g in enumerate(GROUPS)} for l in "SBF"},
        "precision_fine_for_S": round(float((y[pred == 0] == "S").mean()), 3),
        "precision_fine_for_S_or_B": round(float(np.isin(y[pred == 0], ["S", "B"]).mean()), 3),
        "recall_of_S": round(float((pred[y == "S"] == 0).mean()), 3),
        "auc_p_fine_S_vs_rest": round(float(roc_auc_score(y == "S", Gs[:, 0])), 3),
        "auc_p_fine_SB_vs_F": round(float(roc_auc_score(y != "F", Gs[:, 0])), 3),
        "auc_p_free_F_vs_SB": round(float(roc_auc_score(y == "F", Gs[:, 1])), 3),
        "mean_p_fine_by_label": {l: round(float(Gs[y == l, 0].mean()), 3) for l in "SBF"},
    }
    Path("reports/wtag_label_check.json").write_text(json.dumps(res, indent=2))

    h, c = res["head"], res["label_counts"]
    L = [f"# WTaG: {len(lab)} moments labelled by eye", "",
         "Generated by `python -m scripts.wtag_label_check` from `reports/wtag_hand_labels.csv`. "
         "Labels by the AI assistant, blind to onset or random, one pass, no second rater. "
         "See the script docstring for the protocol.", "",
         "## 1. Speech onsets against random seconds of the same recordings", "",
         "| moments | n | S | B | F |", "|---|---|---|---|---|"]
    for k in ("onset", "user_question", "random"):
        L.append(f"| {k} | {sum(c[k].values())} | {c[k]['S']} | {c[k]['B']} | {c[k]['F']} |")
    L += ["", "| onsets | hands | share at onsets | share at random seconds | ratio | 95% interval |",
          "|---|---|---|---|---|---|"]
    for name, r in res["ratio"].items():
        who, hands = name.split(" | ")
        L.append(f"| {who} | {hands} | {r['onset']:.0%} | {r['random']:.0%} | {r['ratio']:.2f} | "
                 f"{r['ci95'][0]:.2f} to {r['ci95'][1]:.2f} |")
    L += ["", "## 2. The HoloAssist hand-state head against the labels", "",
          f"Predicted group, share of the {len(lab)} moments: "
          + ", ".join(f"{g} {v:.0%}" for g, v in h["predicted_share"].items()) + ".", "",
          "| label | n | predicted fine | predicted free | predicted other | mean p(fine) |", "|---|---|---|---|---|---|"]
    for l in "SBF":
        d = h["predicted_by_label"][l]
        L.append(f"| {l} | {sum(d.values())} | {d['fine']} | {d['free']} | {d['other']} | {h['mean_p_fine_by_label'][l]:.2f} |")
    L += ["",
          f"- Of the moments the head calls fine, {h['precision_fine_for_S']:.0%} are S and "
          f"{h['precision_fine_for_S_or_B']:.0%} are S or B.",
          f"- Of the S moments, the head calls {h['recall_of_S']:.0%} fine.",
          f"- Ranking by p(fine): AUC {h['auc_p_fine_S_vs_rest']:.2f} for S against the rest, "
          f"{h['auc_p_fine_SB_vs_F']:.2f} for S or B against F. "
          f"Ranking by p(free): AUC {h['auc_p_free_F_vs_SB']:.2f} for F against S or B.", ""]
    Path("reports/wtag_label_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
