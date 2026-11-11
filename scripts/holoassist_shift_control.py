"""Positive control for scripts.egoextra_from_video.

That script trains two models on HoloAssist and asks, on Ego-EXTRA,
whether speech onsets avoid ticks with predicted fine-motor hand states.
A "no relation" answer there means something only if the same test finds
the relation where we know it exists. This script runs the identical test
on HoloAssist itself:

  * same models (hand-state head; onset score from 32 principal components),
  * same grouping (fine / free / other; score thirds per video),
  * same statistic (observed / expected by circular shifts inside a video),
  * predictions for a session come from models that never saw that session
    (5 folds by recording session).

Events: ticks after which a routine instructor utterance starts within 1 s.
"labelled" rows use the human hand-state label instead of the prediction.

Usage:
    python -m scripts.holoassist_shift_control
Outputs: reports/holoassist_shift_control.json, reports/holoassist_shift_control.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.egoextra_from_video import FINE, FREE, GROUPS, THIRDS, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(0)

    F, ST, Y, names, vid, types = holoassist_ticks(args.window, with_types=True)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    sess = np.array([n.split("-")[0] for n in vid])
    gid = np.array([0 if n in FINE else 1 if n in FREE else 2 for n in names])
    G = np.zeros((len(Y), 3), dtype=np.float32)
    score = np.zeros(len(Y))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, Y, sess)):
        P = predict_head(fit_head(F[tr], ST[tr], len(names), args.device), F[te], args.device)
        G[te] = np.stack([P[:, gid == j].sum(1) for j in range(3)], 1)
        pca = PCA(32, random_state=0).fit(F[tr])
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(pca.transform(F[tr]), Y[tr])
        score[te] = m.predict_proba(pca.transform(F[te]))[:, 1]
        print(f"fold {k + 1}/5", flush=True)

    fields = {"predicted group": G.argmax(1), "labelled group": gid[ST], "onset score third": np.zeros(len(Y), int)}
    for v in np.unique(vid):
        m = vid == v
        lo, hi = np.percentile(score[m], [100 / 3, 200 / 3])
        fields["onset score third"][m] = (score[m] > lo).astype(int) + (score[m] > hi)

    # how good is the classifier at the level that matters for timing: the three groups?
    pg, lg = fields["predicted group"], fields["labelled group"]
    conf = np.array([[int(((lg == i) & (pg == j)).sum()) for j in range(3)] for i in range(3)])
    group_acc = float((pg == lg).mean())
    res = {"ticks": int(len(Y)), "onsets": int(Y.sum()), "videos": int(len(np.unique(vid))), "shifts": args.shifts,
           "share_of_time": {}, "ratio": {}}
    for fname, s in fields.items():
        labels = THIRDS if "third" in fname else GROUPS
        obs, null = np.zeros(3), np.zeros((args.shifts, 3))
        for v in np.unique(vid):
            sv, e = s[vid == v], np.flatnonzero(Y[vid == v] > 0)
            if len(e) == 0:
                continue
            obs += np.bincount(sv[e], minlength=3)
            st = sv[(e[None, :] + rng.integers(0, len(sv), size=args.shifts)[:, None]) % len(sv)]
            for k in range(3):
                null[:, k] += (st == k).sum(1)
        exp = null.mean(0)
        res["share_of_time"][fname] = {l: round(float((s == i).mean()), 3) for i, l in enumerate(labels)}
        res["ratio"][fname] = {l: [round(float(obs[i] / exp[i]), 3),
                                   round(float(np.percentile(null[:, i], 2.5) / exp[i]), 3),
                                   round(float(np.percentile(null[:, i], 97.5) / exp[i]), 3)]
                               for i, l in enumerate(labels)}
    # the pilot's contrast (part 1), now with the hand state seen by the model:
    # routine speech avoids busy hands; do corrections?
    by_type = {}
    for tname, flag in types.items():
        row = {}
        for fname in ("predicted group", "labelled group"):
            sfield = fields[fname]
            obs, null = np.zeros(3), np.zeros((args.shifts, 3))
            for v in np.unique(vid):
                sv, e = sfield[vid == v], np.flatnonzero(flag[vid == v])
                if len(e) == 0:
                    continue
                obs += np.bincount(sv[e], minlength=3)
                st = sv[(e[None, :] + rng.integers(0, len(sv), size=args.shifts)[:, None]) % len(sv)]
                for k in range(3):
                    null[:, k] += (st == k).sum(1)
            exp = null.mean(0)
            row[fname] = {"n": int(obs.sum()),
                          **{g: [round(float(obs[i] / exp[i]), 3), round(float(np.percentile(null[:, i], 2.5) / exp[i]), 3),
                                 round(float(np.percentile(null[:, i], 97.5) / exp[i]), 3)] for i, g in enumerate(GROUPS)}}
        by_type[tname] = row
    res["by_utterance_type"] = by_type
    res["group_accuracy"] = round(group_acc, 4)
    res["group_majority_baseline"] = round(float(np.bincount(lg).max() / len(lg)), 4)
    res["group_confusion_rows_labelled_cols_predicted"] = conf.tolist()
    Path("reports/holoassist_shift_control.json").write_text(json.dumps(res, indent=2))
    L = ["# Positive control: the Ego-EXTRA test, run on HoloAssist", "",
         "Generated by `python -m scripts.holoassist_shift_control`. "
         f"{res['videos']} videos, {res['ticks']} ticks, {res['onsets']} routine instructor onsets. "
         "Predictions are cross-fitted by recording session. "
         f"Each cell: observed / expected by chance [range of 95% of {args.shifts} random shifts]; "
         "second line: share of time in that state.", "",
         "| state from | first | second | third |", "|---|---|---|---|"]
    for fname, r in res["ratio"].items():
        sh = res["share_of_time"][fname]
        L.append(f"| {fname} | " + " | ".join(
            f"{l}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}], time {sh[l]:.0%}" for l, v in r.items()) + " |")
    L += ["", "## By type of instructor utterance", "",
          "Same statistic. Urgent speech (corrections) should not avoid busy hands if the pilot's part 1 holds.", "",
          "| utterance type | state from | n | fine | free | other |", "|---|---|---|---|---|---|"]
    for tname, row in by_type.items():
        for fname, c in row.items():
            L.append(f"| {tname} | {fname} | {c['n']} | " + " | ".join(
                f"{c[g][0]:.2f} [{c[g][1]:.2f}, {c[g][2]:.2f}]" for g in GROUPS) + " |")
    L += ["", "## The classifier at group level", "",
          f"Accuracy on the three groups: {res['group_accuracy']:.3f} "
          f"(always the most common group: {res['group_majority_baseline']:.3f}). "
          "Share of each labelled group (row) that gets each predicted group (column):", "",
          "| labelled | " + " | ".join("predicted " + g for g in GROUPS) + " |", "|---|---|---|---|"]
    L += [f"| {GROUPS[i]} | " + " | ".join(f"{conf[i, j] / conf[i].sum():.2f}" for j in range(3)) + " |" for i in range(3)]
    Path("reports/holoassist_shift_control.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
