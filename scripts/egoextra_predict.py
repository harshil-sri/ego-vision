"""Ego-EXTRA: does video help to PREDICT who speaks next, and when?

The shift tests (scripts.egoextra_from_video) ask about one hand-state
variable at a time. This script asks the practical question: at each
second, how well can a model predict that a turn starts within the next
second, and does the video add anything to the dialogue history?

Unit: one tick per second of each video (the ticks of the VideoMAE
features; the window of a tick ends at the tick, so nothing after it is
seen).

Targets (each is rare, so average precision is compared with its base rate)
    trainee turn           the trainee starts to speak
    expert, unasked, long  the expert speaks up although the trainee was
                           silent for 5 s (at least 1.5 s of speech)
    expert, after trainee  the expert starts a long turn after trainee speech

Feature sets, same model (gradient-boosted trees, 5 folds, whole videos held out)
    history           seconds since the trainee last spoke, since the
                      expert last spoke, who spoke last, time in the video
    video             32 principal components of the VideoMAE vector
    history + video
    history + hands   + the 3 hand-state group probabilities from the
                      HoloAssist-trained head (fine, free, other)

Intervals: videos resampled 500 times, differences computed inside each
resample.

Usage:
    python -m scripts.egoextra_predict
Outputs: reports/egoextra_predict.json, reports/egoextra_predict.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.egoextra_from_video import FINE, FREE, holoassist_ticks
from scripts.egoextra_speech import expert_turns, load_transcript
from scripts.holoassist_from_video import fit_head, predict_head
from src.paths import resolve_data_path

EGO = "/mnt/ssd/ego-extra"
TARGETS = ["trainee turn", "expert, unasked, long", "expert, after trainee"]
SETS = ["history", "video", "history + video", "history + hands"]


def since_last(ends: np.ndarray, t: np.ndarray, cap: float = 120.0) -> np.ndarray:
    """Seconds from the last utterance END at or before t; cap if none or long ago.
    0 means the speaker is talking at t (an utterance started and has not ended)."""
    j = np.searchsorted(ends, t, side="right")
    return np.minimum(np.where(j > 0, t - ends[np.clip(j - 1, 0, None)], cap), cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    root = Path(resolve_data_path(EGO))
    fd = root / f"features_w{args.window}"
    meta, F = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy").astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    tr_files = {p.name.replace("_transcript.txt", ""): p for p in (root / "transcript").rglob("*_transcript.txt")}

    rows, H, Y, vid = [], [], {k: [] for k in TARGETS}, []
    for name, g in meta.groupby("video_name", sort=False):
        if name not in tr_files:
            continue
        u = load_transcript(tr_files[name])
        t = g["t_end"].to_numpy()
        if len(u) < 5 or len(t) < 60 or abs(t[-1] - u[:, 1].max()) > 60:
            continue
        e, a = u[u[:, 2] == 1], u[u[:, 2] == 0]
        # an utterance in progress counts as "0 s ago": use its start as an end that already happened
        s_tr = np.minimum(since_last(np.sort(a[:, 1]), t), np.where(
            (np.searchsorted(np.sort(a[:, 0]), t, side="right") > np.searchsorted(np.sort(a[:, 1]), t, side="right")), 0.0, 1e9))
        s_ex = np.minimum(since_last(np.sort(e[:, 1]), t), np.where(
            (np.searchsorted(np.sort(e[:, 0]), t, side="right") > np.searchsorted(np.sort(e[:, 1]), t, side="right")), 0.0, 1e9))
        H.append(np.c_[s_tr, s_ex, (s_tr < s_ex).astype(float), t])
        starts, unasked, speech = expert_turns(u, 2.0, 5.0)
        long_ = speech >= 1.5
        tr = a[np.r_[True, a[1:, 0] - np.maximum.accumulate(a[:-1, 1]) >= 2.0], 0]
        ev = {"trainee turn": tr, "expert, unasked, long": starts[unasked & long_],
              "expert, after trainee": starts[~unasked & long_]}
        for k, x in ev.items():                                        # onset in (t, t + 1]
            x = np.sort(x)
            Y[k].append((np.searchsorted(x, t + 1.0, side="right") - np.searchsorted(x, t, side="right")) > 0)
        rows.append(g.index.to_numpy()); vid += [name] * len(t)
    rows, H, vid = np.concatenate(rows), np.concatenate(H), np.array(vid)
    Y = {k: np.concatenate(v).astype(int) for k, v in Y.items()}
    Fv = F[rows]

    # hand-state group probabilities from the HoloAssist-trained head (never saw Ego-EXTRA)
    Fh, ST, _, names, _ = holoassist_ticks(args.window)
    Fh = (Fh - Fh.mean(0)) / (Fh.std(0) + 1e-6)
    P = predict_head(fit_head(Fh, ST, len(names), args.device), Fv, args.device)
    gid = np.array([0 if n in FINE else 1 if n in FREE else 2 for n in names])
    G = np.stack([P[:, gid == k].sum(1) for k in range(3)], 1)
    print(f"{len(vid)} ticks, {len(set(vid))} videos; base rates "
          + ", ".join(f"{k} {Y[k].mean():.3%}" for k in TARGETS), flush=True)

    oof = {k: {s: np.zeros(len(vid)) for s in SETS} for k in TARGETS}
    for fold, (tr_, te) in enumerate(GroupKFold(n_splits=5).split(Fv, groups=vid)):
        pca = PCA(32, random_state=0).fit(Fv[tr_])
        Z = pca.transform(Fv)
        X = {"history": H, "video": Z, "history + video": np.c_[H, Z], "history + hands": np.c_[H, G]}
        for k in TARGETS:
            for s in SETS:
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[s][tr_], Y[k][tr_])
                oof[k][s][te] = m.predict_proba(X[s][te])[:, 1]
        print(f"fold {fold + 1}/5", flush=True)

    rng = np.random.default_rng(0)
    by_vid = {v: np.flatnonzero(vid == v) for v in np.unique(vid)}
    keys = list(by_vid)
    res = {"ticks": int(len(vid)), "videos": len(keys), "boot": args.boot, "targets": {}}
    for k in TARGETS:
        r = {"base_rate": round(float(Y[k].mean()), 4), "n": int(Y[k].sum()),
             "ap": {s: round(float(average_precision_score(Y[k], oof[k][s])), 4) for s in SETS},
             "auc": {s: round(float(roc_auc_score(Y[k], oof[k][s])), 4) for s in SETS}}
        d = {"history + video minus history": [], "history + hands minus history": []}
        for _ in range(args.boot):
            idx = np.concatenate([by_vid[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
            base = average_precision_score(Y[k][idx], oof[k]["history"][idx])
            d["history + video minus history"].append(average_precision_score(Y[k][idx], oof[k]["history + video"][idx]) - base)
            d["history + hands minus history"].append(average_precision_score(Y[k][idx], oof[k]["history + hands"][idx]) - base)
        r["ap_difference_95"] = {n: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4),
                                     round(float(np.percentile(v, 97.5)), 4)] for n, v in d.items()}
        res["targets"][k] = r
    Path("reports/egoextra_predict.json").write_text(json.dumps(res, indent=2))

    L = ["# Ego-EXTRA: predicting who starts to speak in the next second", "",
         "Generated by `python -m scripts.egoextra_predict`. "
         f"{res['videos']} videos, {res['ticks']} ticks of 1 s, 5 folds with whole videos held out. "
         "Average precision (AP); the base rate is the AP of a random guess. See the script docstring.", "",
         "| target | n | base rate | " + " | ".join(SETS) + " | video adds to history | hand groups add to history |",
         "|---|---|---|" + "---|" * (len(SETS) + 2)]
    for k, r in res["targets"].items():
        dv, dh = r["ap_difference_95"]["history + video minus history"], r["ap_difference_95"]["history + hands minus history"]
        L.append(f"| {k} | {r['n']} | {r['base_rate']:.4f} | " + " | ".join(f"{r['ap'][s]:.4f}" for s in SETS)
                 + f" | {dv[0]:+.4f} [{dv[1]:+.4f}, {dv[2]:+.4f}] | {dh[0]:+.4f} [{dh[1]:+.4f}, {dh[2]:+.4f}] |")
    L += ["", "ROC AUC, same order: " + "; ".join(
        f"{k}: " + ", ".join(f"{r['auc'][s]:.3f}" for s in SETS) for k, r in res["targets"].items())]
    Path("reports/egoextra_predict.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
