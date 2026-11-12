"""The timing experiment on the OFFICIAL HoloAssist split (train -> val).

Why: the other scripts use our own 5 folds by recording session. Other
groups use the split files of the dataset (data-splits-v1_2.zip). This
script repeats part 5, experiment A, on that split so that the numbers can
be compared and reproduced. Note: the official split shares recording
sessions between train and val, so it is the easier test; our session
folds are the stricter one. The official test videos have no public labels.

Task: one tick per second; target = a routine instructor utterance starts
in the next second. Models (gradient-boosted trees), trained on train:
    task               time since the last utterance and step boundary, progress
    video              32 principal components of the video vector
    predicted hand     hand-state probabilities from the video head
    task + video, task + predicted hand, task + labelled hand (not available to an assistant)
The hand head is trained on train only; its predictions for the train
ticks are cross-fitted (4 folds by session).

Scores on val: average precision, ROC AUC, and an onset F1: a predicted
onset is a tick above a threshold (set on train) with at least 3 s since
the previous predicted onset; it is correct if a true onset lies from 1.5 s
before to 2.5 s after it (the tolerance of EgoVoice, our implementation;
their numbers use other inputs and utterance sets and cannot be compared
directly). Intervals: 1,000 resamples of val videos.

Usage:
    python -m scripts.holoassist_official_split
Outputs: reports/holoassist_official_split.json, reports/holoassist_official_split.md
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.holoassist_from_video import fit_head, predict_head, run_lengths
from scripts.holoassist_predict import tick_features
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

ROOT = "/mnt/ssd/holoassist"


def onset_f1(score: np.ndarray, y: np.ndarray, sl: list, thr: float, gap: int = 3, early: int = 1, late: int = 2):
    """Ticks are 1 s apart. A prediction at tick i (announces an onset in (i, i+1]) is correct if a true
    onset tick lies in [i - early, i + late]. Each true onset can be matched once. -> (f1, precision, recall)."""
    tp = n_pred = n_true = 0
    for a, b in sl:
        s, t = score[a:b], np.flatnonzero(y[a:b] > 0)
        pred, last = [], -10 ** 9
        for i in np.flatnonzero(s >= thr):
            if i - last >= gap:
                pred.append(i); last = i
        used = np.zeros(len(t), dtype=bool)
        for i in pred:
            j = np.flatnonzero(~used & (t >= i - early) & (t <= i + late))
            if len(j):
                used[j[0]] = True; tp += 1
        n_pred += len(pred); n_true += len(t)
    p, r = tp / max(n_pred, 1), tp / max(n_true, 1)
    return (2 * p * r / max(p + r, 1e-9), p, r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    root = Path(resolve_data_path(ROOT))
    z = zipfile.ZipFile(root / "data-splits-v1_2.zip")
    split = {n.split("-")[0]: set(z.read(n).decode().split()) for n in z.namelist()}          # train, val, test
    fd = root / f"features_w{args.window}"
    meta, feats = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy", mmap_mode="r")
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    S = len(top) + 2

    rows, XT, XH, Y, part = [], [], [], [], []
    for name, g in meta.groupby("video_name", sort=False):
        which = "train" if name in split["train"] else "val" if name in split["val"] else None
        v = data.get(name)
        V = build_video(v, top) if v is not None and which else None
        if V is None:
            continue
        t = g["t_end"].to_numpy()
        keep = (t >= V["lo"]) & (t < V["hi"] - 1.0)
        if keep.sum() < 4:
            continue
        f = tick_features(V, v, t[keep], horizon=1.0)
        rows.append(g.index.to_numpy()[keep]); XT.append(f[0]); XH.append(f[1]); Y.append(f[2])
        part += [which] * int(keep.sum())
    rows = np.concatenate(rows); XT = np.concatenate(XT); XH = np.concatenate(XH); Y = np.concatenate(Y)
    part = np.array(part)
    F = np.asarray(feats[rows], dtype=np.float32)
    vid = meta["video_name"].to_numpy()[rows]
    sess = np.array([n.split("-")[0] for n in vid])
    ST = XH[:, 0].astype(int)
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    print(f"train {len(set(vid[tr]))} videos {len(tr)} ticks; val {len(set(vid[va]))} videos {len(va)} ticks", flush=True)

    P = np.zeros((len(Y), S), dtype=np.float32)
    P[va] = predict_head(fit_head(F[tr], ST[tr], S, args.device), F[va], args.device)
    for a, b in GroupKFold(n_splits=4).split(F[tr], ST[tr], sess[tr]):
        P[tr[b]] = predict_head(fit_head(F[tr[a]], ST[tr[a]], S, args.device), F[tr[b]], args.device)
    arg = P.argmax(1)
    idle3 = pd.Series(P[:, S - 1]).groupby(vid).transform(lambda s: s.rolling(3, 1).mean()).to_numpy()
    H = np.concatenate([P, arg[:, None], run_lengths(arg, vid)[:, None], idle3[:, None]], 1)
    Z = PCA(32, random_state=0).fit(F[tr]).transform(F)

    sets = {"task": (XT, None), "video": (Z, None), "predicted hand": (H, None),
            "task + video": (np.c_[XT, Z], None), "task + predicted hand": (np.c_[XT, H], None),
            "task + labelled hand (not available to an assistant)": (np.c_[XT, XH], [4])}
    cut = np.flatnonzero(vid[va][1:] != vid[va][:-1]) + 1
    sl_va = list(zip(np.r_[0, cut], np.r_[cut, len(va)]))
    cut = np.flatnonzero(vid[tr][1:] != vid[tr][:-1]) + 1
    sl_tr = list(zip(np.r_[0, cut], np.r_[cut, len(tr)]))
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(sl_va), len(sl_va)) for _ in range(args.boot)]
    res = {"train_videos": len(sl_tr), "val_videos": len(sl_va), "train_ticks": int(len(tr)), "val_ticks": int(len(va)),
           "val_onsets": int(Y[va].sum()), "base_rate": float(Y[va].mean()),
           "sessions_shared_by_train_and_val": len(set(sess[tr]) & set(sess[va])),
           "hand_state_accuracy_val": float((arg[va] == ST[va]).mean()), "models": {}}
    scores = {}
    for name, (X, cat) in sets.items():
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features=cat,
                                           random_state=0).fit(X[tr], Y[tr])
        p_tr, p = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[va])[:, 1]
        scores[name] = p
        grid = np.quantile(p_tr, np.linspace(0.5, 0.995, 40))
        thr = max(grid, key=lambda q: onset_f1(p_tr, Y[tr], sl_tr, q)[0])       # threshold from train only
        f1 = onset_f1(p, Y[va], sl_va, thr)
        res["models"][name] = {"ap": float(average_precision_score(Y[va], p)), "auc": float(roc_auc_score(Y[va], p)),
                               "f1": f1[0], "precision": f1[1], "recall": f1[2], "threshold": float(thr)}
        print(f"  {name}: AP {res['models'][name]['ap']:.4f}  F1 {f1[0]:.3f}", flush=True)
    yv = Y[va]

    def boot_ap(p):
        out = []
        for b in boots:
            i = np.concatenate([np.arange(*sl_va[j]) for j in b])
            out.append(average_precision_score(yv[i], p[i]))
        return np.array(out)

    base = boot_ap(scores["task"])
    for name, p in scores.items():
        b = boot_ap(p)
        res["models"][name]["ap_interval"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        d = b - base
        res["models"][name]["ap_minus_task"] = [float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    Path("reports/holoassist_official_split.json").write_text(json.dumps(res, indent=2))

    L = ["# HoloAssist, official split: when does the instructor start routine speech?", "",
         "Generated by `python -m scripts.holoassist_official_split`. "
         f"Train: {res['train_videos']} videos, {res['train_ticks']} ticks. Val: {res['val_videos']} videos, "
         f"{res['val_ticks']} ticks, {res['val_onsets']} onsets (base rate {res['base_rate']:.3f}). "
         f"{res['sessions_shared_by_train_and_val']} recording sessions have videos in both train and val, so this split "
         "is easier than the session folds of the other reports. Hand-state accuracy on val (22 states): "
         f"{res['hand_state_accuracy_val']:.2f}. Onset F1: a predicted onset counts if a true onset lies from 1 s before "
         "to 2 s after the predicted tick; threshold set on train.", "",
         "| model | average precision [95% interval] | change against task | ROC AUC | onset F1 | precision | recall |",
         "|---|---|---|---|---|---|---|"]
    for name, r in res["models"].items():
        d = r["ap_minus_task"]
        L.append(f"| {name} | {r['ap']:.3f} [{r['ap_interval'][0]:.3f}, {r['ap_interval'][1]:.3f}] | "
                 + ("-" if name == "task" else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
                 + f" | {r['auc']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    Path("reports/holoassist_official_split.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
