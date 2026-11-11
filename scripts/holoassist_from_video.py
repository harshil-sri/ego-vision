"""Pilot part 5: does the hand-state signal survive when it comes from VIDEO?

Parts 1 to 4 read the performer's hand state from human annotations. A
real assistant has only the camera. This script repeats the two
prediction experiments on a subset of HoloAssist videos, with the hand
state replaced by what a model sees.

Inputs
    scripts.holoassist_download_subset   the videos
    scripts.holoassist_video_features    one frozen VideoMAE vector per
                                         second (last 2 s of video)

Experiment A: when does the instructor start routine speech?
    Unit: one tick per second. Target: a routine instructor utterance
    starts in the next second (same definition as part 2).
    Feature sets, same model (gradient-boosted trees):
        task              need proxies only
        task + oracle     + annotated hand state (the part 2 upper bound)
        task + predicted  + hand-state probabilities predicted from video
        task + video      + the video vector itself, reduced to 32 numbers
    "Predicted" hand state: a small classifier on the video vector, trained
    to name the annotated state at the tick (20 verbs, other verb, idle).

Experiment B: when a mistake happens, does the instructor stay silent?
    Same target as part 4, with the video vector of the window that ends
    about 1 s after the mistake starts.

No leakage: all splits are by recording session (5 folds). The hand-state
classifier that produces predictions for a session never saw that
session. Inside each training fold, its predictions for the training
sessions come from an inner split (cross-fitting), so the timing model is
trained on predictions of the same quality it gets at test time.

Usage:
    python -m scripts.holoassist_from_video
Outputs: reports/holoassist_from_video.json, reports/holoassist_from_video.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.holoassist_predict import tick_features
from scripts.holoassist_silence import BY_INSTRUCTOR, BY_STUDENT
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

ROOT = "/mnt/ssd/holoassist"


def fit_head(X, y, n_classes, device, seed=0, epochs=25):
    """Small MLP on video vectors. X (n, 768) float32, y (n,) int -> model."""
    from src.models.frozen import FeatureHead
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    counts = np.bincount(y, minlength=n_classes)
    w = torch.tensor((counts.sum() / (n_classes * np.maximum(counts, 1))) ** 0.5,
                     dtype=torch.float32, device=device)
    m = FeatureHead(X.shape[1], n_classes, kind="mlp", hidden=256, dropout=0.3).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    B = 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * int(np.ceil(len(y) / B)))
    lf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    for _ in range(epochs):
        m.train()
        perm = torch.from_numpy(rng.permutation(len(y))).to(device)
        for s in range(0, len(perm), B):
            b = perm[s:s + B]
            loss = lf(m(Xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


def predict_head(m, X, device):
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(X).to(device)).float(), 1).cpu().numpy()


def run_lengths(arg: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """Seconds the predicted state has stayed the same (ticks are 1 s apart)."""
    out = np.zeros(len(arg))
    for i in range(1, len(arg)):
        if vid[i] == vid[i - 1] and arg[i] == arg[i - 1]:
            out[i] = out[i - 1] + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="read features_w<W>_smoke, write reports/..._smoke")
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    root = Path(resolve_data_path(ROOT))
    fd = root / (f"features_w{args.window}" + ("_smoke" if args.smoke else ""))
    meta = pd.read_csv(fd / "meta.csv")
    feats = np.load(fd / "feats.npy").astype(np.float32)
    tick = float(json.loads((fd / "info.json").read_text())["tick_s"])
    data = {v["video_name"]: v for v in
            json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    names = top + ["other_verb", "idle"]
    S = len(names)

    # ---- per-tick table -------------------------------------------------
    rows, XT, XH, Y, ST = [], [], [], [], []
    wrong = []                                            # experiment B rows
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
        idx = g.index.to_numpy()[keep]
        rows.append(idx); XT.append(f[0]); XH.append(f[1]); Y.append(f[2]); ST.append(f[1][:, 0].astype(int))
        # mistakes: feature = first window ending >= 1 s after the mistake starts
        fine = sorted((e for e in v["events"] if e["label"] == "Fine grained action"), key=lambda e: e["start"])
        utt_end = np.array(sorted(e["end"] for e in v["events"] if e["label"] == "Conversation"
                                  and e["attributes"].get("Conversation Purpose", "").startswith("instructor")))
        n_wrong = n_corr = 0
        t_all, i_all = g["t_end"].to_numpy(), g.index.to_numpy()
        for e in fine:
            c = e["attributes"].get("Action Correctness", "")
            if c in (BY_INSTRUCTOR, BY_STUDENT):
                j = np.searchsorted(t_all, e["start"] + 1.0)
                if j < len(t_all) and t_all[j] <= e["start"] + 2.5:
                    ju = np.searchsorted(utt_end, e["start"], side="right")
                    wrong.append({"row": int(i_all[j]), "session": name.split("-")[0],
                                  "verb": e["attributes"].get("Verb", "?"), "noun": e["attributes"].get("Noun", "?"),
                                  "task": v.get("taskType", "?"), "t": e["start"] - V["lo"],
                                  "frac": (e["start"] - V["lo"]) / (V["hi"] - V["lo"]),
                                  "since_utt": min(e["start"] - utt_end[ju - 1], 300.0) if ju > 0 else 300.0,
                                  "wrong_so_far": n_wrong, "corrected_so_far": n_corr,
                                  "silent": int(c == BY_STUDENT)})
                n_corr += int(c == BY_INSTRUCTOR)
            if c.startswith("Wrong"):
                n_wrong += 1
    rows = np.concatenate(rows); XT = np.concatenate(XT); XH = np.concatenate(XH)
    Y = np.concatenate(Y); ST = np.concatenate(ST)
    Fv = feats[rows]
    vid = meta["video_name"].to_numpy()[rows]
    sess = np.array([n.split("-")[0] for n in vid])
    if args.smoke:                                        # too few sessions in a smoke run: split by video
        sess = vid.copy()
    print(f"ticks {len(Y)}  onsets {int(Y.sum())} ({Y.mean():.3%})  videos {len(set(vid))}  sessions {len(set(sess))}")

    # ---- experiment A ---------------------------------------------------
    outer = list(GroupKFold(n_splits=5).split(Fv, Y, sess))
    P = np.zeros((len(Y), S), dtype=np.float32)           # test-fold hand-state predictions
    res = {k: {"auc": [], "ap": []} for k in
           ["task", "task + oracle hand", "task + predicted hand", "task + video", "oracle hand only",
            "predicted hand only", "video only"]}
    for k, (tr, te) in enumerate(outer):
        head = fit_head(Fv[tr], ST[tr], S, args.device)
        P[te] = predict_head(head, Fv[te], args.device)
        Ptr = np.zeros((len(tr), S), dtype=np.float32)    # cross-fitted predictions for the training ticks
        for a, b in GroupKFold(n_splits=4).split(Fv[tr], ST[tr], sess[tr]):
            Ptr[b] = predict_head(fit_head(Fv[tr][a], ST[tr][a], S, args.device), Fv[tr][b], args.device)
        pca = PCA(32, random_state=0).fit(Fv[tr])

        def pred_feats(prob, v_ids):
            arg = prob.argmax(1)
            idle3 = pd.Series(prob[:, S - 1]).groupby(v_ids).transform(lambda s: s.rolling(3, 1).mean()).to_numpy()
            return np.concatenate([prob, arg[:, None], run_lengths(arg, v_ids)[:, None], idle3[:, None]], 1)

        sets = {
            "task": (XT[tr], XT[te], None),
            "task + oracle hand": (np.c_[XT[tr], XH[tr]], np.c_[XT[te], XH[te]], [4]),
            "task + predicted hand": (np.c_[XT[tr], pred_feats(Ptr, vid[tr])],
                                      np.c_[XT[te], pred_feats(P[te], vid[te])], None),
            "task + video": (np.c_[XT[tr], pca.transform(Fv[tr])], np.c_[XT[te], pca.transform(Fv[te])], None),
            "oracle hand only": (XH[tr], XH[te], [0]),
            "predicted hand only": (pred_feats(Ptr, vid[tr]), pred_feats(P[te], vid[te]), None),
            "video only": (pca.transform(Fv[tr]), pca.transform(Fv[te]), None),
        }
        for name, (a, b, cat) in sets.items():
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features=cat,
                                               random_state=0).fit(a, Y[tr])
            p = m.predict_proba(b)[:, 1]
            res[name]["auc"].append(roc_auc_score(Y[te], p)); res[name]["ap"].append(average_precision_score(Y[te], p))
        print(f"  fold {k + 1}/5 done", flush=True)

    pred = P.argmax(1)
    busy_auc = roc_auc_score(ST != S - 1, 1 - P[:, S - 1])
    recall = {names[j]: round(float((pred[ST == j] == j).mean()), 3) for j in range(S) if (ST == j).sum() >= 300}
    hand = {"accuracy": round(float((pred == ST).mean()), 4),
            "majority_baseline": round(float(np.bincount(ST).max() / len(ST)), 4),
            "mean_class_recall": round(float(np.mean(list(recall.values()))), 4),
            "idle_vs_busy_auc": round(float(busy_auc), 4), "recall": recall}
    out = {"ticks": int(len(Y)), "onsets": int(Y.sum()), "base_rate": round(float(Y.mean()), 4),
           "videos": int(len(set(vid))), "sessions": int(len(set(sess))), "tick_s": tick,
           "hand_state_from_video": hand,
           "timing": {k: {"auc": round(float(np.mean(r["auc"])), 4), "ap": round(float(np.mean(r["ap"])), 4),
                          "ap_folds": [round(float(x), 4) for x in r["ap"]]} for k, r in res.items()}}

    # ---- experiment B ---------------------------------------------------
    W = pd.DataFrame(wrong)
    if len(W) >= 200:
        def codes(col, topk=None):
            keep = [k for k, _ in Counter(W[col]).most_common(topk)]
            return W[col].map({k: i for i, k in enumerate(keep)}).fillna(len(keep)).to_numpy(float)
        hist = W[["t", "frac", "since_utt", "wrong_so_far", "corrected_so_far"]].to_numpy(float)
        lab = np.c_[codes("verb", 40), codes("noun", 40), codes("task")]
        Fw, yw, gw = feats[W["row"].to_numpy()], W["silent"].to_numpy(), W["session"].to_numpy()
        nd = (W["verb"] != "drop").to_numpy()
        sil = {}
        for tag, keep in (("all mistakes", np.ones(len(W), bool)), ("without drop", nd)):
            r = {k: [] for k in ["history", "video", "history + video", "labels + history (oracle)"]}
            for tr, te in GroupKFold(n_splits=5).split(Fw[keep], yw[keep], gw[keep]):
                pca = PCA(16, random_state=0).fit(Fw[keep][tr])
                Z = pca.transform(Fw[keep])
                H, L_, y_ = hist[keep], lab[keep], yw[keep]
                sets = {"history": (H, None), "video": (Z, None), "history + video": (np.c_[H, Z], None),
                        "labels + history (oracle)": (np.c_[L_, H], [0, 1, 2])}
                for name, (X, cat) in sets.items():
                    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4,
                                                       categorical_features=cat, random_state=0).fit(X[tr], y_[tr])
                    r[name].append(roc_auc_score(y_[te], m.predict_proba(X[te])[:, 1]))
            sil[tag] = {"n": int(keep.sum()), "silent_share": round(float(yw[keep].mean()), 3),
                        "auc": {k: round(float(np.mean(v)), 4) for k, v in r.items()}}
        out["silence"] = sil

    stem = "reports/holoassist_from_video" + ("_smoke" if args.smoke else "")
    Path(stem + ".json").write_text(json.dumps(out, indent=2))
    L = ["# HoloAssist pilot part 5: the same questions, from video", "",
         "Generated by `python -m scripts.holoassist_from_video`. "
         f"{out['videos']} videos from {out['sessions']} recording sessions (a random subset, whole sessions), "
         f"{out['ticks']} ticks of {tick} s. Video = one frozen VideoMAE vector per tick (last {args.window} s). "
         "All splits by session, 5 folds; hand-state predictions are cross-fitted. "
         "See the script docstring for the design.", "",
         "## Can the model see the hand state?", "",
         f"- {S}-way hand state (20 verbs, other verb, idle): accuracy {hand['accuracy']} "
         f"(always the most common state: {hand['majority_baseline']}), mean per-state recall {hand['mean_class_recall']}.",
         f"- Idle against busy: ROC AUC {hand['idle_vs_busy_auc']}.",
         f"- Recall per state: {hand['recall']}", "",
         "## Experiment A: predicting routine instructor speech onsets", "",
         f"Base rate {out['base_rate']:.2%} (chance average precision).", "",
         "| features | ROC AUC | average precision | AP per fold |", "|---|---|---|---|"]
    for k, r in out["timing"].items():
        L.append(f"| {k} | {r['auc']} | {r['ap']} | {r['ap_folds']} |")
    if "silence" in out:
        L += ["", "## Experiment B: does the instructor stay silent on a mistake?", "",
              "ROC AUC, chance 0.5. Video = the window that ends about 1 s after the mistake starts.", "",
              "| mistakes | n | left to performer | history | video | history + video | labels + history (oracle) |",
              "|---|---|---|---|---|---|---|"]
        for tag, r in out["silence"].items():
            a = r["auc"]
            L.append(f"| {tag} | {r['n']} | {r['silent_share']} | {a['history']} | {a['video']} | "
                     f"{a['history + video']} | {a['labels + history (oracle)']} |")
    Path(stem + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
