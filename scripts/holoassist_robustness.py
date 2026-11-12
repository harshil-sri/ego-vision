"""Robustness checks for the main HoloAssist result.

Main result (scripts.holoassist_shift_control): routine instructor speech
starts in seconds with predicted fine-motor work (screw, unscrew, insert) at
about 0.36 times the chance rate. This script asks how much that number
depends on choices we made.

Statistic everywhere: observed onsets in a state / onsets expected by
chance. Chance = 1,000 circular shifts of the onsets inside each video;
the range holds 95% of the shifts.

Checks
    A  the model: MLP head (reference), logistic regression on 64 principal
       components, a head that never saw the OBJECT of the test video
       (leave one object family out), and, with --tag _dino, a second
       encoder (DINOv2 single frames, no motion input).
    B  each task type separately.
    C  each recording session separately: how many sessions show avoidance?
    D  each of the 22 hand states separately (labels): is the grouping
       "screw, unscrew, insert" a fair summary, or picked after the fact?
    E  time alignment: the hand state 1 s and 2 s BEFORE the onset.

Usage:
    python -m scripts.holoassist_robustness               # VideoMAE features
    python -m scripts.holoassist_robustness --tag _dino   # second encoder; writes ..._dino
Outputs: reports/holoassist_robustness[<tag>].json and .md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.egoextra_from_video import FINE, FREE, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head
from scripts.holoassist_timing import LABELS
from src.paths import resolve_data_path

FAMILIES = ["stool", "nightstand", "tray table", "utility cart", "printer", "coffee", "switch", "camera", "gopro",
            "computer", "circuit breaker", "laser scanner", "motorcycle", "belt"]


def family(task: str) -> str:
    return next((f for f in FAMILIES if f in task), task)


def slices(vid: np.ndarray) -> list[tuple[int, int]]:
    """Ticks of one video are contiguous. -> [(start, stop)] per video."""
    cut = np.flatnonzero(vid[1:] != vid[:-1]) + 1
    return list(zip(np.r_[0, cut], np.r_[cut, len(vid)]))


def shift_ratio(state: np.ndarray, event: np.ndarray, sl: list, rng, shifts: int = 1000, lag: int = 0):
    """state (n,) bool, event (n,) bool, sl: video slices. lag: read the state `lag` ticks before the event.
    -> [observed / expected, low, high of 95% of shifts, number of events]."""
    obs, null = 0.0, np.zeros(shifts)
    for a, b in sl:
        e = np.flatnonzero(event[a:b]) - lag
        e = e[e >= 0]
        if len(e) == 0:
            continue
        s = state[a:b]
        obs += s[e].sum()
        null += s[(e[None, :] + rng.integers(0, b - a, size=shifts)[:, None]) % (b - a)].sum(1)
    ex = null.mean()
    if ex == 0:
        return [float("nan")] * 3 + [int(event.sum())]
    return [float(obs / ex), float(np.percentile(null, 2.5) / ex), float(np.percentile(null, 97.5) / ex),
            int(sum(event[a:b].sum() for a, b in sl))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="'_dino' reads features_dino_w<W> (second encoder)")
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    rng = np.random.default_rng(0)

    F, ST, Y, names, vid, types = holoassist_ticks(args.window, with_types=True, tag=args.tag)
    types.pop("_t_end")
    Y = Y > 0
    corr = types["correct"]
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    task_of = {v["video_name"]: v.get("taskType", "?") for v in
               json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    task = np.array([task_of[n] for n in vid])
    fam = np.array([family(t) for t in task])
    sess = np.array([n.split("-")[0] for n in vid])
    is_fine = np.array([n in FINE for n in names])
    lab_fine = is_fine[ST]
    sl = slices(vid)
    S = len(names)

    def oof(splits, fit):
        p = np.zeros(len(ST), dtype=np.float32)
        for k, (tr, te) in enumerate(splits):
            p[te] = fit(tr, te)
            print(f"    fold {k + 1}", flush=True)
        return p

    def mlp(tr, te):
        P = predict_head(fit_head(F[tr], ST[tr], S, args.device), F[te], args.device)
        return P.argmax(1)

    def logreg(tr, te):
        pca = PCA(64, random_state=0).fit(F[tr])
        sub = tr[np.random.default_rng(0).permutation(len(tr))[:120_000]]      # enough for 64 inputs
        m = LogisticRegression(max_iter=300, C=1.0).fit(pca.transform(F[sub]), ST[sub])
        return m.predict(pca.transform(F[te]))

    by_sess = list(GroupKFold(n_splits=5).split(F, ST, sess))
    preds = {}
    print("MLP head, split by session", flush=True)
    preds["MLP head, split by session (reference)"] = oof(by_sess, mlp)
    print("logistic regression", flush=True)
    preds["logistic regression on 64 components, split by session"] = oof(by_sess, logreg)
    print("leave one object out", flush=True)
    preds["MLP head, object of the test video never seen"] = oof(LeaveOneGroupOut().split(F, ST, fam), mlp)

    res = {"tag": args.tag or "videomae", "ticks": int(len(Y)), "videos": len(sl), "sessions": len(set(sess)),
           "routine_onsets": int(Y.sum()), "shifts": args.shifts, "fine_share_of_time": float(lab_fine.mean())}

    # ---- A: the model -----------------------------------------------------
    A = {"labels (no model)": {"routine": shift_ratio(lab_fine, Y, sl, rng, args.shifts),
                               "correction": shift_ratio(lab_fine, corr, sl, rng, args.shifts)}}
    for name, arg in preds.items():
        pf = is_fine[arg.astype(int)]
        A[name] = {"routine": shift_ratio(pf, Y, sl, rng, args.shifts),
                   "correction": shift_ratio(pf, corr, sl, rng, args.shifts),
                   "fine_recall": float(pf[lab_fine].mean()), "fine_precision": float(lab_fine[pf].mean()),
                   "state_accuracy": float((arg == ST).mean())}
    res["A_model"] = A
    ref = is_fine[preds["MLP head, split by session (reference)"].astype(int)]
    loo = is_fine[preds["MLP head, object of the test video never seen"].astype(int)]

    # ---- B: each task type --------------------------------------------------
    B = {}
    for t in sorted(set(task), key=lambda x: -(task == x).sum()):
        tsl = [(a, b) for a, b in sl if task[a] == t]
        m = task == t
        B[t] = {"videos": len(tsl), "fine_share_of_time": float(lab_fine[m].mean()),
                "labels": shift_ratio(lab_fine, Y, tsl, rng, args.shifts),
                "predicted": shift_ratio(ref, Y, tsl, rng, args.shifts),
                "predicted, object never seen": shift_ratio(loo, Y, tsl, rng, args.shifts)}
    res["B_task"] = B

    # ---- C: each session ----------------------------------------------------
    rows = []
    for s_ in sorted(set(sess)):
        ssl = [(a, b) for a, b in sl if sess[a] == s_]
        n = int(sum(Y[a:b].sum() for a, b in ssl))
        fine_t = float(np.mean(np.concatenate([lab_fine[a:b] for a, b in ssl])))
        if n < 20 or fine_t < 0.05:
            continue
        rows.append([shift_ratio(lab_fine, Y, ssl, rng, 200)[0], shift_ratio(ref, Y, ssl, rng, 200)[0], n])
    R = np.array(rows)
    res["C_session"] = {"sessions_used": int(len(R)), "rule": "at least 20 routine onsets and 5% fine-motor time",
                        **{k: {"median": float(np.nanmedian(R[:, i])), "p10": float(np.nanpercentile(R[:, i], 10)),
                               "p90": float(np.nanpercentile(R[:, i], 90)),
                               "share_below_1": float(np.nanmean(R[:, i] < 1)),
                               "share_below_0.7": float(np.nanmean(R[:, i] < 0.7))}
                           for i, k in enumerate(["labels", "predicted"])}}

    # ---- D: each hand state (labels) ------------------------------------------
    D = {}
    for j, n in enumerate(names):
        m = ST == j
        D[n] = {"share_of_time": float(m.mean()), "routine": shift_ratio(m, Y, sl, rng, args.shifts),
                "correction": shift_ratio(m, corr, sl, rng, args.shifts),
                "group": "fine" if n in FINE else "free" if n in FREE else "other"}
    res["D_state"] = D

    # ---- E: time alignment ------------------------------------------------------
    res["E_lag"] = {f"{lag} s before": {"labels": shift_ratio(lab_fine, Y, sl, rng, args.shifts, lag),
                                        "predicted": shift_ratio(ref, Y, sl, rng, args.shifts, lag)}
                    for lag in (0, 1, 2, 4)}

    out = f"reports/holoassist_robustness{args.tag}"
    Path(out + ".json").write_text(json.dumps(res, indent=2).replace("NaN", "null"))

    def c(v):
        return "n/a" if v[0] is None or v[0] != v[0] else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"

    L = [f"# Robustness of the main HoloAssist result ({res['tag']} features)", "",
         f"Generated by `python -m scripts.holoassist_robustness{' --tag ' + args.tag if args.tag else ''}`. "
         f"{res['videos']} videos, {res['sessions']} sessions, {res['routine_onsets']} routine instructor onsets. "
         "Each cell: onsets observed in fine-motor seconds (screw, unscrew, insert) / onsets expected by chance "
         f"[range of 95% of {args.shifts} random shifts]. Below 1 = the instructor avoids these seconds. "
         f"Fine-motor work covers {res['fine_share_of_time']:.0%} of the time.", "",
         "## A. Does the result depend on the model?", "",
         "| hand state from | routine speech | corrections | fine recall | fine precision | accuracy, 22 states |",
         "|---|---|---|---|---|---|"]
    for n, r in A.items():
        L.append(f"| {n} | {c(r['routine'])} | {c(r['correction'])} | " + (
            f"{r['fine_recall']:.2f} | {r['fine_precision']:.2f} | {r['state_accuracy']:.2f} |"
            if "fine_recall" in r else "- | - | - |"))
    L += ["", "## B. Each task type", "",
          "| task | videos | fine-motor time | labels | predicted | predicted, object never seen |", "|---|---|---|---|---|---|"]
    for t, r in B.items():
        L.append(f"| {t} | {r['videos']} | {r['fine_share_of_time']:.0%} | {c(r['labels'])} | {c(r['predicted'])} | "
                 f"{c(r['predicted, object never seen'])} |")
    cs = res["C_session"]
    L += ["", "## C. Each recording session", "",
          f"{cs['sessions_used']} sessions with {cs['rule']}. Ratio per session:", "",
          "| hand state from | median | 10th to 90th percentile | sessions below 1 | sessions below 0.7 |", "|---|---|---|---|---|"]
    for k in ("labels", "predicted"):
        r = cs[k]
        L.append(f"| {k} | {r['median']:.2f} | {r['p10']:.2f} to {r['p90']:.2f} | {r['share_below_1']:.0%} | "
                 f"{r['share_below_0.7']:.0%} |")
    L += ["", "## D. Each labelled hand state", "",
          "| state | group | share of time | routine speech | corrections |", "|---|---|---|---|---|"]
    for n, r in sorted(D.items(), key=lambda kv: kv[1]["routine"][0] if kv[1]["routine"][0] == kv[1]["routine"][0] else 9):
        L.append(f"| {n} | {r['group']} | {r['share_of_time']:.1%} | {c(r['routine'])} | {c(r['correction'])} |")
    L += ["", "## E. Hand state before the onset", "", "| hand state read | labels | predicted |", "|---|---|---|"]
    for k, r in res["E_lag"].items():
        L.append(f"| {k} | {c(r['labels'])} | {c(r['predicted'])} |")
    Path(out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
