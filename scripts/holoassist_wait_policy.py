"""Pilot part 8: a stay-silent score for mistake corrections (HoloAssist, labels only).

Part 4 showed that instructors leave 43% of wrong actions to the performer.
This script turns that into a score for an assistant and asks which simple
policy does well on it.

Setting: the assistant knows that a wrong action has started (a perfect
mistake detector; the detector is not what is measured). It decides IF and
WHEN to speak. Two errors:

    unneeded   it speaks on a mistake that the performer fixed alone
               (the human instructor said nothing)
    failed     on a mistake that the instructor corrected, it does not
               speak, or speaks later than the instructor did

Policies
    wait d          speak d seconds after the wrong action starts, if the
                    wrong action is still under way
    gate            speak at once if a model says "the instructor would
                    correct this", otherwise never. Model: part 4, all
                    features known when the mistake starts, out-of-fold
                    (5 folds split by recording session)
    gate + wait d   speak at once if the model is sure; otherwise wait d
                    seconds and speak if the wrong action is still under way

Quantities from the labels
    D   duration of a self-corrected wrong action. "wait d" speaks on it
        when D > d. Assumption: when the wrong action ends, the performer
        has noticed.
    L   seconds from the start of an instructor-corrected wrong action to
        the instructor's correction. "wait d" fails when d > L. Mistakes
        without a correction utterance within 10 s of the action's end
        are left out (count in the report).

Each family is summarised by its lowest unneeded rate at a failed rate of
at most 10%, 20%, 30%. A last table allows the assistant to be up to 2 s
later than the instructor. Intervals: sessions resampled, the best setting
chosen again inside every resample.

Usage:
    python -m scripts.holoassist_wait_policy
Outputs: reports/holoassist_wait_policy.json, reports/holoassist_wait_policy.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.holoassist_silence import feature_sets, mistake_rows
from scripts.holoassist_timing import LABELS
from src.paths import resolve_data_path

DELAYS = np.r_[np.arange(0.0, 10.01, 0.25), np.inf]          # inf = never speak after the gate
THRESH = np.linspace(0.0, 1.0, 101)                          # speak at once if p(instructor corrects) > 1 - thr
TARGETS = (0.10, 0.20, 0.30)
FAMILIES = ("wait d", "gate", "gate + wait d")


def rates(p_corr: np.ndarray, silent: np.ndarray, D: np.ndarray, L: np.ndarray,
          gate_delay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (unneeded, failed), each (len(THRESH), len(DELAYS)).

    p_corr: model probability that the instructor corrects. THRESH[i] is the
    share of confidence needed: the gate speaks at once when p_corr > 1 - THRESH[i]
    (THRESH 0 = the gate never fires, 1 = it always fires).
    gate_delay: seconds after the mistake starts at which the gate can decide
    (a video model must first see the mistake). A gate that fires speaks then,
    if the wrong action is still under way; it fails if the instructor was faster.
    """
    s, c = silent == 1, silent == 0
    now_s = (p_corr[s][None, :] > 1.0 - THRESH[:, None]) & (D[s] > gate_delay)[None, :]   # (T, n_self)
    now_c = p_corr[c][None, :] > 1.0 - THRESH[:, None]
    gate_late = now_c & (L[c] < gate_delay)[None, :]
    later_s = D[s][None, :] > DELAYS[:, None]                                # (K, n_self): still going at d
    late_c = L[c][None, :] < DELAYS[:, None]                                 # (K, n_corr): instructor was faster
    unneeded = (now_s[:, None, :] | later_s[None, :, :]).mean(2)
    failed = ((~now_c[:, None, :] & late_c[None, :, :]) | gate_late[:, None, :]).mean(2)
    return unneeded, failed


def best(unneeded: np.ndarray, failed: np.ndarray, family: str, target: float) -> tuple[float, float, float]:
    """Lowest unneeded rate with failed <= target -> (unneeded, threshold, delay)."""
    ok = failed <= target
    if family == "wait d":
        ok[1:, :] = False                                    # gate never fires
        ok[:, -1] = False
    elif family == "gate":
        ok[:, :-1] = False                                   # never speak after the gate
    u = np.where(ok, unneeded, np.inf)
    i, k = np.unravel_index(np.argmin(u), u.shape)
    return float(u[i, k]), float(THRESH[i]), float(DELAYS[k])


def main_video(args) -> None:
    """The gate without labels: history, video, history + video; labels + history as the ceiling.

    Video feature: the first 2 s window that ends at least --gate-delay after the
    mistake starts. So a video gate can speak at --gate-delay at the earliest;
    the two gates without video decide at 0 s. "drop" mistakes are removed.
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    fd = Path(resolve_data_path("/mnt/ssd/holoassist")) / "features_w2.0"
    meta, F = pd.read_csv(fd / "meta.csv"), np.load(fd / "feats.npy", mmap_mode="r")
    first = {n: (g.index[0], g["t_end"].to_numpy()) for n, g in meta.groupby("video_name", sort=False)}
    rows_all = mistake_rows(json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8")))
    sets_all = feature_sets(rows_all)[0]
    keep, frow = [], []
    for i, r in enumerate(rows_all):
        if r["video"] not in first or (r["silent"] == 0 and not np.isfinite(r["corr"])) or r["verb"] == "drop":
            continue
        i0, t = first[r["video"]]
        k = np.searchsorted(t, r["start"] + args.gate_delay, side="left")
        if k < len(t) and t[k] <= r["start"] + args.gate_delay + 1.0:
            keep.append(i); frow.append(i0 + k)
    keep = np.array(keep)
    rows = [rows_all[i] for i in keep]
    Fv = np.asarray(F[np.array(frow)], dtype=np.float32)
    Fv = (Fv - Fv.mean(0)) / (Fv.std(0) + 1e-6)
    H = sets_all["history only"][0][keep]
    Xlab, cat = sets_all["all"][0][keep], sets_all["all"][1]
    silent = np.array([r["silent"] for r in rows]); groups = np.array([r["session"] for r in rows])
    D = np.array([r["end"] - r["start"] for r in rows]); L = np.array([r["corr"] - r["start"] for r in rows])
    gates = {"history": 0.0, "video": args.gate_delay, "history + video": args.gate_delay, "labels + history (ceiling)": 0.0}
    P = {g: np.zeros(len(rows)) for g in gates}
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(H, silent, groups):
        Z = PCA(min(32, len(tr) - 1), random_state=0).fit(Fv[tr]).transform(Fv)
        for g, (X, c) in {"history": (H, None), "video": (Z, None), "history + video": (np.c_[H, Z], None),
                          "labels + history (ceiling)": (Xlab, cat)}.items():
            m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, categorical_features=c,
                                               max_depth=4, random_state=0).fit(X[tr], silent[tr])
            P[g][te] = m.predict_proba(X[te])[:, 0]
    rng = np.random.default_rng(0)
    by = {x: np.flatnonzero(groups == x) for x in np.unique(groups)}
    keys = list(by)
    boots = [np.concatenate([by[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(args.boot)]
    res = {"videos": int(len({r["video"] for r in rows})), "sessions": len(keys), "boot": args.boot,
           "gate_delay_s": args.gate_delay, "self_corrected": int((silent == 1).sum()),
           "instructor_corrected": int((silent == 0).sum()), "gates": {}}
    U0, F0 = rates(P["history"], silent, D, L)
    res["wait d"] = {f"{t:.0%}": round(best(U0, F0, "wait d", t)[0], 3) for t in TARGETS}
    base = {t: [best(*rates(P["history"][i], silent[i], D[i], L[i]), "gate", t)[0] for i in boots] for t in TARGETS}
    for g, delay in gates.items():
        U, Fm = rates(P[g], silent, D, L, delay)
        r = {"auc_instructor_corrects": round(float(roc_auc_score(1 - silent, P[g])), 4), "gate_decides_at_s": delay}
        for t in TARGETS:
            v = np.array([best(*rates(P[g][i], silent[i], D[i], L[i], delay), "gate", t)[0] for i in boots])
            d = v - np.array(base[t])
            d = d[np.isfinite(d)]
            r[f"{t:.0%}"] = {"unneeded": best(U, Fm, "gate", t)[0],
                             "interval": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                             "minus_history_gate": [float(d.mean()), float(np.percentile(d, 2.5)),
                                                    float(np.percentile(d, 97.5))] if len(d) else None}
        res["gates"][g] = r
    Path("reports/holoassist_wait_policy_video.json").write_text(
        json.dumps(res, indent=2, default=float).replace("Infinity", "null"))
    L_ = ["# HoloAssist pilot part 8 from video: the stay-silent gate without labels", "",
          "Generated by `python -m scripts.holoassist_wait_policy --video`. "
          f"{res['videos']} videos, {res['sessions']} sessions, \"drop\" removed: {res['self_corrected']} self-corrected and "
          f"{res['instructor_corrected']} instructor-corrected wrong actions. Gate only (speak if the model expects a "
          f"correction, otherwise never). A gate with video decides {args.gate_delay:g} s after the mistake starts; "
          "it fails when the instructor was faster, and a target it cannot reach shows as inf. "
          "Lowest unneeded rate at a failed rate of at most 10% / 20% / 30% "
          f"[95% interval, {args.boot} resamples of sessions]. "
          "Waiting alone on the same mistakes: " + " / ".join(f"{v:.2f}" for v in res["wait d"].values()) + ".", "",
          "| gate input | ROC AUC | 10% | 20% | 30% | change against the history gate at 30% |", "|---|---|---|---|---|---|"]
    for g, r in res["gates"].items():
        c = r["30%"]["minus_history_gate"]
        L_.append(f"| {g} | {r['auc_instructor_corrects']:.3f} | "
                  + " | ".join(f"{r[k]['unneeded']:.2f} [{r[k]['interval'][0]:.2f}, {r[k]['interval'][1]:.2f}]"
                               for k in ("10%", "20%", "30%"))
                  + (f" | {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] |" if c else " | n/a |"))
    Path("reports/holoassist_wait_policy_video.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    print("\n".join(L_))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--video", action="store_true",
                    help="gate from VideoMAE features instead of labels (only mistakes in videos with features)")
    ap.add_argument("--gate-delay", type=float, default=1.0, help="--video: the gate decides this long after the start")
    args = ap.parse_args()
    if args.video:
        return main_video(args)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows = mistake_rows(json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8")))
    X, cat = feature_sets(rows)[0]["all"]
    silent = np.array([r["silent"] for r in rows])
    groups = np.array([r["session"] for r in rows])
    D = np.array([r["end"] - r["start"] for r in rows])
    L = np.array([r["corr"] - r["start"] for r in rows])
    drop = np.array([r["verb"] == "drop" for r in rows])
    p_corr = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, silent, groups):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, categorical_features=cat,
                                           max_depth=4, random_state=0).fit(X[tr], silent[tr])
        p_corr[te] = m.predict_proba(X[te])[:, 0]
    usable = (silent == 1) | np.isfinite(L)                  # instructor-corrected rows need a correction time
    rng = np.random.default_rng(0)
    q = [10, 25, 50, 75, 90]

    def analyse(keep: np.ndarray, slack: float = 0.0) -> dict:
        k = keep & usable
        p, s, d, l, g = p_corr[k], silent[k], D[k], L[k] + slack, groups[k]
        U, F = rates(p, s, d, l)
        by = {x: np.flatnonzero(g == x) for x in np.unique(g)}
        keys = list(by)
        bs = {(f, t): [] for f in FAMILIES for t in TARGETS}
        for _ in range(args.boot):
            idx = np.concatenate([by[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
            Ub, Fb = rates(p[idx], s[idx], d[idx], l[idx])
            for f in FAMILIES:
                for t in TARGETS:
                    bs[(f, t)].append(best(Ub, Fb, f, t)[0])
        out = {"self_corrected": int((s == 1).sum()), "instructor_corrected": int((s == 0).sum()),
               "instructor_corrected_without_correction_time": int((keep & ~usable).sum()),
               "D_percentiles_s": dict(zip(map(str, q), np.percentile(d[s == 1], q).round(2).tolist())),
               "L_percentiles_s": dict(zip(map(str, q), np.percentile(l[s == 0] - slack, q).round(2).tolist())),
               "wait_curve": {f"{x:g}": {"unneeded": round(float(U[0, i]), 3), "failed": round(float(F[0, i]), 3)}
                              for i, x in enumerate(DELAYS) if x in (0, 0.5, 1, 2, 3, 5, 8)},
               "best": {}}
        for f in FAMILIES:
            out["best"][f] = {}
            for t in TARGETS:
                u, th, de = best(U, F, f, t)
                v = np.array(bs[(f, t)])
                out["best"][f][f"{t:.0%}"] = {"unneeded": round(u, 3),
                                              "interval": [round(float(np.percentile(v, 2.5)), 3),
                                                           round(float(np.percentile(v, 97.5)), 3)],
                                              "gate_fires_above_p": round(1.0 - th, 2),
                                              "delay_s": None if np.isinf(de) else de}
        return out

    res = {"sessions": int(len(set(groups))), "boot": args.boot,
           "all": analyse(np.ones(len(rows), bool)), "without_drop": analyse(~drop),
           "without_drop_slack_2s": analyse(~drop, 2.0)}
    Path("reports/holoassist_wait_policy.json").write_text(json.dumps(res, indent=2))

    L_ = ["# HoloAssist pilot part 8: a stay-silent score for mistake corrections", "",
          "Generated by `python -m scripts.holoassist_wait_policy`. Labels only. The assistant is told when a wrong "
          "action starts and decides if and when to speak. **unneeded**: it speaks on a mistake the performer fixed "
          "alone. **failed**: on a mistake the instructor corrected, it stays silent or speaks later than the "
          f"instructor. Intervals: {args.boot} resamples of sessions. See the script docstring.", ""]
    for key, title in (("all", "All wrong actions"), ("without_drop", "Without \"drop\" (99% self-corrected, easy)"),
                       ("without_drop_slack_2s", "Without \"drop\", and up to 2 s later than the instructor still counts as in time")):
        r = res[key]
        L_ += [f"## {title}", "",
               f"{r['self_corrected']} self-corrected, {r['instructor_corrected']} instructor-corrected with a "
               f"correction time ({r['instructor_corrected_without_correction_time']} without one, left out). "
               "Seconds, 10th / 25th / 50th / 75th / 90th percentile: "
               f"duration of a self-corrected wrong action {' / '.join(str(x) for x in r['D_percentiles_s'].values())}; "
               f"instructor's delay before correcting {' / '.join(str(x) for x in r['L_percentiles_s'].values())}.", "",
               "Waiting alone:", "", "| wait (s) | " + " | ".join(r["wait_curve"]) + " |",
               "|---|" + "---|" * len(r["wait_curve"]),
               "| unneeded | " + " | ".join(f"{c['unneeded']:.2f}" for c in r["wait_curve"].values()) + " |",
               "| failed | " + " | ".join(f"{c['failed']:.2f}" for c in r["wait_curve"].values()) + " |", "",
               "Lowest unneeded rate at a failed rate of at most:", "",
               "| policy | " + " | ".join(f"{t:.0%}" for t in TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
        for f in FAMILIES:
            cells = []
            for t in TARGETS:
                c = r["best"][f][f"{t:.0%}"]
                how = []
                if f != "wait d":
                    how.append(f"gate p > {c['gate_fires_above_p']}")
                if f != "gate" and c["delay_s"] is not None:
                    how.append(f"wait {c['delay_s']:g} s")
                cells.append(f"{c['unneeded']:.2f} [{c['interval'][0]:.2f}, {c['interval'][1]:.2f}] ({', '.join(how) or 'never'})")
            L_.append(f"| {f} | " + " | ".join(cells) + " |")
        L_.append("")
    Path("reports/holoassist_wait_policy.md").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
