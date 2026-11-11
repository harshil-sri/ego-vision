"""Pilot part 2: does hand state help PREDICT when a human instructor speaks?

Setup: slice every HoloAssist task window into 0.5 s ticks. At each tick,
using only information from the past, predict whether the instructor will
start a routine utterance (follow-up instruction, high-level instruction,
confirmation) within the next 0.5 s.

Two feature sets, same model (gradient-boosted trees), 5-fold split BY
VIDEO so no video is in both train and test:
    task   time since the instructor last spoke, time since the last coarse
           step boundary, time since task start, fraction of the video
           elapsed. A proxy for "is help due now".
    +hand  task features plus: current hand state (verb id or idle), time
           in that state, time since the last fine action ended, number of
           action ends in the last 5 s.

The hand state comes from human annotations, so this is an upper bound on
what a perfect hand-state recogniser would add. It shows association, not
cause: instructors may speak when hands are free because that is when help
is needed, or because that is when the performer can listen.

Also reports, per wrong-action verb, how often the instructor corrected it
versus stayed silent while the performer fixed it alone.

Usage:
    python -m scripts.holoassist_predict
Outputs: reports/holoassist_predict.json, reports/holoassist_predict.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.holoassist_timing import GROUPS, LABELS, build_video
from src.paths import resolve_data_path

TICK = 0.5
ROUTINE = ("follow_up", "high_level", "confirm")


def tick_features(V: dict, v_raw: dict, t: np.ndarray | None = None, horizon: float = TICK):
    """Per-tick features and target for one video.

    Returns (X_task (n,4), X_hand (n,4), y (n,)), all aligned on ticks
    t = lo, lo+0.5, ... < hi - 0.5, or on the given times t (seconds,
    inside the task window). y = a routine onset starts in (t, t + horizon].
    """
    if t is None:
        t = np.arange(V["lo"], V["hi"] - TICK, TICK)                    # (n,)
    if len(t) < 4:
        return None
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
    state = V["state"][k]
    # time in current state: distance back to the last edge where the state changed
    change = np.flatnonzero(np.diff(V["state"]) != 0) + 1               # piece indices
    change_t = np.concatenate([[V["lo"]], V["edges"][change]])
    in_state = t - change_t[np.searchsorted(change_t, t, side="right") - 1]
    j = np.searchsorted(V["ends"], t, side="right")                     # ends <= t
    since_end = np.where(j > 0, t - V["ends"][np.clip(j - 1, 0, None)], 60.0)
    ends_5s = j - np.searchsorted(V["ends"], t - 5.0, side="right")

    utt = sorted((e["start"], e["end"]) for e in v_raw["events"]
                 if e["label"] == "Conversation"
                 and e["attributes"].get("Conversation Purpose", "").startswith("instructor"))
    u_end = np.array(sorted(b for _, b in utt)) if utt else np.array([-1e9])
    ju = np.searchsorted(u_end, t, side="right")
    since_utt = np.where(ju > 0, t - u_end[np.clip(ju - 1, 0, None)], 120.0)
    coarse = V["coarse"][np.isfinite(V["coarse"])]
    if len(coarse):
        jc = np.searchsorted(coarse, t, side="right")
        since_coarse = np.where(jc > 0, t - coarse[np.clip(jc - 1, 0, None)], 120.0)
    else:
        since_coarse = np.full_like(t, 120.0)

    on = np.sort(np.concatenate([V["onsets"][g] for g in ROUTINE]))
    y = (np.searchsorted(on, t + horizon, side="right") - np.searchsorted(on, t, side="right")) > 0

    X_task = np.stack([np.clip(since_utt, 0, 120), np.clip(since_coarse, 0, 120),
                       t - V["lo"], (t - V["lo"]) / (V["hi"] - V["lo"])], axis=1)
    X_hand = np.stack([state, np.clip(in_state, 0, 60), np.clip(since_end, 0, 60), ends_5s], axis=1)
    return X_task, X_hand, y.astype(np.int8)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]

    XT, XH, Y, Gp = [], [], [], []
    for i, v in enumerate(data):
        V = build_video(v, top)
        if V is None:
            continue
        f = tick_features(V, v)
        if f is None:
            continue
        XT.append(f[0]); XH.append(f[1]); Y.append(f[2]); Gp.append(np.full(len(f[2]), i))
    XT, XH, Y, Gp = map(np.concatenate, (XT, XH, Y, Gp))
    print(f"ticks {len(Y)}  positives {int(Y.sum())}  base rate {Y.mean():.4f}  videos {len(set(Gp))}")

    sets = {"task": XT, "hand_only": XH, "task+hand": np.concatenate([XT, XH], axis=1)}
    cat = {"task": None, "hand_only": [0], "task+hand": [4]}
    res = {k: {"auc": [], "ap": []} for k in sets}
    for tr, te in GroupKFold(n_splits=5).split(XT, Y, Gp):
        for name, X in sets.items():
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                               categorical_features=cat[name], random_state=0)
            m.fit(X[tr], Y[tr])
            p = m.predict_proba(X[te])[:, 1]
            res[name]["auc"].append(roc_auc_score(Y[te], p))
            res[name]["ap"].append(average_precision_score(Y[te], p))
    out = {"ticks": int(len(Y)), "positives": int(Y.sum()), "base_rate": round(float(Y.mean()), 4),
           "videos": int(len(set(Gp))), "tick_seconds": TICK, "models": {}}
    for name, r in res.items():
        out["models"][name] = {"auc_mean": round(float(np.mean(r["auc"])), 4),
                               "auc_folds": [round(float(x), 4) for x in r["auc"]],
                               "ap_mean": round(float(np.mean(r["ap"])), 4),
                               "ap_folds": [round(float(x), 4) for x in r["ap"]]}

    # Which mistakes do instructors correct, and which do they leave alone?
    by = defaultdict(lambda: Counter())
    for v in data:
        for e in v["events"]:
            if e["label"] != "Fine grained action":
                continue
            c = e["attributes"].get("Action Correctness", "")
            if c.startswith("Wrong Action"):
                verb = e["attributes"].get("Verb")
                by[verb if verb in top else "other_verb"][c] += 1
                by["ALL"][c] += 1
    wrong = {}
    for verb, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n >= 60:
            wrong[verb] = {"n": n,
                           "instructor_corrected": round(c["Wrong Action, corrected by instructor verbally"] / n, 3),
                           "self_corrected": round(c["Wrong Action, corrected by student"] / n, 3),
                           "not_corrected": round(c["Wrong Action, not corrected"] / n, 3)}
    out["wrong_actions"] = wrong

    Path("reports/holoassist_predict.json").write_text(json.dumps(out, indent=2))
    L = ["# HoloAssist pilot part 2: predicting instructor speech onsets", "",
         "Generated by `python -m scripts.holoassist_predict`. "
         f"{out['ticks']} ticks of {TICK} s from {out['videos']} videos; "
         f"{out['positives']} ticks ({out['base_rate']:.2%}) are followed by a routine instructor "
         "onset within 0.5 s. 5-fold cross-validation grouped by video. "
         "AP = average precision (chance = base rate).", "",
         "| features | ROC AUC (mean of 5 folds) | AP (mean) | AP per fold |", "|---|---|---|---|"]
    for name, r in out["models"].items():
        L.append(f"| {name} | {r['auc_mean']} | {r['ap_mean']} | {r['ap_folds']} |")
    L += ["", "## Wrong actions: corrected by the instructor, or left to the performer", "",
          "| verb of the wrong action | n | instructor corrected | performer self-corrected | not corrected |",
          "|---|---|---|---|---|"]
    for verb, r in wrong.items():
        L.append(f"| {verb} | {r['n']} | {r['instructor_corrected']} | {r['self_corrected']} | {r['not_corrected']} |")
    Path("reports/holoassist_predict.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
