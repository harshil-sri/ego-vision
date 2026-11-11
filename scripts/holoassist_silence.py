"""Pilot part 4: when a mistake happens, does the instructor speak or stay silent?

HoloAssist marks every wrong hand action with who fixed it:
    "corrected by instructor verbally"   the instructor spoke
    "corrected by student"               the performer fixed it alone;
                                         the instructor did not correct it
    "not corrected"                      rare, left out here

Question: can the instructor's choice be predicted from information that
is available WHILE the mistake is under way? If yes, "stay silent on this
kind of mistake" is a learnable part of expert timing, and a benchmark
that rewards speaking on every mistake does not measure expert behaviour.

Features (all known at the moment the wrong action starts):
    verb        verb of the wrong action (categorical)
    noun        object of the wrong action (categorical, 40 most common)
    task        task type of the video, e.g. "setup gopro" (categorical)
    history     seconds since task start, fraction of the video elapsed,
                seconds since the instructor last spoke, number of wrong
                actions so far, number of instructor corrections so far
The duration of the wrong action is NOT used: a correction cuts the action
short, so duration is an effect of the label, not a cause.

Model: gradient-boosted trees, 5 folds split BY RECORDING SESSION (the
"R0027" part of the video name), so the same people never appear in both
train and test. Metrics: ROC AUC, and average precision for the class
"instructor stays silent" (chance = its base rate).

Usage:
    python -m scripts.holoassist_silence
Outputs: reports/holoassist_silence.json, reports/holoassist_silence.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_timing import LABELS
from src.paths import resolve_data_path

BY_INSTRUCTOR = "Wrong Action, corrected by instructor verbally"
BY_STUDENT = "Wrong Action, corrected by student"


CORRECTION = "instructor-start-conversation_correct the wrong action"


def mistake_rows(data: list[dict], max_after: float = 10.0) -> list[dict]:
    """One row per wrong action that was fixed (by the instructor or by the performer).

    Besides the features of part 4, each row has "start" and "end" of the
    wrong action (seconds in the video) and "corr": the onset of the first
    "correct the wrong action" utterance in [start, end + max_after], nan
    if there is none.
    """
    rows = []
    for v in data:
        ev = v["events"]
        fine = sorted((e for e in ev if e["label"] == "Fine grained action"), key=lambda e: e["start"])
        if not fine:
            continue
        t0, dur = fine[0]["start"], max(max(e["end"] for e in fine) - fine[0]["start"], 1.0)
        utt_end = np.array(sorted(e["end"] for e in ev if e["label"] == "Conversation"
                                  and e["attributes"].get("Conversation Purpose", "").startswith("instructor")))
        corr = np.array(sorted(e["start"] for e in ev if e["label"] == "Conversation"
                               and e["attributes"].get("Conversation Purpose", "") == CORRECTION))
        wrong_so_far = corrected_so_far = 0
        for e in fine:
            c = e["attributes"].get("Action Correctness", "")
            if c in (BY_INSTRUCTOR, BY_STUDENT):
                j = np.searchsorted(utt_end, e["start"], side="right")
                since_utt = e["start"] - utt_end[j - 1] if j > 0 else 300.0
                k = np.searchsorted(corr, e["start"], side="left")
                first = float(corr[k]) if k < len(corr) and corr[k] <= e["end"] + max_after else float("nan")
                rows.append({"session": v["video_name"].split("-")[0], "task": v.get("taskType", "?"),
                             "verb": e["attributes"].get("Verb", "?"), "noun": e["attributes"].get("Noun", "?"),
                             "t": e["start"] - t0, "frac": (e["start"] - t0) / dur,
                             "since_utt": min(since_utt, 300.0), "wrong_so_far": wrong_so_far,
                             "corrected_so_far": corrected_so_far, "silent": int(c == BY_STUDENT),
                             "start": e["start"], "end": e["end"], "corr": first})
                corrected_so_far += int(c == BY_INSTRUCTOR)
            if c.startswith("Wrong"):
                wrong_so_far += 1
    return rows


def feature_sets(rows: list[dict]) -> tuple[dict, list[str], list[str]]:
    """-> ({name: (X, categorical column indices or None)}, verb names, task names)"""
    def codes(key, top=None):
        cnt = Counter(r[key] for r in rows)
        keep = [k for k, _ in cnt.most_common(top)]
        idx = {k: i for i, k in enumerate(keep)}
        return np.array([idx.get(r[key], len(keep)) for r in rows], dtype=np.float64), keep

    verb, verb_names = codes("verb", 40)
    noun, _ = codes("noun", 40)
    task, task_names = codes("task")
    hist = np.array([[r["t"], r["frac"], r["since_utt"], r["wrong_so_far"], r["corrected_so_far"]]
                     for r in rows])
    return {
        "verb": (verb[:, None], [0]),
        "verb + noun": (np.stack([verb, noun], 1), [0, 1]),
        "verb + noun + task": (np.stack([verb, noun, task], 1), [0, 1, 2]),
        "history only": (hist, None),
        "all": (np.concatenate([np.stack([verb, noun, task], 1), hist], 1), [0, 1, 2]),
    }, verb_names, task_names


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    rows = mistake_rows(data)

    sets, _, _ = feature_sets(rows)
    y = np.array([r["silent"] for r in rows])
    groups = np.array([r["session"] for r in rows])
    print(f"{len(y)} wrong actions, {y.mean():.3f} left to the performer, {len(set(groups))} sessions")

    # "drop" is 16% of all wrong actions and 99% self-corrected (a dropped
    # part is obvious to the performer). It makes the task look easier than
    # it is, so every model is also scored with drops removed.
    not_drop = np.array([r["verb"] != "drop" for r in rows])
    res, res_nd = {}, {}
    for keep, target in ((np.ones(len(y), bool), res), (not_drop, res_nd)):
        for name, (X, cat) in sets.items():
            auc, ap = [], []
            Xk, yk, gk = X[keep], y[keep], groups[keep]
            for tr, te in GroupKFold(n_splits=5).split(Xk, yk, gk):
                m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, categorical_features=cat,
                                                   max_depth=4, random_state=0).fit(Xk[tr], yk[tr])
                p = m.predict_proba(Xk[te])[:, 1]
                auc.append(roc_auc_score(yk[te], p)); ap.append(average_precision_score(yk[te], p))
            target[name] = {"auc": round(float(np.mean(auc)), 4), "auc_folds": [round(float(a), 3) for a in auc],
                            "ap_silent": round(float(np.mean(ap)), 4)}
            print(name, target[name])

    def table(key, names_min=40):
        out = {}
        cnt = Counter(r[key] for r in rows)
        for k, n in cnt.most_common():
            if n >= names_min:
                out[k] = {"n": n, "silent_share": round(float(np.mean([r["silent"] for r in rows if r[key] == k])), 3)}
        return out

    out = {"wrong_actions": int(len(y)), "silent_share": round(float(y.mean()), 4),
           "sessions": int(len(set(groups))), "models": res,
           "without_drop": {"wrong_actions": int(not_drop.sum()),
                            "silent_share": round(float(y[not_drop].mean()), 4), "models": res_nd},
           "by_task": table("task"), "by_verb": table("verb", 60)}
    Path("reports/holoassist_silence.json").write_text(json.dumps(out, indent=2))
    L = ["# HoloAssist pilot part 4: which mistakes do instructors leave alone?", "",
         "Generated by `python -m scripts.holoassist_silence`. "
         f"{out['wrong_actions']} wrong actions from {out['sessions']} recording sessions; in "
         f"{out['silent_share']:.1%} the instructor did not correct and the performer fixed it alone. "
         "Prediction target: instructor stays silent. 5 folds split by session. "
         f"Chance: ROC AUC 0.5, average precision {out['silent_share']:.3f}.", "",
         "| features known when the mistake starts | ROC AUC | AUC per fold | average precision (silent) |",
         "|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {r['auc']} | {r['auc_folds']} | {r['ap_silent']} |")
    nd = out["without_drop"]
    L += ["", "## The same, with \"drop\" mistakes removed", "",
          f"\"drop\" is 99% self-corrected and easy to predict. Without it: {nd['wrong_actions']} wrong actions, "
          f"{nd['silent_share']:.1%} left to the performer (chance average precision {nd['silent_share']:.3f}).", "",
          "| features known when the mistake starts | ROC AUC | AUC per fold | average precision (silent) |",
          "|---|---|---|---|"]
    for name, r in res_nd.items():
        L.append(f"| {name} | {r['auc']} | {r['auc_folds']} | {r['ap_silent']} |")
    L += ["", "## Share left to the performer, by task", "", "| task | wrong actions | instructor silent |", "|---|---|---|"]
    L += [f"| {k} | {r['n']} | {r['silent_share']} |" for k, r in out["by_task"].items()]
    L += ["", "## Share left to the performer, by verb of the wrong action", "",
          "| verb | wrong actions | instructor silent |", "|---|---|---|"]
    L += [f"| {k} | {r['n']} | {r['silent_share']} |" for k, r in out["by_verb"].items()]
    Path("reports/holoassist_silence.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-60:]))


if __name__ == "__main__":
    main()
