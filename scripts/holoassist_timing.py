"""Pilot study: do human instructors time their speech around the hands?

Data: HoloAssist labels only (no video). A performer does a manual task
wearing a headset; an instructor watches the first-person view live and
speaks. Annotators marked
    - every instructor utterance (start, end, purpose), and
    - every fine-grained hand action of the performer (start, end, verb).

Question: at the moment an instructor STARTS speaking, what are the
performer's hands doing, compared with what they would be doing at a
random moment of the same video?

Null model (per video, circular shift): move ALL instructor onsets of a
video by one random offset, wrapping around inside the task window
[first action start, last action end]. This keeps the number of
utterances and the gaps between them, and destroys only their alignment
with the hand actions. Repeating it gives the distribution of each
statistic "if timing ignored the hands".

Outputs:
    reports/holoassist_timing.json   every number
    reports/holoassist_timing.md     readable tables

Usage:
    python -m scripts.holoassist_timing --perms 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.paths import resolve_data_path

LABELS = "/mnt/ssd/holoassist/data-annotation-trainval-v1_1.json"

#: instructor-initiated purposes analysed as separate groups
GROUPS = {
    "follow_up": "instructor-start-conversation_follow-up instruction",
    "high_level": "instructor-start-conversation_describing high-level instruction",
    "confirm": "instructor-start-conversation_confirming the previous action",
    "correct": "instructor-start-conversation_correct the wrong action",
}
GROUP_NAMES = list(GROUPS) + ["all_instructor"]
STAT_NAMES = ["idle", "after_end_0.5s", "near_boundary_0.5s", "near_coarse_1s",
              "phase_early", "phase_late"]
WRONG_BY_INSTRUCTOR = "Wrong Action, corrected by instructor verbally"


def build_video(v: dict, top_verbs: list[str]):
    """Turn one video's events into lookup arrays.

    Output dict (None if the video has no usable actions/utterances):
        lo, hi: task window in seconds
        edges (K+1,), state (K,): piecewise-constant hand state over time;
            state[k] is a verb id for edges[k] <= t < edges[k+1], where id
            len(top_verbs) means "other verb" and len(top_verbs)+1 "idle"
        a_start, a_end (K,): start/end of the action covering that piece
            (nan when idle)
        ends, bounds, coarse: sorted boundary times
        onsets: {group: (n,) seconds}
        wrong: list of (start, end, verb) for instructor-corrected mistakes
        corr: (m,) onset times of 'correct the wrong action' utterances
    """
    fine = {}
    for e in v["events"]:
        if e["label"] == "Fine grained action" and e["end"] > e["start"]:
            key = (round(e["start"], 3), round(e["end"], 3), e["attributes"].get("Verb"))
            fine[key] = e                      # drop exact duplicate annotations
    if len(fine) < 5:
        return None
    acts = sorted(fine.values(), key=lambda e: e["start"])
    lo, hi = acts[0]["start"], max(e["end"] for e in acts)
    if sum(e["end"] - e["start"] for e in acts) > 1.05 * (hi - lo):
        return None                            # heavily overlapping labels: skip

    vid = {v_: i for i, v_ in enumerate(top_verbs)}
    OTHER, IDLE = len(top_verbs), len(top_verbs) + 1
    edges = np.unique(np.array([lo, hi] + [e["start"] for e in acts] + [e["end"] for e in acts]))
    K = len(edges) - 1
    state = np.full(K, IDLE, dtype=np.int64)
    a_start = np.full(K, np.nan); a_end = np.full(K, np.nan)
    for e in acts:                             # later starts overwrite earlier
        i0 = np.searchsorted(edges, e["start"]); i1 = np.searchsorted(edges, e["end"])
        state[i0:i1] = vid.get(e["attributes"].get("Verb"), OTHER)
        a_start[i0:i1] = e["start"]; a_end[i0:i1] = e["end"]

    onsets = defaultdict(list)
    for e in v["events"]:
        if e["label"] != "Conversation":
            continue
        p = e["attributes"].get("Conversation Purpose", "")
        if not (lo <= e["start"] < hi):
            continue
        for g, name in GROUPS.items():
            if p == name:
                onsets[g].append(e["start"])
        if p.startswith("instructor-start-conversation"):
            onsets["all_instructor"].append(e["start"])
    if not onsets["all_instructor"]:
        return None
    coarse = sorted({t for e in v["events"] if e["label"] == "Coarse grained action"
                     for t in (e["start"], e["end"])})
    return {
        "lo": lo, "hi": hi, "edges": edges, "state": state,
        "a_start": a_start, "a_end": a_end,
        "ends": np.array(sorted(e["end"] for e in acts)),
        "bounds": np.array(sorted([e["start"] for e in acts] + [e["end"] for e in acts])),
        "coarse": np.array(coarse) if coarse else np.array([np.inf]),
        "onsets": {g: np.array(onsets[g]) for g in GROUP_NAMES},
        "wrong": [(e["start"], e["end"], e["attributes"].get("Verb")) for e in acts
                  if e["attributes"].get("Action Correctness") == WRONG_BY_INSTRUCTOR],
        "corr": np.array(sorted(onsets["correct"])),
        "state_seconds": np.bincount(state, weights=np.diff(edges), minlength=IDLE + 1),
    }


def nearest_dist(sorted_times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """|t - nearest element of sorted_times|, elementwise. t any shape."""
    i = np.clip(np.searchsorted(sorted_times, t), 1, len(sorted_times) - 1) if len(sorted_times) > 1 \
        else np.zeros_like(t, dtype=np.int64)
    if len(sorted_times) == 1:
        return np.abs(t - sorted_times[0])
    return np.minimum(np.abs(t - sorted_times[i - 1]), np.abs(t - sorted_times[i]))


def features(V: dict, t: np.ndarray):
    """Hand-state features at times t (any shape, inside [lo, hi)).

    Returns (state id, indicator stats (..., 4), phase in [0,1] or nan).
    """
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
    st = V["state"][k]
    idle = st == V["state"].max() if False else np.isnan(V["a_start"][k])
    j = np.searchsorted(V["ends"], t, side="right") - 1            # last end <= t
    since_end = np.where(j >= 0, t - V["ends"][np.clip(j, 0, None)], np.inf)
    ind = np.stack([idle,
                    since_end <= 0.5,
                    nearest_dist(V["bounds"], t) <= 0.5,
                    nearest_dist(V["coarse"], t) <= 1.0], axis=-1)
    phase = (t - V["a_start"][k]) / (V["a_end"][k] - V["a_start"][k])   # nan if idle
    return st, ind, phase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--top-verbs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = json.loads(Path(resolve_data_path(args.labels)).read_text(encoding="utf-8"))
    verb_counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                          if e["label"] == "Fine grained action")
    top = [v for v, _ in verb_counts.most_common(args.top_verbs)]
    state_names = top + ["other_verb", "idle"]
    S, G, P = len(state_names), len(GROUP_NAMES), args.perms

    videos = [V for V in (build_video(v, top) for v in data) if V is not None]
    print(f"{len(videos)}/{len(data)} videos usable")

    # accumulators: index 0 = observed, 1..P = shifted copies
    st_cnt = np.zeros((P + 1, G, S)); ind_sum = np.zeros((P + 1, G, 4))
    ph_n = np.zeros((P + 1, G)); ph_early = np.zeros((P + 1, G)); ph_late = np.zeros((P + 1, G))
    n_on = np.zeros(G); state_seconds = np.zeros(S)
    for V in videos:
        W = V["hi"] - V["lo"]
        state_seconds += V["state_seconds"]
        off = np.concatenate([[0.0], rng.uniform(0, W, P)])                 # (P+1,)
        for g, name in enumerate(GROUP_NAMES):
            t = V["onsets"][name]
            if len(t) == 0:
                continue
            n_on[g] += len(t)
            tt = V["lo"] + np.mod(t[None, :] - V["lo"] + off[:, None], W)   # (P+1, n)
            st, ind, phase = features(V, tt)
            for s in range(S):
                st_cnt[:, g, s] += (st == s).sum(axis=1)
            ind_sum[:, g] += ind.sum(axis=1)
            ok = ~np.isnan(phase)
            ph_n[:, g] += ok.sum(axis=1)
            ph_early[:, g] += (ok & (phase < 0.2)).sum(axis=1)
            ph_late[:, g] += (ok & (phase > 0.8)).sum(axis=1)

    def summarise(x: np.ndarray) -> dict:
        """x (P+1,): observed at 0, null after. Two-sided permutation p."""
        obs, null = float(x[0]), x[1:]
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (len(null) + 1)
        return {"observed": round(obs, 4), "null_mean": round(float(null.mean()), 4),
                "null_lo": round(float(np.percentile(null, 2.5)), 4),
                "null_hi": round(float(np.percentile(null, 97.5)), 4),
                "ratio": round(obs / null.mean(), 3) if null.mean() > 0 else None,
                "p": round(float(p), 5)}

    out = {"videos": len(videos), "perms": P, "onsets": dict(zip(GROUP_NAMES, n_on.astype(int).tolist())),
           "hours": round(sum(V["hi"] - V["lo"] for V in videos) / 3600, 1),
           "state_time_share": {n: round(float(s / state_seconds.sum()), 4)
                                for n, s in zip(state_names, state_seconds)},
           "stats": {}, "states": {}}
    for g, name in enumerate(GROUP_NAMES):
        if n_on[g] == 0:
            continue
        rows = {STAT_NAMES[i]: summarise(ind_sum[:, g, i] / n_on[g]) for i in range(4)}
        rows["phase_early"] = summarise(ph_early[:, g] / np.maximum(ph_n[:, g], 1))
        rows["phase_late"] = summarise(ph_late[:, g] / np.maximum(ph_n[:, g], 1))
        out["stats"][name] = rows
        out["states"][name] = {state_names[s]: summarise(st_cnt[:, g, s] / n_on[g])
                               for s in range(S)}

    # Corrections: how long after a wrong action starts does the instructor speak,
    # and does the correction arrive before that action has finished?
    lat = defaultdict(list)
    for V in videos:
        for (s, e, verb) in V["wrong"]:
            c = V["corr"][(V["corr"] >= s) & (V["corr"] <= e + 15.0)]
            if len(c):
                lat[verb if verb in top else "other_verb"].append((c[0] - s, c[0] - e, e - s))
    allv = np.array([x for xs in lat.values() for x in xs])
    corr = {"matched_wrong_actions": int(len(allv)),
            "wrong_actions_total": int(sum(len(V["wrong"]) for V in videos)),
            "all": {"before_action_end": round(float((allv[:, 1] < 0).mean()), 4),
                    "median_latency_from_start_s": round(float(np.median(allv[:, 0])), 2),
                    "median_delay_after_end_s": round(float(np.median(allv[:, 1])), 2),
                    "median_action_dur_s": round(float(np.median(allv[:, 2])), 2)},
            "by_verb": {}}
    for verb, xs in sorted(lat.items(), key=lambda kv: -len(kv[1])):
        a = np.array(xs)
        if len(a) >= 40:
            corr["by_verb"][verb] = {
                "n": int(len(a)), "before_action_end": round(float((a[:, 1] < 0).mean()), 3),
                "median_latency_from_start_s": round(float(np.median(a[:, 0])), 2),
                "median_delay_after_end_s": round(float(np.median(a[:, 1])), 2),
                "median_action_dur_s": round(float(np.median(a[:, 2])), 2)}
    out["corrections"] = corr

    Path("reports").mkdir(exist_ok=True)
    Path("reports/holoassist_timing.json").write_text(json.dumps(out, indent=2))

    L = ["# HoloAssist pilot: instructor speech onsets vs performer hand state", "",
         f"Generated by `python -m scripts.holoassist_timing --perms {P}` from the HoloAssist "
         f"train/val labels. {len(videos)} of {len(data)} videos used "
         f"({out['hours']} h inside task windows). Null = per-video circular shift of all "
         f"instructor onsets, {P} repetitions; interval = central 95% of the null.", "",
         "Onsets analysed: " + ", ".join(f"{k} {v}" for k, v in out["onsets"].items()), ""]
    for name in GROUP_NAMES:
        if name not in out["stats"]:
            continue
        L += [f"## {name}", "", "| statistic | observed | null mean | null 95% | ratio | p |", "|---|---|---|---|---|---|"]
        for k, r in out["stats"][name].items():
            L.append(f"| {k} | {r['observed']} | {r['null_mean']} | {r['null_lo']} to {r['null_hi']} | {r['ratio']} | {r['p']} |")
        L += ["", "Hand state at onset (share of onsets), most over- and under-represented first:", "",
              "| state | observed | null mean | null 95% | ratio | p |", "|---|---|---|---|---|---|"]
        rows = sorted(out["states"][name].items(), key=lambda kv: -(kv[1]["ratio"] or 0))
        for k, r in rows:
            L.append(f"| {k} | {r['observed']} | {r['null_mean']} | {r['null_lo']} to {r['null_hi']} | {r['ratio']} | {r['p']} |")
        L.append("")
    c = corr
    L += ["## Corrections of wrong actions", "",
          f"{c['matched_wrong_actions']} of {c['wrong_actions_total']} instructor-corrected wrong actions "
          f"have a 'correct the wrong action' utterance starting between the action's start and 15 s after its end.", "",
          f"All: {c['all']}", "",
          "| wrong-action verb | n | correction starts before action ends | median latency from action start (s) | median delay after action end (s) | median action duration (s) |",
          "|---|---|---|---|---|---|"]
    for verb, r in c["by_verb"].items():
        L.append(f"| {verb} | {r['n']} | {r['before_action_end']} | {r['median_latency_from_start_s']} | {r['median_delay_after_end_s']} | {r['median_action_dur_s']} |")
    Path("reports/holoassist_timing.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
