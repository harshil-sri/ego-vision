"""What follows when an instructor speaks during a screw, unscrew or insert action? (HoloAssist, labels only)

Section 4.4 of the paper shows that routine instructor speech avoids these actions. This script asks what
happens in the cases where it does not. For every routine instructor onset (next-step instruction,
follow-up, confirmation) we look up the hand state at the onset and measure three outcomes:

  end_soon   the action that covers the onset ends within 1.5 s after it (the hands stop)
  resumed    that action is taken up again: an action with the same verb and noun starts within
             10 s after it ends (the action was broken off and continued)
  mistake    a wrong action (any label other than "Correct Action") starts within 15 s after the onset

The chance level is the same circular shift as in the rest of the paper: all routine onsets of a video
are moved by one random offset inside the task window, so a shifted onset lands in an action in
proportion to its length. Ratio = (share of onsets in a context with the outcome) / (the same share for
shifted onsets in that context). 95% interval: 2,000 resamples of videos.

The key number is the ratio for fine-motor actions over the ratio for other actions: does speech during
screw/unscrew/insert go with more disruption than speech during other hand actions?

This is an observation. An instructor may speak because the performer already paused (the pause causes
the speech, not the other way round). A higher end_soon ratio is therefore expected under both readings;
resumed and mistake are less exposed to it, and the report says so.

Usage:
    python -m scripts.holoassist_speech_cost [--reps 500] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_need_controlled import FINE
from scripts.holoassist_timing import GROUPS, LABELS, build_video
from src.paths import resolve_data_path

ROUTINE = {GROUPS["follow_up"], GROUPS["high_level"], GROUPS["confirm"]}
CTX = ["fine", "other action", "idle"]
OUTCOMES = ["end_soon", "resumed", "mistake"]
END_SOON, RESUME_S, MISTAKE_S = 1.5, 10.0, 15.0


def parse(v: dict, top: list[str]):
    """-> per-video arrays, or None when build_video would skip the video."""
    if build_video(v, top) is None:
        return None
    acts = {}
    for e in v["events"]:
        if e["label"] == "Fine grained action" and e["end"] > e["start"]:
            a = e["attributes"]
            acts[(round(e["start"], 3), round(e["end"], 3), a.get("Verb"))] = (
                e["start"], e["end"], a.get("Verb"), a.get("Noun"), a.get("Action Correctness", "Correct Action"))
    acts = sorted(acts.values())
    lo, hi = acts[0][0], max(a[1] for a in acts)
    st = np.array([a[0] for a in acts]); en = np.array([a[1] for a in acts])
    fine = np.array([a[2] in FINE for a in acts])
    resumed = np.zeros(len(acts), bool)
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            if b[0] > a[1] + RESUME_S:
                break
            if b[0] >= a[1] and b[2] == a[2] and b[3] == a[3]:
                resumed[i] = True
                break
    corr = [a[4] for a in acts]
    wrong = {"any": np.sort(st[[c != "Correct Action" for c in corr]]),
             "self": np.sort(st[[c == "Wrong Action, corrected by student" for c in corr]])}
    bad = np.array([c != "Correct Action" for c in corr])
    # piecewise map time -> covering action (later starts overwrite earlier), -1 = idle
    edges = np.unique(np.r_[lo, hi, st, en])
    cover = np.full(len(edges) - 1, -1)
    for i in range(len(acts)):
        cover[np.searchsorted(edges, st[i]):np.searchsorted(edges, en[i])] = i
    conv = sorted((e["start"], e["attributes"].get("Conversation Purpose")) for e in v["events"]
                  if e["label"] == "Conversation" and lo <= e["start"] < hi)
    on = {g: np.array([t for t, p in conv if p == GROUPS[g]]) for g in ("follow_up", "high_level", "confirm")}
    on["routine"] = np.array([t for t, p in conv if p in ROUTINE])
    if len(on["routine"]) == 0:
        return None
    return dict(lo=lo, hi=hi, st=st, en=en, fine=fine, resumed=resumed, wrong=wrong, bad=bad, edges=edges,
                cover=cover, on=on)


def tally(V: dict, T: np.ndarray, mistake: str = "any", skip_bad: bool = False) -> np.ndarray:
    """T (..., n) onset times -> counts (3 contexts, 1 + 3 outcomes) summed over the last axis.
    Column 0 = onsets in the context, then the number with each outcome (nan where undefined)."""
    k = np.clip(np.searchsorted(V["edges"], T, side="right") - 1, 0, len(V["cover"]) - 1)
    a = V["cover"][k]
    ai = np.where(a >= 0, a, 0)
    keep = ~(skip_bad & (a >= 0) & V["bad"][ai])       # optionally drop onsets inside a wrong action
    act = (a >= 0) & keep
    is_fine = act & V["fine"][ai]
    end_soon = act & (V["en"][ai] - T <= END_SOON)
    res = act & V["resumed"][ai]
    w = V["wrong"][mistake]
    mis = (np.searchsorted(w, T + MISTAKE_S, side="right") - np.searchsorted(w, T, side="right")) > 0
    out = np.zeros(T.shape[:-1] + (3, 4))
    for c, m in enumerate((is_fine, act & ~is_fine, (a < 0) & keep)):
        out[..., c, 0] = m.sum(-1)
        out[..., c, 1] = (m & end_soon).sum(-1)
        out[..., c, 2] = (m & res).sum(-1)
        out[..., c, 3] = (m & mis).sum(-1)
    return out


#: (name, onset group, mistake kind, drop onsets inside a wrong action)
VARIANTS = [("main", "routine", "any", False),
            ("mistake = self-corrected only", "routine", "self", False),
            ("onsets inside a wrong action dropped", "routine", "any", True),
            ("next-step instructions only", "high_level", "any", False),
            ("follow-ups only", "follow_up", "any", False),
            ("confirmations only", "confirm", "any", False)]


def ratios(O, N):
    """(videos, 3, 4) observed and null counts -> (3 contexts, 3 outcomes) observed share over null share."""
    o, n = O.sum(0), N.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (o[:, 1:] / o[:, :1]) / (n[:, 1:] / n[:, :1])


def run(Vs, group, mistake, skip_bad, reps, boot, rng) -> dict:
    obs, null = [], []
    for V in Vs:
        on = V["on"][group]
        if len(on) == 0:
            continue
        obs.append(tally(V, on[None, :], mistake, skip_bad)[0])
        u = rng.uniform(0, V["hi"] - V["lo"], size=(reps, 1))
        T = V["lo"] + np.mod(on[None, :] - V["lo"] + u, V["hi"] - V["lo"])
        null.append(tally(V, T, mistake, skip_bad).mean(0))
    obs, null = np.array(obs), np.array(null)
    point = ratios(obs, null)
    B = np.array([ratios(obs[i], null[i]) for i in (rng.integers(0, len(obs), len(obs)) for _ in range(boot))])
    lo_, hi_ = np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)
    diff = np.log(B[:, 0]) - np.log(B[:, 1])               # fine over other action, per outcome
    dpt = point[0] / point[1]
    o, n = obs.sum(0), null.sum(0)
    r3 = lambda a, b, c: [round(float(a), 3), round(float(b), 3), round(float(c), 3)]
    res = {"videos": len(obs), "onsets": int(o[:, 0].sum()), "contexts": {}}
    for c, name in enumerate(CTX):
        res["contexts"][name] = {"onsets": int(o[c, 0]), "null_onsets": round(float(n[c, 0]), 1)}
        for j, oc in enumerate(OUTCOMES):
            if name == "idle" and oc != "mistake":
                continue
            res["contexts"][name][oc] = {"observed_share": round(float(o[c, j + 1] / o[c, 0]), 4),
                                         "null_share": round(float(n[c, j + 1] / n[c, 0]), 4),
                                         "ratio": r3(point[c, j], lo_[c, j], hi_[c, j])}
    res["fine_over_other"] = {oc: r3(dpt[j], np.exp(np.nanpercentile(diff[:, j], 2.5)),
                                     np.exp(np.nanpercentile(diff[:, j], 97.5))) for j, oc in enumerate(OUTCOMES)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    Vs = [V for V in (parse(v, top) for v in data) if V is not None]
    out = {"reps": args.reps, "boot": args.boot,
           "windows_s": {"end_soon": END_SOON, "resumed": RESUME_S, "mistake": MISTAKE_S}, "variants": {}}
    for name, group, mistake, skip in VARIANTS:
        out["variants"][name] = run(Vs, group, mistake, skip, args.reps, args.boot, np.random.default_rng(0))
    Path("reports/holoassist_speech_cost.json").write_text(json.dumps(out, indent=2))

    f = lambda r: f"{r[0]:.2f} ({r[1]:.2f} to {r[2]:.2f})"
    m = out["variants"]["main"]
    L = ["# HoloAssist: what follows routine instructor speech during fine-motor work?", "",
         f"Generated by `python -m scripts.holoassist_speech_cost`. Labels only. {m['videos']} videos, "
         f"{m['onsets']} routine instructor onsets (next-step, follow-up, confirmation). "
         f"Chance level: per-video circular shift of the onsets, {args.reps} repetitions. 95% interval: "
         f"{args.boot} resamples of videos. See the script docstring for the definitions and the main caveat "
         "(a pause of the performer can cause the speech).", "",
         f"- end_soon: the action under the onset ends within {END_SOON} s",
         f"- resumed: the same verb and noun start again within {RESUME_S:g} s after that action ends",
         f"- mistake: a wrong action starts within {MISTAKE_S:g} s after the onset", "",
         "## Main analysis", "",
         "| hands at onset | onsets (chance) | outcome | observed | chance | ratio |", "|---|---|---|---|---|---|"]
    for name, r in m["contexts"].items():
        for oc in OUTCOMES:
            if oc in r:
                L.append(f"| {name} | {r['onsets']} ({r['null_onsets']:.0f}) | {oc} | {r[oc]['observed_share']:.3f} | "
                         f"{r[oc]['null_share']:.3f} | {f(r[oc]['ratio'])} |")
    L += ["", "## Robustness: fine-motor ratio, and fine-motor over other actions", "",
          "| variant | onsets in fine | mistake: fine | mistake: fine / other | end_soon: fine / other | resumed: fine / other |",
          "|---|---|---|---|---|---|"]
    for name, r in out["variants"].items():
        L.append(f"| {name} | {r['contexts']['fine']['onsets']} | {f(r['contexts']['fine']['mistake']['ratio'])} | "
                 f"{f(r['fine_over_other']['mistake'])} | {f(r['fine_over_other']['end_soon'])} | "
                 f"{f(r['fine_over_other']['resumed'])} |")
    text = "\n".join(L) + "\n"
    Path("reports/holoassist_speech_cost.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
