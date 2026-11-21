"""Timing audit: does an assistant start speaking in the same hand states as human instructors?

Give it the times at which any proactive assistant starts to speak on HoloAssist videos. It applies
the test of the paper (section 4.1) to those times and to the routine speech of the real instructors
on the same videos and the same time windows, and reports both side by side.

Input: a CSV with one row per speech onset of the assistant:

    video,t[,end]
    R090-28July-ATV,12.5
    R090-28July-ATV,31.0,240.0

  video  HoloAssist video name (the "video_name" field of the annotation file)
  t      onset time in seconds of video time
  end    optional: the last second of the video that the assistant saw. Use it when the assistant
         ran on only part of a video. The largest value per video is used. Without it, the whole
         labelled task window counts.

Every video that appears in the CSV counts, also when the assistant never spoke on it: add one row
with an empty t (e.g. "R0027-12-GoPro,,300") for such a video, or it is left out.

What it computes, for the assistant and for the instructors:
  - the share of onsets in each of four hand states (idle; screw, unscrew or insert; hold, place or
    inspect; other action), observed over chance. Chance = all onsets of a video moved by one random
    offset inside the window (circular shift), --shifts times.
  - 95% intervals from resampling videos (--boots times).
  - for each hand state, the paired difference assistant minus instructor, with its interval.

A ratio below 1 means the speaker starts less often in that hand state than chance. The instructors'
routine speech is at about 0.34 in screw, unscrew or insert actions over all videos (paper, 4.2).

Usage:
    python -m scripts.timing_audit --speech my_assistant.csv --name "My assistant" --out reports/audit_mine
    python -m scripts.timing_audit --self-test         # sanity checks on the real labels

Output: <out>.json and <out>.md. Needs the HoloAssist annotation file
(data-annotation-trainval-v1_1.json, see docs/data_access.md); no video is read.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.holoassist_need_controlled import GROUPS, group_ids
from scripts.holoassist_student_speech import event_type
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

STATE_NAMES = {"idle": "idle", "fine": "screw, unscrew, insert", "hold/place/inspect": "hold, place, inspect",
               "other action": "other action"}
MIN_WINDOW = 10.0


def load_labels():
    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    return {v["video_name"]: v for v in data}, top, group_ids(top)


def instructor_onsets(v: dict) -> np.ndarray:
    return np.array([e["start"] for e in v["events"] if e["label"] == "Conversation"
                     and event_type(e["attributes"].get("Conversation Purpose", "")) == "instructor starts, routine"], float)


def counts(V: dict, gid: np.ndarray, t: np.ndarray, lo: float, hi: float, off: np.ndarray):
    """-> observed (4,), mean circular-shift (4,) onset counts per hand state inside [lo, hi)."""
    t = t[(t >= lo) & (t < hi)]
    if len(t) == 0:
        return np.zeros(4), np.zeros(4)
    last = len(V["state"]) - 1
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, last)
    obs = np.bincount(gid[V["state"][k]], minlength=4).astype(float)
    ts = lo + (t[None, :] - lo + off[:, None] * (hi - lo)) % (hi - lo)
    g = gid[V["state"][np.clip(np.searchsorted(V["edges"], ts, side="right") - 1, 0, last)]]
    return obs, np.array([(g == j).sum(1).mean() for j in range(4)])


def read_speech(path: Path) -> tuple[dict, dict]:
    """-> {video: onset times}, {video: end or None}"""
    times, ends = defaultdict(list), {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video") or "").strip()
            if not vid:
                continue
            times[vid]
            if (row.get("t") or "").strip():
                times[vid].append(float(row["t"]))
            if (row.get("end") or "").strip():
                ends[vid] = max(ends.get(vid, 0.0), float(row["end"]))
    return {k: np.array(sorted(v)) for k, v in times.items()}, ends


def audit(speech: dict, ends: dict, data: dict, top, gid, shifts=1000, boots=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    per, skipped = {}, Counter()
    for vid, t in speech.items():
        v = data.get(vid)
        if v is None:
            skipped["not a HoloAssist video with labels"] += 1
            continue
        V = build_video(v, top)
        if V is None:
            skipped["too few labels or no instructor speech"] += 1
            continue
        hi = min(V["hi"], ends.get(vid, V["hi"]))
        if hi - V["lo"] < MIN_WINDOW:
            skipped[f"window shorter than {MIN_WINDOW:g} s"] += 1
            continue
        a = counts(V, gid, t, V["lo"], hi, rng.uniform(0, 1, shifts))
        h = counts(V, gid, instructor_onsets(v), V["lo"], hi, rng.uniform(0, 1, shifts))
        per[vid] = (a, h)
    if not per:
        raise SystemExit("no usable video: check the video names against the annotation file")
    vids = list(per)
    O = {w: np.array([per[u][i][0] for u in vids]) for i, w in enumerate(("assistant", "instructor"))}
    E = {w: np.array([per[u][i][1] for u in vids]) for i, w in enumerate(("assistant", "instructor"))}
    idx = rng.integers(0, len(vids), size=(boots, len(vids)))
    out = {"videos": len(vids), "skipped": dict(skipped), "shifts": shifts, "boots": boots, "states": {}}
    with np.errstate(divide="ignore", invalid="ignore"):
        Rb = {w: O[w][idx].sum(1) / E[w][idx].sum(1) for w in O}
        for j, g in enumerate(GROUPS):
            row = {}
            for w in O:
                pt = O[w][:, j].sum() / E[w][:, j].sum()
                lo, hi = np.nanpercentile(Rb[w][:, j], [2.5, 97.5])
                row[w] = [round(float(pt), 3), round(float(lo), 3), round(float(hi), 3)]
            d = Rb["assistant"][:, j] - Rb["instructor"][:, j]
            row["assistant minus instructor"] = [round(row["assistant"][0] - row["instructor"][0], 3),
                                                 *[round(float(x), 3) for x in np.nanpercentile(d, [2.5, 97.5])]]
            out["states"][g] = row
    out["onsets"] = {w: int(O[w].sum()) for w in O}
    return out


def report(res: dict, name: str) -> str:
    f = lambda r: f"{r[0]:.2f} ({r[1]:.2f} to {r[2]:.2f})"
    fd = lambda r: f"{r[0]:+.2f} ({r[1]:+.2f} to {r[2]:+.2f})"
    L = [f"# Timing audit: {name}", "",
         f"Generated by `python -m scripts.timing_audit`. {res['videos']} HoloAssist videos; "
         f"{res['onsets']['assistant']} onsets of {name}, {res['onsets']['instructor']} routine onsets of the human "
         f"instructors in the same windows. Ratio = onsets in the hand state, observed over circular-shift chance "
         f"({res['shifts']} shifts). 95% intervals: {res['boots']} resamples of videos.", "",
         f"| hand state | {name} | human instructors | difference |", "|---|---|---|---|"]
    for g, r in res["states"].items():
        L.append(f"| {STATE_NAMES[g]} | {f(r['assistant'])} | {f(r['instructor'])} | {fd(r['assistant minus instructor'])} |")
    if res["skipped"]:
        L += ["", "Videos left out: " + ", ".join(f"{k}: {v}" for k, v in res["skipped"].items()) + "."]
    L += ["", "Reading: a ratio below 1 means fewer onsets in that hand state than chance. A difference whose "
          "interval excludes 0 means the assistant's timing differs from the instructors' there. This compares "
          "timing only; it does not show which timing users prefer (paper, section 7)."]
    return "\n".join(L) + "\n"


def self_test(data, top, gid) -> None:
    """Two checks on real labels: the instructors against themselves (difference 0), and onsets at
    random times (ratio near 1 in every hand state)."""
    rng = np.random.default_rng(1)
    vids = [n for n, v in data.items() if build_video(v, top) is not None][:300]
    same = {n: instructor_onsets(data[n]) for n in vids}
    r1 = audit(same, {}, data, top, gid, shifts=200, boots=500)
    rand = {}
    for n in vids:
        V = build_video(data[n], top)
        rand[n] = np.sort(rng.uniform(V["lo"], V["hi"], size=20))
    r2 = audit(rand, {}, data, top, gid, shifts=200, boots=500)
    ok1 = all(abs(r["assistant"][0] - r["instructor"][0]) < 0.08 for r in r1["states"].values())
    ok2 = all(r["assistant"][1] <= 1.0 <= r["assistant"][2] or abs(r["assistant"][0] - 1) < 0.05
              for r in r2["states"].values())
    print("instructors vs themselves:", {g: r["assistant"][0] for g, r in r1["states"].items()}, "->", "PASS" if ok1 else "FAIL")
    print("random onsets:", {g: r["assistant"] for g, r in r2["states"].items()}, "->", "PASS" if ok2 else "FAIL")
    if not (ok1 and ok2):
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speech", type=Path)
    ap.add_argument("--name", default="assistant")
    ap.add_argument("--out", type=Path, default=Path("reports/timing_audit"))
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    data, top, gid = load_labels()
    if args.self_test:
        self_test(data, top, gid)
        return
    if args.speech is None:
        ap.error("--speech is required (or use --self-test)")
    speech, ends = read_speech(args.speech)
    res = audit(speech, ends, data, top, gid, args.shifts, args.boots)
    res["name"] = args.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    text = report(res, args.name)
    args.out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
