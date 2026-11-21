"""Baseline: does a published proactive assistant (ProAssist) avoid fine-motor actions?

ProAssist (arXiv 2506.05904) is a video-language model that watches an
egocentric stream at 2 frames per second and decides at every frame
whether to speak. Its authors released their evaluation outputs with the
checkpoints (Hugging Face 594zyc/ProAssist-Model-L4096-I1, file eval.zip).
For HoloAssist the outputs hold, per video and frame, what the model said
(field "gen") and the reference turn of ProAssist's own dialogue data
("ref"). That reference dialogue was written by a language model from the
HoloAssist annotations; it is not the real instructor speech.

So no model needs to run here. We read when ProAssist spoke and apply the
test of paper section 4 (scripts.holoassist_student_speech): the share of
speech onsets inside each hand state, observed over expected by chance
(circular shift inside the task window).

Three kinds of onset on the same videos and the same time window:
    ProAssist speaks       a frame where "gen" is not empty
    ProAssist reference    a frame where "ref" is not empty (its training target)
    real instructor        routine instructor onsets from the HoloAssist labels

Time: the stream time equals the video time. The reference turns sit
closest to the labelled action ends at zero lag (checked 2026-11-21:
57% within 0.5 s at lag 0, at most 46% at any other lag in 0.5 s steps).

Each HoloAssist video has three ProAssist samples (three generated
dialogues). Each sample is its own stream with its own shifts. The window
of a sample is [first action start, min(last action end, stream end)];
instructor onsets use the longest window of the video. The interval comes
from a bootstrap over videos.

Usage:
    python -m scripts.proassist_baseline
Input:  /mnt/ssd/proassist/eval_L4096_I1.zip (download from the URL above)
Outputs: reports/proassist_baseline.json, reports/proassist_baseline.md
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.holoassist_need_controlled import GROUPS, group_ids
from scripts.holoassist_student_speech import event_type
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

ZIP = "/mnt/ssd/proassist/eval_L4096_I1.zip"
KINDS = ["ProAssist speaks", "ProAssist reference", "real instructor, routine"]


def counts(V: dict, gid: np.ndarray, t: np.ndarray, lo: float, hi: float, off: np.ndarray):
    """-> observed (4,), mean shifted (4,) onset counts per hand-state group inside [lo, hi)."""
    t = t[(t >= lo) & (t < hi)]
    if len(t) == 0:
        return np.zeros(4), np.zeros(4)
    last = len(V["state"]) - 1
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, last)
    obs = np.bincount(gid[V["state"][k]], minlength=4).astype(float)
    ts = lo + (t[None, :] - lo + off[:, None] * (hi - lo)) % (hi - lo)
    ks = np.clip(np.searchsorted(V["edges"], ts, side="right") - 1, 0, last)
    g = gid[V["state"][ks]]
    exp = np.array([(g == j).sum(1).mean() for j in range(4)])
    return obs, exp


def run_split(z, data, top, gid, split, thr, shifts, rng):
    pat = re.compile(rf"eval/holoassist-dialog_{split}_L0_I1/stream/notalk{thr}-maxlen_4k/results/\d+\.json")
    samples = defaultdict(list)
    for name in z.namelist():
        if pat.match(name):
            d = json.loads(z.read(name))
            samples[d["metadata"]["video_uid"]].append(d["predictions"])
    per_video = {}            # video -> kind -> (obs, exp)
    n_on = Counter()
    for uid, preds_list in samples.items():
        v = data.get(uid)
        V = build_video(v, top) if v is not None else None
        if V is None:
            continue
        acc = {k: [np.zeros(4), np.zeros(4)] for k in KINDS}
        ends = []
        for preds in preds_list:
            end = float(preds[-1]["timestamp_in_stream"])
            hi = min(V["hi"], end)
            ends.append(hi)
            if hi - V["lo"] < 10:
                continue
            off = rng.uniform(0, 1, size=shifts)
            for kind, field in (("ProAssist speaks", "gen"), ("ProAssist reference", "ref")):
                t = np.array([p["timestamp_in_stream"] for p in preds if p[field]], dtype=float)
                o, e = counts(V, gid, t, V["lo"], hi, off)
                acc[kind][0] += o; acc[kind][1] += e
                n_on[kind] += int(o.sum())
        hi = max(ends)
        if hi - V["lo"] >= 10:
            t = np.array([e["start"] for e in v["events"] if e["label"] == "Conversation"
                          and event_type(e["attributes"].get("Conversation Purpose", "")) == "instructor starts, routine"],
                         dtype=float)
            o, e = counts(V, gid, t, V["lo"], hi, rng.uniform(0, 1, size=shifts))
            acc["real instructor, routine"] = [o, e]
            n_on["real instructor, routine"] += int(o.sum())
        per_video[uid] = acc
    return per_video, n_on


def summarise(per_video: dict, n_on: Counter, boots: int, rng) -> dict:
    vids = list(per_video)
    out = {}
    for kind in KINDS:
        O = np.array([per_video[u][kind][0] for u in vids])      # (videos, 4)
        E = np.array([per_video[u][kind][1] for u in vids])
        ratio = O.sum(0) / E.sum(0)
        idx = rng.integers(0, len(vids), size=(boots, len(vids)))
        R = O[idx].sum(1) / E[idx].sum(1)                        # (boots, 4)
        lo, hi = np.percentile(R, [2.5, 97.5], axis=0)
        out[kind] = {"n_onsets": n_on[kind],
                     **{g: [round(float(ratio[j]), 3), round(float(lo[j]), 3), round(float(hi[j]), 3)]
                        for j, g in enumerate(GROUPS)}}
    # paired difference on the fine group: ProAssist speaks minus real instructor
    j = GROUPS.index("fine")
    O = {k: np.array([per_video[u][k][0][j] for u in vids]) for k in KINDS}
    E = {k: np.array([per_video[u][k][1][j] for u in vids]) for k in KINDS}
    idx = rng.integers(0, len(vids), size=(boots, len(vids)))
    d = (O[KINDS[0]][idx].sum(1) / E[KINDS[0]][idx].sum(1)) - (O[KINDS[2]][idx].sum(1) / E[KINDS[2]][idx].sum(1))
    out["fine: ProAssist minus instructor"] = [round(float(O[KINDS[0]].sum() / E[KINDS[0]].sum()
                                                         - O[KINDS[2]].sum() / E[KINDS[2]].sum()), 3),
                                               *[round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]]
    out["videos"] = len(vids)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--threshold", default="0.3", help="ProAssist not-talk threshold its authors chose on val")
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--boots", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    cnt = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                  if e["label"] == "Fine grained action")
    top = [v for v, _ in cnt.most_common(20)]
    gid = group_ids(top)
    z = zipfile.ZipFile(resolve_data_path(args.zip))
    res = {"source": "huggingface.co/594zyc/ProAssist-Model-L4096-I1 eval.zip", "threshold": args.threshold,
           "shifts": args.shifts, "boots": args.boots}
    for split in ("test", "val"):
        pv, n_on = run_split(z, data, top, gid, split, args.threshold, args.shifts, rng)
        res[split] = summarise(pv, n_on, args.boots, rng)
    Path("reports/proassist_baseline.json").write_text(json.dumps(res, indent=2))

    L = ["# Baseline: when does ProAssist speak, relative to the hands?", "",
         "Generated by `python -m scripts.proassist_baseline`. ProAssist's own released evaluation outputs "
         f"(model L4096-I1, not-talk threshold {args.threshold}) on HoloAssist, against the HoloAssist hand "
         "labels. Each cell: onsets in this hand state, observed / expected by chance (circular shift), "
         f"[95% bootstrap interval over videos, {args.boots} draws]. 'ProAssist reference' is ProAssist's "
         "generated training dialogue, not real speech.", ""]
    for split in ("test", "val"):
        r = res[split]
        L += [f"## {split} split ({r['videos']} videos)", "",
              "| onsets | n | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * 4]
        for k in KINDS:
            c = r[k]
            L.append(f"| {k} | {c['n_onsets']} | " + " | ".join(f"{c[g][0]:.2f} [{c[g][1]:.2f}, {c[g][2]:.2f}]"
                                                              for g in GROUPS) + " |")
        dd = r["fine: ProAssist minus instructor"]
        L += ["", f"Fine column, ProAssist minus real instructor: {dd[0]:+.2f} [{dd[1]:+.2f}, {dd[2]:+.2f}].", ""]
    Path("reports/proassist_baseline.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
