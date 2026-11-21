"""Data for the project page demo: one minute of a HoloAssist video with its hand state and who speaks when.

    python -m scripts.demo_data            # -> site/media/demo.mp4, site/media/demo.json

Steps
  1. Pick the video and minute by a fixed rule: a HoloAssist test video that ProAssist's released
     outputs cover; a 60 s window with at least three screw/unscrew/insert actions, at least five
     routine instructor onsets, and ProAssist speech in it; ranked by onsets outside fine work.
  2. Train the hand-state head (as in paper section 5) on every recording session except the
     demo video's own session, and predict the demo video. So the curve on the page is what the
     model says about video it never saw.
  3. Cut the minute from the video (the HoloAssist mp4 files carry no sound) and write the timeline,
     with the instructor's transcribed words and ProAssist's generated words, as JSON.

HoloAssist is released under CDLA-Permissive-2.0 (holoassist.github.io); the page credits it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.egoextra_from_video import FINE, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head
from scripts.holoassist_need_controlled import group_ids
from scripts.holoassist_student_speech import event_type
from scripts.holoassist_timing import LABELS, build_video
from scripts.proassist_baseline import ZIP
from src.paths import resolve_data_path

VIDEOS = "/mnt/ssd/holoassist/videos"
OUT = Path("site/media")
W = 60.0
GROUP_NAMES = ["idle", "fine", "hold", "other"]


def session(name: str) -> str:
    return name.split("-")[0]


def proassist_test() -> dict:
    pat = re.compile(r"eval/holoassist-dialog_test_L0_I1/stream/notalk0.3-maxlen_4k/results/\d+\.json")
    out = defaultdict(list)
    with zipfile.ZipFile(resolve_data_path(ZIP)) as z:
        for n in sorted(z.namelist()):
            if pat.match(n):
                d = json.loads(z.read(n))
                out[d["metadata"]["video_uid"]].append(
                    [(float(p["timestamp_in_stream"]), str(p["gen"]).strip()) for p in d["predictions"] if p["gen"]])
    return out


def pick(data, top, gid, pa):
    best = None
    vdir = Path(resolve_data_path(VIDEOS))
    for name, runs in pa.items():
        v = data.get(name)
        if v is None or not (vdir / f"{name}.mp4").exists():
            continue
        V = build_video(v, top)
        if V is None:
            continue
        on = np.array(sorted(e["start"] for e in v["events"] if e["label"] == "Conversation"
                             and event_type(e["attributes"].get("Conversation Purpose", "")) == "instructor starts, routine"))
        run_text = sorted(runs, key=len)[len(runs) // 2]      # the ProAssist run with the middle number of turns
        run = np.array([t for t, _ in run_text], float)
        g = gid[V["state"]]
        for a in np.arange(V["lo"], V["hi"] - W, 5.0):
            m = (V["edges"][:-1] >= a) & (V["edges"][1:] <= a + W)
            n_fine = int((m & (g == 1)).sum())
            o = on[(on >= a) & (on < a + W)]
            p = run[(run >= a) & (run < a + W)]
            if n_fine < 3 or len(o) < 5 or len(p) < 2:
                continue
            k = np.clip(np.searchsorted(V["edges"], o, side="right") - 1, 0, len(g) - 1)
            score = len(o) - 3 * int((g[k] == 1).sum()) + min(len(p), 6) * 0.5
            if best is None or score > best[0]:
                best = (score, name, float(a), V, on, run_text)
    return best[1:]


def held_out_curve(name: str, a: float, device: str):
    """Train the hand-state head without the demo video's session; -> ([[t, p_fine]] in the window, train ticks)."""
    X, st, _, names, vids = holoassist_ticks(2.0)
    train = np.array([session(n) for n in vids]) != session(name)
    head = fit_head(X[train], st[train], len(names), device)
    test = vids == name
    _, _, _, _, _, flags = holoassist_ticks(2.0, with_types=True)
    t_end = flags["_t_end"][test]
    P = predict_head(head, X[test], device)
    p_fine = P[:, [i for i, n in enumerate(names) if n in FINE]].sum(1)
    keep = (t_end >= a) & (t_end <= a + W)
    return [[round(float(t) - a, 2), round(float(p), 3)] for t, p in zip(t_end[keep], p_fine[keep])], int(train.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--retrain", action="store_true", help="train the held-out head again even if demo.json exists")
    args = ap.parse_args()
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    gid = group_ids(top)
    name, a, V, on, run_text = pick(data, top, gid, proassist_test())
    print("demo:", name, "from", round(a, 1), "s")
    old = json.loads((OUT / "demo.json").read_text()) if (OUT / "demo.json").exists() else {}
    if old.get("video") == name and old.get("start_s") == round(a, 2) and not args.retrain:
        p_rows, train_n = old["p_fine"], old["train_ticks"]
    else:
        p_rows, train_n = held_out_curve(name, a, args.device)
    v = data[name]
    utt = sorted((e["start"], e["end"], e["attributes"].get("Transcription", "")) for e in v["events"]
                 if e["label"] == "Conversation"
                 and event_type(e["attributes"].get("Conversation Purpose", "")) == "instructor starts, routine")

    segs = []
    for k in range(len(V["state"])):
        s, e = V["edges"][k], V["edges"][k + 1]
        if e <= a or s >= a + W:
            continue
        segs.append([round(max(s, a) - a, 2), round(min(e, a + W) - a, 2), GROUP_NAMES[gid[V["state"][k]]]])
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"video": name, "start_s": round(a, 2), "duration_s": W,
            "held_out_session": session(name), "train_ticks": train_n,
            "segments": segs,
            "p_fine": p_rows,
            "instructor": [round(float(t) - a, 2) for t in on if a <= t < a + W],
            "proassist": [round(t - a, 2) for t, _ in run_text if a <= t < a + W],
            "instructor_text": [[round(s - a, 2), round(e - a, 2), x] for s, e, x in utt if a <= s < a + W],
            "proassist_text": [[round(t - a, 2), x] for t, x in run_text if a <= t < a + W],
            "credit": "Video: HoloAssist (Wang et al., ICCV 2023), CDLA-Permissive-2.0. "
                      "ProAssist speech times: released evaluation outputs of ProAssist (Zhang et al., 2025)."}
    (OUT / "demo.json").write_text(json.dumps(meta, indent=1))
    src = Path(resolve_data_path(VIDEOS)) / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", str(src), "-t", f"{W}",
                    "-vf", "scale=896:-2", "-c:v", "libx264", "-crf", "29", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", str(OUT / "demo.mp4")], check=True)
    print("wrote", OUT / "demo.json", OUT / "demo.mp4", f"({(OUT / 'demo.mp4').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
