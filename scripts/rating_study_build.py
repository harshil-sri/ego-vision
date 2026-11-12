"""Build the rating study: short clips with an assistant message at two moments.

Question of the study: do people judge that a ROUTINE message should wait
until fine-motor work ends, and that a WARNING should not wait?

Each clip shows a person in fine-motor work (screw, unscrew, insert; from the
HoloAssist labels) that ends inside the clip. A message is submitted during
the work. Two versions of every clip:
    now    the message is spoken at once, during the fine-motor work
    held   the message is spoken when the app's rule releases it
           (src/pipeline/delivery.py with the hand state from VIDEO, the
           cross-fitted predictions saved by scripts.holoassist_shift_control)
Message types:
    routine  the real next instruction of the human instructor in that video
    warning  a short correction ("Stop. That is the wrong part.")
A rater sees every clip once, in one version, and rates the timing 1 to 5.

Selection rules (all automatic, fixed seed): the message moment is inside
labelled fine-motor work that lasts 2 s before and 2.5 s after it; the rule
releases the message 2.5 to 7 s later because the hands became free (not by
the time limit), and the labels agree that the fine-motor work has ended;
nobody speaks in the clip; at most one clip per video and three per task.

Usage:
    python -m scripts.rating_study_build --list      # print candidates only
    python -m scripts.rating_study_build             # write study/clips/*.mp4 and study/manifest.json
Needs ffmpeg and Windows (the voice is made with System.Speech).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_need_controlled import FINE, state_at
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path
from src.pipeline.delivery import HandAwareDelivery

ROOT = "/mnt/ssd/holoassist"
WARNINGS = ["Stop. That is the wrong part.", "Wait. That is not the right way.",
            "Stop. Check that again.", "Wait. That one is wrong."]
PRE, POST = 3.0, 4.5                      # seconds of video before the message moment and after the release


def clean(text: str) -> str | None:
    """A real instruction, usable as the routine message, or None."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if "*" in t or not 5 <= len(t.split()) <= 14 or "?" in t:
        return None
    low = t.lower()
    if re.search(r"\b(um+|uh+|er+|hmm+)\b", low) or re.search(r"\b(\w+),? \1\b", low):   # fillers, repeated words
        return None
    if low.split()[0].strip(",") in ("and", "so", "yeah", "yes", "just", "okay", "ok", "but", "or", "then", "no",
                                     "it", "that", "this", "because", "like", "well", "oh"):
        return None                                         # a fragment that needs the sentence before it
    return t[0].upper() + t[1:] + ("" if t[-1] in ".!" else ".")


def candidates(seed: int = 0) -> list[dict]:
    root = Path(resolve_data_path(ROOT))
    z = np.load(root / "features_w2.0" / "oof_groups.npz", allow_pickle=True)
    vid, t_end, p = z["video"], z["t_end"], z["p_group"][:, 0].astype(float)
    data = {v["video_name"]: v for v in json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))}
    counts = Counter(e["attributes"].get("Verb") for v in data.values() for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    is_fine = np.array([v in FINE for v in top] + [False, False])
    cut = np.flatnonzero(vid[1:] != vid[:-1]) + 1
    rule = HandAwareDelivery()
    rng = np.random.default_rng(seed)
    out = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(vid)]):
        name = vid[a]
        v = data[name]
        V = build_video(v, top)
        if V is None or not (root / "videos" / f"{name}.mp4").exists():
            continue
        t, pf = t_end[a:b], p[a:b].copy()
        for i in range(1, len(pf)):
            pf[i] = 0.5 * pf[i] + 0.5 * pf[i - 1]                      # the app's smoothing
        talk = np.array([[e["start"], e["end"]] for e in v["events"] if e["label"] == "Conversation"])
        instr = sorted((e["start"], e["attributes"].get("Transcription", "")) for e in v["events"]
                       if e["label"] == "Conversation" and e["attributes"].get("Conversation Purpose", "").endswith(
                           ("describing high-level instruction", "follow-up instruction"))
                       and e["attributes"].get("Transcription Confidence") != "low-confidence-transcription")

        def fine(x):
            x = np.atleast_1d(x)
            return bool(np.all((x >= V["lo"]) & (x < V["hi"])) and np.all(is_fine[state_at(V, x)]))

        found = []
        for i in rng.permutation(len(t)):
            c = float(t[i])
            if pf[i] < rule.fine_threshold or not fine(np.arange(c - 2.0, c + 2.51, 0.5)):
                continue
            j = i + 1
            while j < len(t) and pf[j] >= rule.fine_threshold:
                j += 1
            if j >= len(t):
                continue
            r = float(t[j])
            if not (2.5 <= r - c <= 7.0 and r - c < rule.max_hold_s) or fine(np.arange(r, r + 2.01, 0.5)) \
                    or fine(r + 0.5):
                continue
            lo, hi = c - PRE, r + POST
            if lo < 1.0 or hi > t[-1] or (len(talk) and np.any((talk[:, 0] < hi) & (talk[:, 1] > lo))):
                continue
            nxt = next((clean(x) for s, x in instr if s > hi and clean(x)), None)
            verb = top[state_at(V, np.array([c]))[0]]
            found.append({"video": name, "task": v.get("taskType", "?"), "verb": verb, "c": round(c, 2),
                          "r": round(r, 2), "start": round(lo, 2), "end": round(hi, 2), "next_instruction": nxt})
            break
        out += found
    return out


def choose(cands: list[dict], n_routine: int, n_warning: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    order = [cands[i] for i in rng.permutation(len(cands))]
    per_task, items = Counter(), []
    for want, n in (("routine", n_routine), ("warning", n_warning)):
        k = 0
        for c in order:
            if k == n:
                break
            if c in items or per_task[c["task"]] >= 3 or (want == "routine" and not c["next_instruction"]):
                continue
            c = dict(c, type=want, text=c["next_instruction"] if want == "routine" else WARNINGS[k % len(WARNINGS)])
            items.append(c); per_task[c["task"]] += 1; k += 1
    for i, c in enumerate(items):
        c["id"] = f"c{i + 1:02d}"
    extra = next(c for c in order if c["next_instruction"] and all(c["video"] != x["video"] for x in items))
    items.append(dict(extra, type="practice", text=extra["next_instruction"], id="p01"))   # shown first, not scored
    return items


def speak(text: str, wav: Path) -> None:
    """Text -> wav with the Windows voice (offline)."""
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.Rate = 1; $s.SetOutputToWaveFile('{wav}'); $s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--routine", type=int, default=12)
    ap.add_argument("--warning", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study")
    args = ap.parse_args()
    cands = candidates(args.seed)
    print(f"{len(cands)} candidate clips, {sum(1 for c in cands if c['next_instruction'])} with a usable next instruction")
    items = choose(cands, args.routine, args.warning, args.seed)
    for c in items:
        print(f"  {c['id']} {c['type']:8s} {c['task']:32s} {c['verb']:8s} wait {c['r'] - c['c']:.1f} s  {c['text']}")
    if args.list:
        return
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    root = Path(resolve_data_path(ROOT))
    for c in items:
        wav = tmp / f"{c['id']}.wav"
        speak(c["text"], wav.resolve())
        dur = c["end"] - c["start"]
        c["duration"] = round(dur, 2)
        c["at"] = {"now": round(c["c"] - c["start"], 2), "held": round(c["r"] - c["start"], 2)}
        for cond, at in c["at"].items():
            f = out / "clips" / f"{c['id']}_{cond}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{c['start']}", "-t", f"{dur}",
                 "-i", str(root / "videos" / f"{c['video']}.mp4"), "-i", str(wav),
                 "-filter_complex", f"[1:a]adelay={int(at * 1000)}:all=1,apad[a];[0:v]scale=640:-2[v]",
                 "-map", "[v]", "-map", "[a]", "-t", f"{dur}", "-c:v", "libx264", "-crf", "27", "-preset", "slow",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(f)], check=True)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    keep = ("id", "type", "text", "duration", "at", "video", "task", "verb", "c", "r")
    manifest = {"version": 1, "seed": args.seed, "question": "The assistant spoke at a good moment.",
                "source": "HoloAssist (CDLA-Permissive-2.0)", "items": [{k: c[k] for k in keep} for c in items]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "manifest.js").write_text("window.STUDY = " + json.dumps(manifest) + ";\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "clips").iterdir()) / 1e6
    print(f"wrote {len(items) * 2} clips ({size:.1f} MB) and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
