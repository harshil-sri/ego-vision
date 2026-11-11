"""Ego-EXTRA, transcripts only: how often does a real expert speak UNASKED?

Ego-EXTRA (arXiv 2512.13238) has 50 h of a trainee doing a procedure while
a remote expert watches the trainee's camera and talks. Every recording
follows one of two protocols (protocol_split_wacv.json):

    1  proactive      the expert guides on purpose
    2  non-proactive  the expert talks only when asked, or when the expert
                      thinks advice is necessary

Protocol 2 is the behaviour an assistant should copy: speak only when it is
needed. This script measures, from the time-stamped transcripts alone
(1 MB download), how much expert speech there is and how much of it is
unasked.

Definitions
    utterance   one transcript line: start, end, speaker (E expert, A trainee)
    expert turn consecutive expert utterances; a new turn starts when the
                expert was silent for at least --turn-gap seconds
    unasked     an expert turn that starts when the trainee has not spoken
                in the last --ask-window seconds (no trainee utterance
                overlaps [start - ask_window, start])
    long        a turn with at least --long seconds of expert speech. Short
                turns are mostly "ok" / "perfect" (feedback, not advice).

Usage:
    python -m scripts.egoextra_speech
Outputs: reports/egoextra_speech.json, reports/egoextra_speech.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.paths import resolve_data_path

ROOT = "/mnt/ssd/ego-extra"
LINE = re.compile(r"^\s*\d+\s*\[(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)\]\s*([EA]):")


def load_transcript(path: Path) -> np.ndarray:
    """-> (n, 3) float array: start s, end s, is_expert (1.0 / 0.0), sorted by start."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups()[:8])
            rows.append((h1 * 3600 + m1 * 60 + s1 + ms1 / 1000, h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                         float(m.group(9) == "E")))
    a = np.array(rows, dtype=float).reshape(-1, 3)
    return a[np.argsort(a[:, 0])]


def expert_turns(u: np.ndarray, turn_gap: float, ask_window: float):
    """-> (turn start times, unasked flags, seconds of speech per turn) for one video."""
    e, a = u[u[:, 2] == 1], u[u[:, 2] == 0]
    starts, unasked, speech, last_end = [], [], [], -1e9
    for s, t, _ in e:
        if s - last_end >= turn_gap:
            starts.append(s); speech.append(0.0)
            # trainee spoke recently = any trainee utterance overlapping [s - ask_window, s]
            unasked.append(not np.any((a[:, 0] <= s) & (a[:, 1] >= s - ask_window)))
        speech[-1] += t - s
        last_end = max(last_end, t)
    return np.array(starts), np.array(unasked, dtype=bool), np.array(speech)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-gap", type=float, default=2.0)
    ap.add_argument("--ask-window", type=float, default=5.0)
    ap.add_argument("--long", type=float, default=1.5)
    args = ap.parse_args()

    root = Path(resolve_data_path(ROOT))
    proto = {v["ID"]: v["Protocol"] for v in json.loads((root / "protocol_split_wacv.json").read_text())["videos"]}
    per_video = []
    for p in sorted((root / "transcript").rglob("*_transcript.txt")):
        vid = p.name.replace("_transcript.txt", "")
        if vid not in proto:
            continue
        u = load_transcript(p)
        if len(u) < 5:
            continue
        minutes = (u[:, 1].max() - u[:, 0].min()) / 60
        starts, unasked, speech = expert_turns(u, args.turn_gap, args.ask_window)
        long_ = speech >= args.long
        e = u[u[:, 2] == 1]
        gaps = np.diff(starts[unasked]) if unasked.sum() > 1 else np.array([])
        per_video.append({
            "video": vid, "scenario": vid.split("_")[0], "protocol": proto[vid], "minutes": minutes,
            "expert_talk_share": float((e[:, 1] - e[:, 0]).sum() / 60 / minutes),
            "expert_turns_per_min": len(starts) / minutes,
            "unasked_per_min": float(unasked.sum() / minutes),
            "unasked_share": float(unasked.mean()) if len(unasked) else float("nan"),
            "unasked_long_per_min": float((unasked & long_).sum() / minutes),
            "unasked_short_per_min": float((unasked & ~long_).sum() / minutes),
            "trainee_utts_per_min": float((u[:, 2] == 0).sum() / minutes),
            "median_gap_between_unasked_s": float(np.median(gaps)) if len(gaps) else float("nan"),
        })

    keys = ["expert_talk_share", "expert_turns_per_min", "unasked_per_min", "unasked_share",
            "unasked_long_per_min", "unasked_short_per_min",
            "trainee_utts_per_min", "median_gap_between_unasked_s"]

    def summary(rows):
        out = {"videos": len(rows), "hours": round(sum(r["minutes"] for r in rows) / 60, 1)}
        for k in keys:                                    # median over videos, with the quartiles
            x = np.array([r[k] for r in rows], dtype=float)
            x = x[~np.isnan(x)]
            out[k] = [round(float(np.percentile(x, q)), 3) for q in (25, 50, 75)]
        return out

    res = {"turn_gap_s": args.turn_gap, "ask_window_s": args.ask_window,
           "by_protocol": {str(k): summary([r for r in per_video if r["protocol"] == k]) for k in (1, 2)},
           "by_scenario_protocol2": {s: summary([r for r in per_video if r["protocol"] == 2 and r["scenario"] == s])
                                     for s in sorted({r["scenario"] for r in per_video if r["protocol"] == 2})}}
    # sensitivity of the headline number to the two thresholds
    sens = {}
    for tg in (1.0, 2.0, 4.0):
        for aw in (2.0, 5.0, 10.0):
            vals = []
            for p in sorted((root / "transcript").rglob("*_transcript.txt")):
                vid = p.name.replace("_transcript.txt", "")
                if proto.get(vid) != 2:
                    continue
                u = load_transcript(p)
                if len(u) < 5:
                    continue
                s, un, sp = expert_turns(u, tg, aw)
                vals.append((un & (sp >= args.long)).sum() / ((u[:, 1].max() - u[:, 0].min()) / 60))
            sens[f"turn_gap {tg}, ask_window {aw}"] = round(float(np.median(vals)), 2)
    res["sensitivity_unasked_long_per_min_protocol2"] = sens

    Path("reports/egoextra_speech.json").write_text(json.dumps(res, indent=2))
    names = {"expert_talk_share": "share of time the expert is talking",
             "expert_turns_per_min": "expert turns per minute",
             "unasked_per_min": "unasked expert turns per minute",
             "unasked_share": "share of expert turns that are unasked",
             "unasked_long_per_min": f"unasked turns per minute, at least {args.long} s of speech",
             "unasked_short_per_min": f"unasked turns per minute, shorter (mostly 'ok', 'perfect')",
             "trainee_utts_per_min": "trainee utterances per minute",
             "median_gap_between_unasked_s": "median gap between unasked turns (s)"}
    L = ["# Ego-EXTRA: how often does an expert speak unasked?", "",
         "Generated by `python -m scripts.egoextra_speech` from the transcripts only. "
         f"Expert turn = expert speech after at least {args.turn_gap} s of expert silence. "
         f"Unasked = the trainee did not speak in the {args.ask_window} s before the turn. "
f"Long = at least {args.long} s of expert speech in the turn. "
         "Each cell: median over videos [25th, 75th percentile].", "",
         "| | protocol 1 (proactive) | protocol 2 (speak only when asked or necessary) |", "|---|---|---|"]
    p1, p2 = res["by_protocol"]["1"], res["by_protocol"]["2"]
    L.append(f"| videos, hours | {p1['videos']}, {p1['hours']} h | {p2['videos']}, {p2['hours']} h |")
    for k in keys:
        L.append(f"| {names[k]} | {p1[k][1]} [{p1[k][0]}, {p1[k][2]}] | {p2[k][1]} [{p2[k][0]}, {p2[k][2]}] |")
    L += ["", "## Protocol 2 by scenario", "", "| scenario | videos | unasked turns per minute | of these, long | share unasked | expert talk share |",
          "|---|---|---|---|---|---|"]
    for s, r in res["by_scenario_protocol2"].items():
        L.append(f"| {s} | {r['videos']} | {r['unasked_per_min'][1]} | {r['unasked_long_per_min'][1]} | {r['unasked_share'][1]} | {r['expert_talk_share'][1]} |")
    L += ["", "## Sensitivity (protocol 2, median unasked long turns per minute)", "",
          "| thresholds | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sens.items()]
    Path("reports/egoextra_speech.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
