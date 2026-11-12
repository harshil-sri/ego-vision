"""Analysis of the rating study (scripts.rating_study_build, study/index.html).

Input: study/responses/, any mix of
    *.json   one result file per rater (the "Download as file" button)
    *.txt    result codes, one per line (the "Copy code" button; base64 of the same JSON)

Design: every rater sees each clip once, in one of two versions.
    now    the message is spoken during fine-motor work
    held   the message is spoken when the rule releases it (hands free)
Message type: routine instruction or warning. Rating 1 (very bad moment)
to 5 (very good moment).

Prediction, fixed before any data was collected:
    routine   held is rated higher than now
    warning   held is NOT rated higher than now (lower is expected)
    so the difference (held - now) is larger for routine than for warning.

Statistics: each rater's mean rating per cell, then the mean over raters.
Intervals: 10,000 resamples of raters. A rater who gave the same rating to
every clip is listed and left out (no information about timing).

Usage:
    python -m scripts.rating_study_analyze
Outputs: reports/rating_study.json, reports/rating_study.md
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

CELLS = [("routine", "now"), ("routine", "held"), ("warning", "now"), ("warning", "held")]


def load(folder: Path) -> list[dict]:
    """-> one result dict per rater; a rater id that appears twice is kept once (the first)."""
    out, seen = [], set()
    for p in sorted(folder.glob("*")):
        texts = []
        if p.suffix == ".json":
            texts = [p.read_text(encoding="utf-8")]
        elif p.suffix == ".txt":
            for line in p.read_text(encoding="utf-8").split():
                try:
                    texts.append(base64.b64decode(line.strip()).decode("utf-8"))
                except Exception:
                    print(f"  skipped a line in {p.name}: not a result code")
        for t in texts:
            try:
                r = json.loads(t)
            except json.JSONDecodeError:
                print(f"  skipped an entry in {p.name}: not JSON")
                continue
            if r.get("id") not in seen and r.get("answers"):
                seen.add(r["id"]); out.append(r)
    return out


def cell_means(r: dict, type_of: dict) -> np.ndarray:
    """One rater -> (4,) mean rating per cell in the order of CELLS (nan if a cell is empty)."""
    v = {c: [] for c in CELLS}
    for a in r["answers"]:
        v[(type_of[a["c"]], a["k"])].append(a["r"])
    return np.array([np.mean(v[c]) if v[c] else np.nan for c in CELLS])


def summarize(M: np.ndarray, boot: int = 10000, seed: int = 0) -> dict:
    """M (raters, 4) cell means -> means, differences and intervals over raters."""
    rng = np.random.default_rng(seed)

    def stats(x):
        x = x[np.isfinite(x)]
        b = x[rng.integers(0, len(x), (boot, len(x)))].mean(1)
        return {"mean": float(x.mean()), "low": float(np.percentile(b, 2.5)), "high": float(np.percentile(b, 97.5)),
                "raters_above_0": int((x > 0).sum()), "raters_below_0": int((x < 0).sum()), "n": int(len(x))}

    d_r, d_w = M[:, 1] - M[:, 0], M[:, 3] - M[:, 2]
    return {"cells": {f"{t}, {k}": stats(M[:, i]) for i, (t, k) in enumerate(CELLS)},
            "routine: held - now": stats(d_r), "warning: held - now": stats(d_w),
            "interaction: (routine held - now) - (warning held - now)": stats(d_r - d_w)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="study/responses")
    ap.add_argument("--manifest", default="study/manifest.json")
    args = ap.parse_args()
    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["items"]
    type_of = {i["id"]: i["type"] for i in items}
    raters = load(Path(args.responses))
    flat = [r for r in raters if len({a["r"] for a in r["answers"]}) == 1]
    used = [r for r in raters if r not in flat]
    if not used:
        raise SystemExit(f"no usable responses in {args.responses}")
    M = np.stack([cell_means(r, type_of) for r in used])
    res = {"raters": len(raters), "used": len(used), "left_out_same_rating_everywhere": len(flat),
           "heard_voice": sum(r.get("sound") == "yes" for r in used), "all": summarize(M)}
    heard = np.array([r.get("sound") == "yes" for r in used])
    if 2 <= heard.sum() < len(used):
        res["only raters who heard the voice"] = summarize(M[heard])
    clip = {}
    for r in used:
        for a in r["answers"]:
            clip.setdefault(a["c"], {"now": [], "held": []})[a["k"]].append(a["r"])
    res["clips"] = {c: {"type": type_of[c], **{k: [float(np.mean(v)) if v else None, len(v)] for k, v in d.items()}}
                    for c, d in sorted(clip.items())}
    res["median_minutes"] = float(np.median([sum(a["ms"] for a in r["answers"]) / 60000 for r in used]))
    res["comments"] = [r["comment"] for r in raters if r.get("comment")]
    Path("reports/rating_study.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def line(name, s):
        return (f"| {name} | {s['mean']:+.2f} | {s['low']:+.2f} to {s['high']:+.2f} | "
                f"{s['raters_above_0']} above 0, {s['raters_below_0']} below 0 |")

    L = ["# Rating study: when should the assistant speak?", "",
         f"Generated by `python -m scripts.rating_study_analyze`. {res['raters']} raters, {res['used']} used "
         f"({res['left_out_same_rating_everywhere']} gave the same rating to every clip), "
         f"{res['heard_voice']} heard the voice. Rating 1 (very bad moment) to 5 (very good moment). "
         "\"now\" = the message during fine-motor work; \"held\" = the message when the rule releases it.", ""]
    for title, s in [(k, v) for k, v in res.items() if isinstance(v, dict) and "cells" in v]:
        L += [f"## {title.capitalize()}", "", "| message | now | held |", "|---|---|---|"]
        for t in ("routine", "warning"):
            a, b = s["cells"][f"{t}, now"], s["cells"][f"{t}, held"]
            L.append(f"| {t} | {a['mean']:.2f} ({a['low']:.2f} to {a['high']:.2f}) | "
                     f"{b['mean']:.2f} ({b['low']:.2f} to {b['high']:.2f}) |")
        L += ["", "| difference | mean | 95% interval over raters | raters |", "|---|---|---|---|"]
        L += [line(k, v) for k, v in s.items() if k != "cells"]
        L.append("")
    L += ["## Each clip", "", "| clip | type | now: mean (n) | held: mean (n) |", "|---|---|---|---|"]
    for c, d in res["clips"].items():
        f = lambda x: "-" if x[0] is None else f"{x[0]:.2f} ({x[1]})"
        L.append(f"| {c} | {d['type']} | {f(d['now'])} | {f(d['held'])} |")
    if res["comments"]:
        L += ["", "## Comments", ""] + [f"- {c}" for c in res["comments"]]
    Path("reports/rating_study.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
