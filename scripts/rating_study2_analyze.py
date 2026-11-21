"""Analysis of the second, confirmatory round of the rating study (study/rate/).

Written before any rater of this round took part. The hypotheses, the
stopping rule and the exclusions are in study/PREREGISTRATION_v2.md;
this script implements them and nothing else. Exploratory analyses are
marked as such in the report.

Input: study/responses_v2/, any mix of
    *.csv    the "ratings" tab of the collection sheet (column "json")
    *.json   one result file per rater ("Download as file")
    *.txt    result codes, one per line ("Copy code")

Rules (pre-registered):
    - only results of this round (v == "v2") with consent recorded
    - complete: 20 answers, every clip once
    - the same rater id twice: the first is kept
    - left out and counted: a rater who gave the same rating to every clip
Statistics as in the pilot (scripts.rating_study_analyze): each rater's
mean per cell, then the mean over raters, 95% interval from 10,000
resamples of raters.
    H1 warning: held - now, interval below 0
    H2 interaction (main test): (routine held - now) - (warning held - now), interval above 0
    H3 routine: held - now, upper end of the interval below +0.25

Usage:  python -m scripts.rating_study2_analyze
Outputs: reports/rating_study2.json, reports/rating_study2.md
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rating_study_analyze import cell_means, summarize

SESOI = 0.25  # smallest difference of interest for H3, in scale points


def load(folder: Path) -> list[dict]:
    texts = []
    for p in sorted(folder.glob("*")):
        if p.suffix == ".csv":
            texts += pd.read_csv(p)["json"].dropna().astype(str).tolist()
        elif p.suffix == ".json":
            texts.append(p.read_text(encoding="utf-8"))
        elif p.suffix == ".txt":
            for line in p.read_text(encoding="utf-8").split():
                try:
                    texts.append(base64.b64decode(line.strip()).decode("utf-8"))
                except Exception:
                    print(f"  skipped a line in {p.name}: not a result code")
    out, seen = [], set()
    for t in texts:
        try:
            r = json.loads(t)
        except json.JSONDecodeError:
            continue
        if r.get("v") == "v2" and r.get("id") not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="study/responses_v2")
    ap.add_argument("--manifest", default="study/manifest.json")
    args = ap.parse_args()
    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["items"]
    type_of = {i["id"]: i["type"] for i in items}
    clips = sorted(i["id"] for i in items if i["type"] != "practice")
    raters = load(Path(args.responses))
    no_consent = [r for r in raters if not (r.get("consent") or {}).get("agree")]
    incomplete = [r for r in raters if r not in no_consent and sorted(a["c"] for a in r["answers"]) != clips]
    ok = [r for r in raters if r not in no_consent and r not in incomplete]
    flat = [r for r in ok if len({a["r"] for a in r["answers"]}) == 1]
    used = [r for r in ok if r not in flat]
    if not used:
        raise SystemExit(f"no usable responses in {args.responses}")
    M = np.stack([cell_means(r, type_of) for r in used])
    s = summarize(M)
    w, x, rt = (s["warning: held - now"], s["interaction: (routine held - now) - (warning held - now)"],
                s["routine: held - now"])
    tests = {
        "H1 warning: held < now": {"supported": w["high"] < 0, **w},
        "H2 interaction > 0 (main test)": {"supported": x["low"] > 0, **x},
        f"H3 routine: held - now below +{SESOI}": {"supported": rt["high"] < SESOI, "held_worse_than_now": rt["high"] < 0, **rt},
    }
    res = {"raters_received": len(raters), "no_consent": len(no_consent), "incomplete": len(incomplete),
           "same_rating_everywhere": len(flat), "used": len(used),
           "heard_voice": sum(r.get("sound") == "yes" for r in used), "tests": tests, "all": s}
    heard = np.array([r.get("sound") == "yes" for r in used])
    if 2 <= heard.sum() < len(used):
        res["exploratory: only raters who heard the voice"] = summarize(M[heard])
    clip = {}
    for r in used:
        for a in r["answers"]:
            clip.setdefault(a["c"], {"now": [], "held": []})[a["k"]].append(a["r"])
    res["clips"] = {c: {"type": type_of[c], **{k: [float(np.mean(v)) if v else None, len(v)] for k, v in d.items()}}
                    for c, d in sorted(clip.items())}
    res["median_minutes_rating"] = float(np.median([sum(a["ms"] for a in r["answers"]) / 60000 for r in used]))
    res["age"] = pd.Series([r.get("age") or "not given" for r in used]).value_counts().to_dict()
    res["field"] = pd.Series([r.get("field") or "not given" for r in used]).value_counts().to_dict()
    res["comments"] = [r["comment"] for r in used if r.get("comment")]
    Path("reports/rating_study2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    f = lambda d: f"{d['mean']:+.2f} ({d['low']:+.2f} to {d['high']:+.2f})"
    L = ["# Rating study, second round (pre-registered, confirmatory)", "",
         "Generated by `python -m scripts.rating_study2_analyze`. Plan: `study/PREREGISTRATION_v2.md`.", "",
         f"Received {len(raters)}; without consent {len(no_consent)}; incomplete {len(incomplete)}; "
         f"same rating for every clip {len(flat)}; **used {len(used)}** ({res['heard_voice']} heard the voice). "
         f"Median time on the ratings: {res['median_minutes_rating']:.1f} min.", "",
         "## Pre-registered tests", "", "| test | mean (95% interval over raters) | raters above / below 0 | result |",
         "|---|---|---|---|"]
    for name, t in tests.items():
        L.append(f"| {name} | {f(t)} | {t['raters_above_0']} / {t['raters_below_0']} | "
                 f"{'supported' if t['supported'] else 'not supported'} |")
    L += ["", "## Cell means", "", "| message | now | held |", "|---|---|---|"]
    for typ in ("routine", "warning"):
        c = s["cells"]
        L.append(f"| {typ} | {c[f'{typ}, now']['mean']:.2f} ({c[f'{typ}, now']['low']:.2f} to {c[f'{typ}, now']['high']:.2f}) | "
                 f"{c[f'{typ}, held']['mean']:.2f} ({c[f'{typ}, held']['low']:.2f} to {c[f'{typ}, held']['high']:.2f}) |")
    if "exploratory: only raters who heard the voice" in res:
        e = res["exploratory: only raters who heard the voice"]
        L += ["", "## Exploratory: only raters who heard the voice", ""]
        L += [f"- {k}: {f(e[k])}" for k in ("routine: held - now", "warning: held - now",
                                            "interaction: (routine held - now) - (warning held - now)")]
    L += ["", "## Each clip", "", "| clip | type | now: mean (n) | held: mean (n) |", "|---|---|---|---|"]
    for c, d in res["clips"].items():
        L.append(f"| {c} | {d['type']} | {d['now'][0]:.2f} ({d['now'][1]}) | {d['held'][0]:.2f} ({d['held'][1]}) |"
                 if d["now"][0] is not None and d["held"][0] is not None else f"| {c} | {d['type']} | n/a | n/a |")
    L += ["", "## Raters (optional answers)", "", f"- Age: {res['age']}", f"- Field: {res['field']}", "",
          "## Comments", ""] + [f"- {c}" for c in res["comments"]]
    Path("reports/rating_study2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
