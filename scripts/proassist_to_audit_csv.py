"""Write ProAssist's released HoloAssist outputs as input for scripts.timing_audit (an example converter).

    python -m scripts.proassist_to_audit_csv --split test --out runs/proassist_test.csv
    python -m scripts.timing_audit --speech runs/proassist_test.csv --name ProAssist --out reports/timing_audit_proassist

One row per frame where ProAssist spoke ("gen" not empty), with the end of the stream as "end" (with
--sample all: the three dialogues are pooled and the shortest stream end is used). A video
on which it never spoke gets one row with an empty time, so that it still counts.
Input: the eval.zip of Hugging Face 594zyc/ProAssist-Model-L4096-I1 (see scripts/proassist_baseline.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.proassist_baseline import ZIP
from src.paths import resolve_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--sample", default="all",
                    help="which of the three generated dialogues per video (0, 1, 2), or 'all' to pool them "
                         "as scripts.proassist_baseline does")
    ap.add_argument("--threshold", default="0.3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pat = re.compile(rf"eval/holoassist-dialog_{args.split}_L0_I1/stream/notalk{args.threshold}-maxlen_4k/results/\d+\.json")
    samples = defaultdict(list)
    with zipfile.ZipFile(resolve_data_path(ZIP)) as z:
        for name in sorted(z.namelist()):
            if pat.match(name):
                d = json.loads(z.read(name))
                samples[d["metadata"]["video_uid"]].append(d["predictions"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "t", "end"])
        for vid, preds_list in sorted(samples.items()):
            chosen = preds_list if args.sample == "all" else preds_list[int(args.sample):int(args.sample) + 1]
            if not chosen:
                continue
            end = min(float(p[-1]["timestamp_in_stream"]) for p in chosen)
            t = sorted(float(x["timestamp_in_stream"]) for p in chosen for x in p if x["gen"])
            for x in t or [""]:
                w.writerow([vid, x, end]); rows += 1
    print(f"wrote {rows} rows for {len(samples)} videos to {args.out}")


if __name__ == "__main__":
    main()
