"""Download the selected EPIC-KITCHENS video subset from the HF mirror.

Resumable: huggingface_hub skips files already present with the right size,
so this can be re-run across sessions. Files land in the same
<participant>/videos/<video>.MP4 layout the official downloader uses, so
downstream code does not care which route the data came from.

Usage:
    python -m scripts.download_subset \
        --selected reports/selected_videos.csv \
        --repo awsaf49/epic_kitchens_100 \
        --out /mnt/ssd/epic-kitchens/EPIC-KITCHENS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True,
                        help="CSV written by scripts/select_videos.py")
    parser.add_argument("--repo", default="awsaf49/epic_kitchens_100")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected)
    patterns = sorted(selected["video_id"].map(lambda v: f"*{v}.MP4"))
    total_gb = 0.0
    for vid, size in zip(selected["video_id"], selected["size_gb"]):
        total_gb += size
    print(f"downloading {len(patterns)} videos, about {total_gb:.1f} GB "
          f"from {args.repo} -> {args.out}")

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        max_workers=args.max_workers,
    )
    print("done")


if __name__ == "__main__":
    main()
