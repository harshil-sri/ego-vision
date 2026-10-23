"""Build a video manifest from annotations + HF mirror sizes, then greedily
select a subset of videos that maximises candidate-verb training instances
per GB within a disk budget.

Two outputs:
  reports/hf_video_manifest.csv  — every video available on the HF mirror with
                                   size, fps, split and candidate-verb counts
  reports/selected_videos.csv    — the greedy pick, in selection order

Value rule (kept deliberately simple so it can be explained and defended):
  density = candidate-verb train instances in the video / size in GB.
Videos are taken in density order until the budget is spent. Validation
videos are selected the same way from a small reserved share of the budget,
so evaluation data exists too.

Usage:
    python -m scripts.select_videos --budget-gb 80 \
        --annotations /mnt/ssd/epic-tools/epic-kitchens-100-annotations \
        --repo awsaf49/epic_kitchens_100 \
        --manifest reports/hf_video_manifest.csv \
        --out reports/selected_videos.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

CANDIDATE_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]
GB = 1024**3
VAL_SHARE = 0.15  # fraction of the budget reserved for validation videos


def build_manifest(annotations: Path, repo: str) -> pd.DataFrame:
    """Join annotation stats with HF file sizes.

    Output DataFrame: one row per video present on the HF mirror, columns:
      video_id, split, size_bytes, fps, duration_sec, resolution,
      candidate_train, candidate_val, total_segments
    """
    info = pd.read_csv(annotations / "EPIC_100_video_info.csv")
    info = info.set_index("video_id")  # fps/duration/resolution per video

    counts = {}
    for split, csv_name in [("train", "EPIC_100_train.csv"),
                            ("validation", "EPIC_100_validation.csv")]:
        df = pd.read_csv(annotations / csv_name, usecols=["video_id", "verb"])
        cand = df[df["verb"].isin(CANDIDATE_VERBS)]
        per_video = cand.groupby("video_id").size()
        total = df.groupby("video_id").size()
        for vid in df["video_id"].unique():
            entry = counts.setdefault(vid, {"candidate_train": 0,
                                            "candidate_val": 0,
                                            "total_segments": 0,
                                            "split": split})
            entry[f"candidate_{split if split == 'train' else 'val'}"] = \
                int(per_video.get(vid, 0))
            entry["total_segments"] += int(total.get(vid, 0))

    api = HfApi()
    rows = []
    for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True):
        if not f.path.upper().endswith(".MP4"):
            continue
        vid = f.path.split("/")[-1].removesuffix(".MP4")
        if vid not in counts:
            continue  # mirror holds videos whose annotations are test-only
        meta = info.loc[vid] if vid in info.index else None
        rows.append({
            "video_id": vid,
            "hf_path": f.path,
            "size_bytes": f.size,
            "fps": float(meta["fps"]) if meta is not None else -1.0,
            "duration_sec": float(meta["duration"]) if meta is not None else -1.0,
            "resolution": str(meta["resolution"]) if meta is not None else "?",
            **counts[vid],
        })
    return pd.DataFrame(rows).sort_values("video_id")


def greedy_select(manifest: pd.DataFrame, budget_gb: float) -> pd.DataFrame:
    """Greedy pick by candidate-instance density within the budget.

    Input: manifest DataFrame from build_manifest.
    Output: selected rows in pick order, with a running 'cumulative_gb'.
    """
    picked_rows = []
    used = 0.0
    val_budget = budget_gb * VAL_SHARE
    val_used = 0.0

    manifest = manifest.copy()
    manifest["size_gb"] = manifest["size_bytes"] / GB

    for split, column, limit, used_ref in [
        ("train", "candidate_train", budget_gb - val_budget, None),
        ("validation", "candidate_val", val_budget, None),
    ]:
        part = manifest[manifest["split"] == split]
        # Density: candidate instances per GB; drop zero-value videos for
        # train (they waste budget) but keep any for validation only if
        # they carry candidates too.
        part = part[part[column] > 0].sort_values(
            "size_gb", key=lambda s: s)  # stable base order
        part["density"] = part[column] / part["size_gb"]
        part = part.sort_values("density", ascending=False)
        split_used = 0.0
        for _, row in part.iterrows():
            if split_used + row["size_gb"] > limit:
                continue
            split_used += row["size_gb"]
            used += row["size_gb"]
            picked_rows.append({"video_id": row["video_id"],
                                "split": split,
                                "size_gb": round(row["size_gb"], 3),
                                "density": round(row["density"], 1),
                                "candidate_instances": int(row[column]),
                                "cumulative_gb": round(used, 3)})
    return pd.DataFrame(picked_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gb", type=float, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--repo", default="awsaf49/epic_kitchens_100")
    parser.add_argument("--manifest", type=Path,
                        default=Path("reports/hf_video_manifest.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/selected_videos.csv"))
    args = parser.parse_args()

    manifest = build_manifest(args.annotations, args.repo)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"manifest: {len(manifest)} videos on mirror, "
          f"{manifest['size_bytes'].sum() / GB:.1f} GB total -> {args.manifest}")

    selected = greedy_select(manifest, args.budget_gb)
    selected.to_csv(args.out, index=False)
    by_split = selected.groupby("split").agg(
        videos=("video_id", "count"),
        gb=("size_gb", "sum"),
        candidate_instances=("candidate_instances", "sum"))
    print(f"selected {len(selected)} videos, "
          f"{selected['size_gb'].sum():.1f} GB -> {args.out}")
    print(by_split.to_string())


if __name__ == "__main__":
    main()
