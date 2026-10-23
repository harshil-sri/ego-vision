"""Train/validation splits restricted to videos actually present on disk.

Split policy (V1 in DECISIONS_NEEDED.md — an internal split, a forced
deviation the owner must sign off on): the only fast data route (the HF
mirror ``awsaf49/epic_kitchens_100``) carries TRAIN-split videos only; the
official validation videos are reachable at ~182 KB/s from the official
server, which is unusable. Until that changes, validation is carved from the
downloaded train videos by holding out whole participants (configs/data.yaml
``heldout_participants``, currently [P22, P30]). Training never touches the
held-out participants' videos, so the validation numbers are not contaminated
by seen-videos (though the participants' kitchens appear in neither side).

Disk layout (both roots, interchangeable for the Dataset):
    <videos_root>/<Pxx>/videos/<video_id>.MP4     original 1080p
    <downscaled_root>/<Pxx>/videos/<video_id>.MP4 short-side-256 copy
The downscaled copies keep fps and frame count identical, so annotation
frame indices are valid against either root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.annotations import load_segments

TRAIN_CSV = "EPIC_100_train.csv"


def video_path(videos_root: str | Path, video_id: str) -> Path:
    """Filesystem path of one video inside a root with the EPIC layout.

    Inputs: videos_root (str/Path), video_id (str, e.g. "P01_01"; the
        participant prefix "P01" is everything before the first underscore).
    Output: Path <videos_root>/<participant>/videos/<video_id>.MP4
    """
    participant = video_id.split("_")[0]
    return Path(videos_root) / participant / "videos" / f"{video_id}.MP4"


def find_available_video_ids(videos_root: str | Path,
                             downscaled_root: str | Path | None = None,
                             video_ids=None) -> set[str]:
    """Video ids that exist on disk in at least one root.

    Inputs:
        videos_root: original-video root (may not exist yet; downloads are
            still running, so existence is checked per file, never assumed).
        downscaled_root: optional downscaled root; a video counts as
            available if present in EITHER root.
        video_ids: optional iterable to restrict the check to; if None, the
            participant folders under videos_root are scanned.
    Output: set[str] of video_id values for which os.path.exists is True.

    This only tests existence (cheap); it does not verify the file is fully
    downloaded. The bulk downloader only moves a file to its final name when
    complete (partials live under .cache/ with a .incomplete suffix).
    """
    available: set[str] = set()

    def _check(root, ids):
        for vid in ids:
            if os.path.exists(video_path(root, vid)):
                available.add(vid)

    if video_ids is not None:
        ids = list(dict.fromkeys(video_ids))  # dedupe, keep order
        _check(videos_root, ids)
        if downscaled_root is not None:
            _check(downscaled_root, ids)
        return available

    # No candidate list given: scan the root's participant folders.
    root = Path(videos_root)
    if root.exists():
        for mp4 in sorted(root.glob("P*/videos/*.MP4")):
            if ".incomplete" in mp4.name:
                continue
            available.add(mp4.stem)
    return available


def make_splits(annotations_dir: str | Path,
                videos_root: str | Path,
                heldout_participants: list[str],
                verbs: list[str] | None = None,
                downscaled_root: str | Path | None = None,
                split_csv: str | Path | None = None) -> tuple[pd.DataFrame,
                                                               pd.DataFrame,
                                                               set[str],
                                                               set[str]]:
    """Build train/val segment tables restricted to videos present on disk.

    Inputs:
        annotations_dir: dir with EPIC_100_train.csv and
            EPIC_100_video_info.csv.
        videos_root: original videos root (existence-checked per video).
        heldout_participants: e.g. ["P22", "P30"]; ALL of their rows go to
            validation (internal split V1, see module docstring).
        verbs: candidate verbs to keep, or None for all verbs.
        downscaled_root: optional; a video counts as available if it exists
            under either root.
        split_csv: optional explicit CSV path; defaults to
            <annotations_dir>/EPIC_100_train.csv (the internal split is cut
            from the official TRAIN split because that is all the mirror
            provides).

    Outputs (train_df, val_df, train_video_ids, val_video_ids):
        train_df, val_df: pd.DataFrame with columns video_id (str),
            narration_id (str), start_frame (int64), stop_frame (int64,
            INCLUSIVE), verb (str), verb_class (int64).
        train_video_ids, val_video_ids: set[str] of the video ids actually
            used on each side (subset of what exists on disk).

    If no requested video exists on disk, both DataFrames are empty and both
    sets are empty (no crash) — the state while the bulk download warms up.
    """
    annotations_dir = Path(annotations_dir)
    if split_csv is None:
        split_csv = annotations_dir / TRAIN_CSV

    # Candidate universe: every video id mentioned in the split CSV.
    all_ids = set(pd.read_csv(split_csv, usecols=["video_id"])["video_id"])
    available = find_available_video_ids(videos_root, downscaled_root, all_ids)

    df = load_segments(split_csv, verbs=verbs, video_ids=available)

    heldout = set(heldout_participants)

    def _participant(video_id: str) -> str:
        return video_id.split("_")[0]

    is_val = df["video_id"].map(_participant).isin(heldout)
    val_df = df[is_val].reset_index(drop=True)
    train_df = df[~is_val].reset_index(drop=True)

    train_video_ids = set(train_df["video_id"])
    val_video_ids = set(val_df["video_id"])
    return train_df, val_df, train_video_ids, val_video_ids


def write_selected_inventory(out_csv: str | Path,
                             video_ids,
                             videos_root: str | Path,
                             downscaled_root: str | Path | None = None) -> pd.DataFrame:
    """Write a CSV inventory of the videos a run will use.

    Inputs:
        out_csv: destination path (parents are created).
        video_ids: iterable of video_id strings.
        videos_root / downscaled_root: the two roots to check.

    Output: the same inventory as a pd.DataFrame with columns
        video_id (str), participant (str),
        original_path (str, "" if absent),
        original_size_bytes (Int64, NA if absent),
        downscaled_path (str, "" if absent),
        downscaled_size_bytes (Int64, NA if absent).
    Rows are sorted by video_id; one row per requested video.
    """
    rows = []
    for vid in sorted(set(video_ids)):
        orig = video_path(videos_root, vid)
        down = video_path(downscaled_root, vid) if downscaled_root else None
        rows.append({
            "video_id": vid,
            "participant": vid.split("_")[0],
            "original_path": str(orig) if orig.exists() else "",
            "original_size_bytes": orig.stat().st_size if orig.exists() else pd.NA,
            "downscaled_path": str(down) if (down is not None and down.exists()) else "",
            "downscaled_size_bytes": down.stat().st_size if (down is not None and down.exists()) else pd.NA,
        })
    inv = pd.DataFrame(rows, columns=[
        "video_id", "participant", "original_path", "original_size_bytes",
        "downscaled_path", "downscaled_size_bytes",
    ])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out_csv, index=False)
    return inv
