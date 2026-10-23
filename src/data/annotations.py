"""Load EPIC-KITCHENS-100 annotation CSVs into typed DataFrames.

Annotation facts (verified against the real CSVs, see DECISIONS_NEEDED.md S1):
  - One row per narrated action segment ("narration").
  - ``stop_frame`` is INCLUSIVE: a segment covers frames
    start_frame..stop_frame, i.e. clip length = stop_frame - start_frame + 1.
    Evidence: P01_01 row 0 has frames 8..202 with timestamps 0.14s..3.37s at
    59.94 fps -> 3.23 s * 59.94 = 193.6 frames ~= 202 - 8 = 194 (inclusive).
  - ``EPIC_100_video_info.csv`` has one row per video with the true fps
    (EPIC videos vary; do arithmetic in frames at this fps, never assume 50).

Label-id convention (IMPORTANT for inference): ``verb_label_map`` assigns ids
0..len(verbs)-1 ALPHABETICALLY. The ids are therefore stable for a fixed verb
list, but any change to the list renumbers classes. The SAME map must be
saved into every run directory (see ``save_verb_map``) and reloaded at
inference time; a checkpoint without its verb map is uninterpretable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns we keep from the split CSVs (a subset of the 15 official columns).
SEGMENT_COLUMNS = [
    "video_id",      # str, e.g. "P01_01"
    "narration_id",  # str, e.g. "P01_01_0"
    "start_frame",   # int64, 0-based frame number of the first frame
    "stop_frame",    # int64, INCLUSIVE last frame of the segment
    "verb",          # str, verb word, e.g. "take"
    "verb_class",    # int64, official EPIC verb class id (97 classes)
]

VIDEO_INFO_CSV = "EPIC_100_video_info.csv"


def load_segments(split_csv: str | Path,
                  verbs: list[str] | None = None,
                  video_ids: set[str] | None = None) -> pd.DataFrame:
    """Load one annotation split CSV, filtered, as a typed DataFrame.

    Inputs:
        split_csv: path to EPIC_100_train.csv / EPIC_100_validation.csv
            (or any CSV with the official 15 columns).
        verbs: if not None, keep only rows whose ``verb`` is in this list.
        video_ids: if not None, keep only rows whose ``video_id`` is in this
            set (the set of videos actually present on disk).

    Output: pd.DataFrame with exactly the columns
        video_id (str), narration_id (str),
        start_frame (int64), stop_frame (int64),
        verb (str), verb_class (int64)
    one row per action segment, reset to a fresh RangeIndex. Rows with
    missing frames/verb values are dropped before casting (annotation
    files contain a few such rows; they cannot be used for training).

    Raises FileNotFoundError if split_csv does not exist.
    """
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(
            f"annotation CSV not found: {split_csv}. Expected one of the "
            f"EPIC_100_*.csv files from the epic-kitchens-100-annotations "
            f"repo (see configs/data.yaml annotations_dir)."
        )

    # Read only the columns we need (the train CSV has ~70k rows x 15 cols).
    df = pd.read_csv(split_csv, usecols=SEGMENT_COLUMNS)

    # Drop rows the pipeline cannot use (NaN in any needed column).
    df = df.dropna(subset=SEGMENT_COLUMNS)

    if verbs is not None:
        df = df[df["verb"].isin(verbs)]
    if video_ids is not None:
        df = df[df["video_id"].isin(video_ids)]

    df = df.copy()
    df["start_frame"] = df["start_frame"].astype("int64")
    df["stop_frame"] = df["stop_frame"].astype("int64")
    df["verb_class"] = df["verb_class"].astype("int64")
    return df.reset_index(drop=True)


def load_video_info(annotations_dir: str | Path) -> pd.DataFrame:
    """Load EPIC_100_video_info.csv.

    Input: annotations_dir containing EPIC_100_video_info.csv.
    Output: pd.DataFrame indexed by video_id (str) with columns
        fps (float64, the video's true frame rate, e.g. 59.9400599...),
        duration (float64, seconds),
        resolution (str, "WIDTHxHEIGHT", e.g. "1920x1080").
    """
    path = Path(annotations_dir) / VIDEO_INFO_CSV
    if not path.exists():
        raise FileNotFoundError(f"video info CSV not found: {path}")
    df = pd.read_csv(path)
    return df.set_index("video_id")


def verb_label_map(verbs: list[str]) -> dict[str, int]:
    """Map each verb to a contiguous class id, ids assigned alphabetically.

    Input:  verbs, list[str], e.g. ["take", "put", "wash"] (any order).
    Output: dict[str, int], e.g. {"put": 0, "take": 1, "wash": 2}.

    The alphabetical order makes ids reproducible from the verb list alone
    (no dependence on config ordering). Duplicates raise ValueError.

    WARNING (inference contract): a trained checkpoint is only valid with the
    exact verb->id map it was trained with. Save the map into every run dir
    (``save_verb_map`` / ``load_verb_map`` below) and load it back at
    inference; never re-derive it from a possibly-edited config alone.
    """
    if len(set(verbs)) != len(verbs):
        raise ValueError(f"duplicate verbs in list: {sorted(verbs)}")
    return {verb: idx for idx, verb in enumerate(sorted(verbs))}


def save_verb_map(path: str | Path, verb_map: dict[str, int]) -> None:
    """Persist a verb->id map as a small CSV (verb,label), sorted by label.

    Input: path to write; verb_map from ``verb_label_map``.
    Output: None (file with one row per class: verb str, label int).
    """
    rows = sorted(verb_map.items(), key=lambda kv: kv[1])
    df = pd.DataFrame(rows, columns=["verb", "label"])
    df.to_csv(path, index=False)


def load_verb_map(path: str | Path) -> dict[str, int]:
    """Read back a map written by ``save_verb_map``.

    Input: path to a verb-map CSV with columns verb (str), label (int).
    Output: dict[str, int] mapping verb -> label id.
    """
    df = pd.read_csv(path)
    return dict(zip(df["verb"].astype(str), df["label"].astype(int)))
