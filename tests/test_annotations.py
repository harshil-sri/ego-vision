"""Tests for src/data/annotations.py and src/data/splits.py.

Run against the REAL EPIC-KITCHENS-100 annotation CSVs (they exist on this
machine at /mnt/ssd/epic-tools/epic-kitchens-100-annotations; skipped with
a clear message elsewhere).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.annotations import (load_segments, load_video_info,
                                  verb_label_map, save_verb_map, load_verb_map)
from src.data.splits import make_splits, write_selected_inventory

ANNOTATIONS = Path("/mnt/ssd/epic-tools/epic-kitchens-100-annotations")
TRAIN_CSV = ANNOTATIONS / "EPIC_100_train.csv"

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV.exists(),
    reason=f"real EPIC annotation CSVs not available at {ANNOTATIONS}",
)

CANDIDATE_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]


def test_load_segments_filters_verbs():
    """Only the requested verbs survive; columns/dtypes match the contract."""
    df = load_segments(TRAIN_CSV, verbs=["take", "put"], video_ids=None)
    assert set(df["verb"]).issubset({"take", "put"})
    assert len(df) > 0  # 'take'/'put' are the two most frequent EPIC verbs


def test_load_segments_filters_video_ids():
    """Only rows of the requested video survive."""
    df = load_segments(TRAIN_CSV, verbs=None, video_ids={"P01_01"})
    assert len(df) > 0
    assert set(df["video_id"]) == {"P01_01"}


def test_load_segments_columns_and_dtypes():
    """Exactly the contract columns, with int64 frames and class ids."""
    df = load_segments(TRAIN_CSV, verbs=None, video_ids={"P01_01"})
    assert list(df.columns) == ["video_id", "narration_id", "start_frame",
                                "stop_frame", "verb", "verb_class"]
    assert str(df["start_frame"].dtype) == "int64"
    assert str(df["stop_frame"].dtype) == "int64"
    assert str(df["verb_class"].dtype) == "int64"
    # inclusive stop convention: no segment has stop < start
    assert (df["stop_frame"] >= df["start_frame"]).all()


def test_verb_label_map_is_alphabetical():
    """Ids 0..7 assigned in alphabetical order regardless of input order."""
    shuffled = ["wash", "pour", "close", "take", "mix", "open", "put", "cut"]
    m = verb_label_map(shuffled)
    assert m == {"close": 0, "cut": 1, "mix": 2, "open": 3, "pour": 4,
                 "put": 5, "take": 6, "wash": 7}
    assert sorted(m.values()) == list(range(len(shuffled)))


def test_verb_label_map_rejects_duplicates():
    with pytest.raises(ValueError):
        verb_label_map(["take", "take"])


def test_save_load_verb_map_roundtrip(tmp_path):
    """The inference contract: the saved map survives a save/load roundtrip."""
    m = verb_label_map(CANDIDATE_VERBS)
    p = tmp_path / "verb_map.csv"
    save_verb_map(p, m)
    assert load_verb_map(p) == m


def test_load_video_info():
    """Indexed by video_id with a positive fps for a known video."""
    info = load_video_info(ANNOTATIONS)
    assert "P01_01" in info.index
    assert float(info.loc["P01_01", "fps"]) > 0
    assert {"fps", "duration", "resolution"}.issubset(info.columns)


def test_splits_empty_with_no_videos(tmp_path):
    """No videos on disk -> empty frames, empty id sets, no crash."""
    train_df, val_df, train_ids, val_ids = make_splits(
        ANNOTATIONS, tmp_path, heldout_participants=["P22", "P30"],
        verbs=CANDIDATE_VERBS)
    assert len(train_df) == 0 and len(val_df) == 0
    assert train_ids == set() and val_ids == set()


def test_splits_heldout_participants_go_to_val_only(tmp_path):
    """Rows of heldout participants land in val; train never sees them."""
    root = tmp_path / "videos"
    # Fake EMPTY marker files: make_splits only checks existence, it never
    # opens the videos, so touch-ing the paths is enough.
    for vid in ["P22_101", "P01_01", "P30_103"]:
        p = root / vid.split("_")[0] / "videos" / f"{vid}.MP4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    train_df, val_df, train_ids, val_ids = make_splits(
        ANNOTATIONS, root, heldout_participants=["P22", "P30"],
        verbs=None)  # all verbs: the marker videos may lack candidate verbs

    val_parts = {v.split("_")[0] for v in val_df["video_id"]}
    train_parts = {v.split("_")[0] for v in train_df["video_id"]}
    assert val_parts <= {"P22", "P30"}
    assert train_parts.isdisjoint({"P22", "P30"})
    assert train_ids | val_ids == {"P22_101", "P01_01", "P30_103"}
    # every val row comes from a held-out participant and vice versa
    assert len(val_df) > 0 and len(train_df) > 0


def test_write_selected_inventory(tmp_path):
    """Inventory CSV lists existing paths and sizes, blank for absent ones."""
    root = tmp_path / "videos"
    p = root / "P01" / "videos" / "P01_01.MP4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 1234)

    out = tmp_path / "inv.csv"
    inv = write_selected_inventory(out, ["P01_01", "P22_101"], root,
                                   downscaled_root=tmp_path / "EPIC-256")
    saved = pd.read_csv(out)
    assert list(saved["video_id"]) == ["P01_01", "P22_101"]
    row = saved[saved["video_id"] == "P01_01"].iloc[0]
    assert row["original_size_bytes"] == 1234
    assert pd.isna(row["downscaled_path"])  # absent root -> blank cell
    assert len(inv) == 2  # helper also returns the frame
