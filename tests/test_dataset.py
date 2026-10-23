"""Tests for src/data/dataset.py (EpicVerbDataset) on a tiny local mp4.

The video fixture is a 100-frame 160x120 25 fps testsrc2 encode (same
command as tests/test_video_io.py). A minimal EPIC_100_video_info.csv is
written into a tmp dir so the Dataset's fps lookup works without touching
/mnt/ssd — the dataset reads fps from that CSV, and only the FRAME indices
matter here, so the declared fps (25) matches the fixture on purpose.
"""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.annotations import verb_label_map
from src.data.dataset import EpicVerbDataset, default_transform

VIDEO_ID = "P01_01"  # participant prefix P01 -> <root>/P01/videos/P01_01.MP4


def make_tiny_mp4(path) -> str:
    """100-frame 160x120 25 fps mp4 at path; returns the str path."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return str(path)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """(videos_root, annotations_dir) with one tiny video + its fps CSV."""
    tmp = tmp_path_factory.mktemp("dataset")
    video = tmp / "P01" / "videos" / f"{VIDEO_ID}.MP4"
    video.parent.mkdir(parents=True)
    make_tiny_mp4(video)
    annotations = tmp / "annotations"
    annotations.mkdir()
    pd.DataFrame({
        "video_id": [VIDEO_ID],
        "duration": [4.0],
        "fps": [25.0],
        "resolution": ["160x120"],
    }).to_csv(annotations / "EPIC_100_video_info.csv", index=False)
    return tmp, annotations


def make_segments(n_rows: int = 3) -> pd.DataFrame:
    """Fake segments table inside the 100-frame video (contract columns)."""
    df = pd.DataFrame({  # always built with 3 rows ...
        "video_id": [VIDEO_ID] * 3,
        "narration_id": [f"{VIDEO_ID}_{i}" for i in range(3)],
        "start_frame": [0, 10, 40],
        "stop_frame": [99, 30, 80],
        "verb": ["take", "put", "take"],
        "verb_class": [0, 1, 0],
    })
    return df.iloc[:n_rows].reset_index(drop=True)  # ... then truncated


VERB_MAP = verb_label_map(["take", "put"])  # alphabetical: put=0, take=1


def test_item_shapes_and_range(env):
    """__getitem__ -> clip (T,C,H,W) float32 in [0,1]; label matches map."""
    videos_root, annotations = env
    ds = EpicVerbDataset(annotations, videos_root, make_segments(),
                         VERB_MAP, T=4)
    clip, label = ds[0]
    assert clip.shape == (4, 3, 120, 160)
    assert clip.dtype == torch.float32
    assert 0.0 <= float(clip.min()) and float(clip.max()) <= 1.0
    assert label == VERB_MAP["take"]  # row 0 verb is 'take'
    _, label1 = ds[1]
    assert label1 == VERB_MAP["put"]


def test_labels_match_verb_map(env):
    """Every row's label equals verb_map[verb]."""
    videos_root, annotations = env
    ds = EpicVerbDataset(annotations, videos_root, make_segments(),
                         VERB_MAP, T=4)
    for i in range(len(ds)):
        verb = ds.segments_df.iloc[i]["verb"]
        _, label = ds[i]
        assert label == VERB_MAP[verb]


def test_missing_video_rows_filtered_with_warning(env):
    """Rows whose video file is absent are dropped at init + warn once."""
    videos_root, annotations = env
    df = make_segments(3)
    # P99_99 does not exist under the tmp root; P22_90 neither.
    extra = pd.DataFrame({
        "video_id": ["P99_99", "P22_90"],
        "narration_id": ["P99_99_0", "P22_90_0"],
        "start_frame": [0, 5],
        "stop_frame": [50, 60],
        "verb": ["take", "put"],
        "verb_class": [0, 1],
    })
    df = pd.concat([df, extra], ignore_index=True)

    with pytest.warns(UserWarning, match="dropped 2/5"):
        ds = EpicVerbDataset(annotations, videos_root, df, VERB_MAP, T=4)
    assert len(ds) == 3
    assert set(ds.segments_df["video_id"]) == {VIDEO_ID}


def test_dataloader_batches(env):
    """DataLoader(batch_size=2) stacks items into (B, T, C, H, W)."""
    videos_root, annotations = env
    ds = EpicVerbDataset(annotations, videos_root, make_segments(3),
                         VERB_MAP, T=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2  # 3 items -> ceil(3/2)
    first_clip, first_labels = batches[0]
    assert first_clip.shape == (2, 4, 3, 120, 160)
    assert first_clip.dtype == torch.float32
    assert first_labels.tolist() == [VERB_MAP["take"], VERB_MAP["put"]]


def test_img_size_resizes_short_side(env):
    """img_size=64 shrinks the short side (120) to 64, aspect preserved."""
    videos_root, annotations = env
    ds = EpicVerbDataset(annotations, videos_root, make_segments(1),
                         VERB_MAP, T=2, img_size=64)
    clip, _ = ds[0]
    assert clip.shape == (2, 3, 64, 85)  # 120:160 -> 64:85 (85.33 rounds)


def test_default_transform_standalone():
    """default_transform alone: (T,H,W,3) uint8 -> (T,3,H,W) float32 [0,1]."""
    import numpy as np
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    frames[..., 0] = 255  # "red" frames
    clip = default_transform()(frames)
    assert clip.shape == (2, 3, 8, 10)
    assert clip.dtype == torch.float32
    assert torch.allclose(clip[:, 0], torch.ones(2, 8, 10))  # red channel = 1
