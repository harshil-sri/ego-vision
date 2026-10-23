"""Tests for src/data/video_io.py against a tiny synthetic mp4.

The fixture video is generated with ffmpeg's testsrc2 source:
    testsrc2=size=160x120:rate=25:duration=4 -> exactly 100 frames, 160x120,
    libx264 yuv420p (B-frames included, like real EPIC files). testsrc2
    contains strongly colored regions, which the RGB-order test needs.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from src.data.video_io import decode_frames, probe_fps, probe_frame_count


def make_tiny_mp4(path) -> str:
    """Create a 100-frame 160x120 25 fps mp4 at path; return str path.

    Input: path (pytest tmp_path / "tiny.mp4"). Output: the path as str.
    Uses ffmpeg lavfi testsrc2 -> deterministic content; x264 defaults add
    B-frames, so this exercises the same decode paths as real videos.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return str(path)


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory):
    """One shared 100-frame video for this module (generation is ~1 s)."""
    return make_tiny_mp4(tmp_path_factory.mktemp("video_io") / "tiny.mp4")


def test_decode_single_indices(tiny_mp4):
    """Each single-index decode returns (1, H, W, 3) uint8."""
    for i in [0, 10, 99]:
        frames = decode_frames(tiny_mp4, np.asarray([i], dtype=np.int64))
        assert frames.shape == (1, 120, 160, 3)
        assert frames.dtype == np.uint8


def test_decode_batch_shape(tiny_mp4):
    """A (T,) index array decodes to (T, H, W, 3) in the same order."""
    idx = np.asarray([0, 10, 99], dtype=np.int64)
    frames = decode_frames(tiny_mp4, idx)
    assert frames.shape == (3, 120, 160, 3)
    assert frames.dtype == np.uint8
    # order preserved: row k equals decoding index k alone
    for k, i in enumerate(idx):
        alone = decode_frames(tiny_mp4, np.asarray([i], dtype=np.int64))
        assert np.array_equal(frames[k], alone[0])


def test_decode_rgb_not_bgr(tiny_mp4):
    """testsrc2 has saturated colored regions: R and G channels differ."""
    frames = decode_frames(tiny_mp4, np.asarray([5], dtype=np.int64))
    frame = frames[0].astype(np.int16)  # (H, W, 3)
    # somewhere in the frame the red channel dominates green by a wide
    # margin (testsrc2's color bars/blocks); a BGR swap would break this.
    assert np.abs(frame[..., 0] - frame[..., 1]).max() > 30


def test_decode_frames_differ_over_time(tiny_mp4):
    """testsrc2 animates: frame 0 and frame 90 are not identical."""
    frames = decode_frames(tiny_mp4, np.asarray([0, 90], dtype=np.int64))
    assert frames[0].astype(np.int16).mean() != frames[1].astype(np.int16).mean()
    assert np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).mean() > 1


def test_fast_seek_returns_same_frames_within_tolerance(tiny_mp4):
    """fast_seek mode gives (T,H,W,3) uint8 close to the sequential decode.

    B-frames can make frame-number seeks land slightly off on some files
    (documented risk); on this x264/yuv420p mp4 the two modes must agree to
    within a small tolerance.
    """
    idx = np.asarray([3, 40, 77], dtype=np.int64)
    seq = decode_frames(tiny_mp4, idx, fast_seek=False)
    fast = decode_frames(tiny_mp4, idx, fast_seek=True)
    assert fast.shape == seq.shape and fast.dtype == np.uint8
    assert np.abs(seq.astype(np.int16) - fast.astype(np.int16)).mean() < 10


def test_missing_file_raises_filenotfound(tmp_path):
    """A missing video raises FileNotFoundError with the path in the message."""
    missing = tmp_path / "P99" / "videos" / "P99_99.MP4"
    with pytest.raises(FileNotFoundError, match="P99_99"):
        decode_frames(str(missing), np.asarray([0], dtype=np.int64))


def test_index_beyond_end_raises_runtimeerror(tiny_mp4):
    """Requesting frame 150 of a 100-frame video names video and index."""
    with pytest.raises(RuntimeError, match="150"):
        decode_frames(tiny_mp4, np.asarray([150], dtype=np.int64))


def test_negative_index_rejected(tiny_mp4):
    with pytest.raises(ValueError):
        decode_frames(tiny_mp4, np.asarray([-1], dtype=np.int64))


def test_probe_fps(tiny_mp4):
    """fps of the fixture is 25 (rate=25 in the lavfi source)."""
    fps = probe_fps(tiny_mp4)
    assert abs(fps - 25.0) < 0.5


def test_probe_frame_count(tiny_mp4):
    """Header frame count of the fixture is exactly 100."""
    assert probe_frame_count(tiny_mp4) == 100
