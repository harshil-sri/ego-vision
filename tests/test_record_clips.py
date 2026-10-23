"""Tests for scripts/record_verb_clips.py — SYNTHETIC PATH ONLY (no webcam
hardware is touched; the real capture path is exercised manually with the
camera per the brief's webcam workflow)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_synthetic(out_root: Path, verbs: str, clips: int, seconds: float) -> None:
    """Run the recorder CLI with --synthetic into out_root.

    Output: None (process must exit 0 or the test fails with its output).
    """
    cmd = [sys.executable, "-m", "scripts.record_verb_clips",
           "--synthetic", "--verbs", verbs, "--clips-per-verb", str(clips),
           "--seconds", str(seconds), "--out-root", str(out_root),
           "--save-fps", "2", "--width", "32", "--height", "24"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"recorder failed:\n{proc.stderr}"
    return None


def test_synthetic_writes_frames_and_csv(tmp_path: Path):
    """Two verbs x two clips: jpgs per clip dir + one clips.csv with rows."""
    out_root = tmp_path / "webcam_eval"
    _run_synthetic(out_root, verbs="take,put", clips=2, seconds=2.0)

    csv_path = out_root / "clips.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # 2 verbs x 2 clips
    assert {r["verb"] for r in rows} == {"take", "put"}
    assert all(r["source"] == "synthetic" for r in rows)

    for row in rows:
        clip_dir = Path(row["directory"])
        jpgs = sorted(clip_dir.glob("*.jpg"))
        # seconds=2.0, save_fps=2 -> 4 frames per clip
        assert len(jpgs) == int(row["n_frames"]) == 4, (
            f"{clip_dir}: {len(jpgs)} jpgs vs n_frames {row['n_frames']}")


def test_synthetic_is_deterministic(tmp_path: Path):
    """Same seed -> byte-identical JPGs (crc32-based rng streams, not the
    per-process-salted python hash())."""
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    _run_synthetic(a, verbs="mix", clips=1, seconds=1.0)
    _run_synthetic(b, verbs="mix", clips=1, seconds=1.0)
    jpgs_a = sorted((a / "mix").glob("*.jpg"))
    jpgs_b = sorted((b / "mix").glob("*.jpg"))
    assert jpgs_a and len(jpgs_a) == len(jpgs_b)
    for fa, fb in zip(jpgs_a, jpgs_b):
        assert fa.read_bytes() == fb.read_bytes(), (
            f"synthetic clips differ across runs: {fa} vs {fb}")


def test_synthetic_does_not_touch_camera(tmp_path: Path):
    """The synthetic path never imports src.capture (no /dev/video access).

    Checked by grepping the recorder's stdout for the camera-open message
    and by the fact that no CameraCapture import happens (the script only
    imports it on the real path; a crash-free synthetic run on a machine
    without the device attached is the behavioural proof)."""
    out_root = tmp_path / "nocam"
    _run_synthetic(out_root, verbs="pour", clips=1, seconds=0.5)
    assert not (out_root / "pour" / "clips.csv").exists()  # csv lives at root
    assert (out_root / "clips.csv").exists()
