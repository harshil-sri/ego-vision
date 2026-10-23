"""Record labelled webcam clips for the evaluation-only domain-gap check.

Records --clips-per-verb clips per verb into data/webcam_eval/<verb>/ using
src.capture.CameraCapture (MJPG 640x480 @ 30 requested), saving JPG frames
at ~10 fps (every 3rd captured frame) plus one row per clip in
data/webcam_eval/clips.csv. These clips are EVALUATION ONLY — never
training data (brief section 11).

--synthetic writes deterministic noise JPGs with the identical directory /
CSV layout, so the downstream eval path is testable without hardware.

Usage:
    python -m scripts.record_verb_clips                       # webcam
    python -m scripts.record_verb_clips --synthetic           # smoke
    python -m scripts.record_verb_clips --verbs take,put --clips-per-verb 5
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]
DEFAULT_OUT_ROOT = Path("data/webcam_eval")
#: frames kept per captured second (capture requests 30 fps; keep ~10)
SAVE_EVERY_NTH = 3


def _append_clip_row(csv_path: Path, row: dict) -> None:
    """Append one clip row to clips.csv (header written on first use).

    Input: csv_path Path; row dict with the CLI_COLUMNS keys below.
    Output: None (file grows by one line).
    """
    cli_columns = ["verb", "clip_id", "directory", "n_frames",
                   "source", "recorded_at"]
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cli_columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def record_synthetic_clip(out_dir: Path, verb: str, clip_id: int,
                          n_frames: int, width: int, height: int,
                          rng: np.random.Generator) -> int:
    """Write one clip of deterministic noise JPGs (no hardware).

    Inputs:
        out_dir: clip directory (created); verb/clip_id only label the files.
        n_frames: frames to write (≈ seconds * save_fps).
        width, height: JPG size.
        rng: np.random.Generator stream for THIS clip (determinism).

    Output: int n_frames written; files <verb>_c<clip_id>_<i:03d>.jpg.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        # (H, W, 3) uint8 RGB-ish noise; cv2.imwrite treats it as BGR, which
        # is irrelevant for smoke data (content carries no signal by design).
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = out_dir / f"{verb}_c{clip_id:02d}_{i:03d}.jpg"
        cv2.imwrite(str(path), frame)
    return n_frames


def record_webcam_clip(capture, out_dir: Path, verb: str, clip_id: int,
                       seconds: float) -> int:
    """Record one clip from an ALREADY-OPEN CameraCapture.

    Inputs:
        capture: src.capture.CameraCapture (opened by the caller).
        out_dir: clip directory (created).
        seconds: wall-clock seconds to record.
        verb, clip_id: only used for file naming.

    Output: int number of SAVED frames (~10/s; every SAVE_EVERY_NTH read).
    """
    import time as _time

    out_dir.mkdir(parents=True, exist_ok=True)
    saved, read_total, dropped_reads = 0, 0, 0
    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        packet = capture.read()  # FramePacket | None
        if packet is None:  # transient read failure; count, do not crash
            dropped_reads += 1
            continue
        read_total += 1
        if read_total % SAVE_EVERY_NTH != 0:  # 30 fps capture -> ~10 saved
            continue
        frame_bgr = packet.image                     # (H, W, 3) uint8 BGR
        path = out_dir / f"{verb}_c{clip_id:02d}_{saved:03d}.jpg"
        cv2.imwrite(str(path), frame_bgr)            # imwrite wants BGR
        saved += 1
    if dropped_reads:
        print(f"  note: {dropped_reads} failed camera reads during this clip")
    return saved


def main() -> None:
    """CLI entry point. Outputs: JPGs under <out-root>/<verb>/ and clip rows
    appended to <out-root>/clips.csv; prints progress per clip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbs", type=str,
                        default=",".join(DEFAULT_VERBS),
                        help="comma-separated verb list to record")
    parser.add_argument("--clips-per-verb", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="seconds per clip")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--synthetic", action="store_true",
                        help="write deterministic noise JPGs (no webcam)")
    parser.add_argument("--source", type=str, default="/dev/video0",
                        help="camera source for the real path")
    parser.add_argument("--width", type=int, default=640,
                        help="requested capture width (MJPG)")
    parser.add_argument("--height", type=int, default=480,
                        help="requested capture height (MJPG)")
    parser.add_argument("--fps", type=int, default=30,
                        help="requested capture fps")
    parser.add_argument("--pixel-format", type=str, default="MJPG",
                        help="requested pixel format (WSL2 needs MJPG)")
    parser.add_argument("--save-fps", type=int, default=10,
                        help="JPG frames saved per second (~10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="rng seed for --synthetic")
    args = parser.parse_args()

    verbs = [v.strip() for v in args.verbs.split(",") if v.strip()]
    csv_path = args.out_root / "clips.csv"

    if args.synthetic:
        # Deterministic per-clip rng streams: [seed, crc32(verb), clip_id]
        # (crc32, NOT hash(): python's hash() is salted per process).
        import zlib
        for verb in verbs:
            for clip_id in range(args.clips_per_verb):
                rng = np.random.default_rng(
                    [args.seed, zlib.crc32(verb.encode("utf-8")), clip_id])
                out_dir = args.out_root / verb
                n = record_synthetic_clip(out_dir, verb, clip_id,
                                          n_frames=int(round(
                                              args.seconds * args.save_fps)),
                                          width=args.width, height=args.height,
                                          rng=rng)
                _append_clip_row(csv_path, {
                    "verb": verb, "clip_id": clip_id, "directory": str(out_dir),
                    "n_frames": n, "source": "synthetic",
                    "recorded_at": datetime.now().isoformat(timespec="seconds")})
                print(f"[synthetic] {verb} clip {clip_id}: {n} jpgs -> {out_dir}")
        print(f"clips index: {csv_path}")
        return

    # Real webcam path (lazy import: never loads capture code for --synthetic).
    from src.capture import CameraCapture, CameraConfig

    config = CameraConfig(source=args.source, width=args.width,
                          height=args.height, fps=args.fps,
                          pixel_format=args.pixel_format)
    print(f"opening camera {args.source} at "
          f"{args.width}x{args.height}@{args.fps} {args.pixel_format}")
    with CameraCapture(config) as capture:
        print(f"camera reports {capture.actual_width}x{capture.actual_height} "
              f"@ {capture.reported_fps} fps")
        for verb in verbs:
            for clip_id in range(args.clips_per_verb):
                input(f"Press Enter to record {args.seconds:.0f}s of "
                      f"'{verb}' (clip {clip_id + 1}/{args.clips_per_verb})... ")
                out_dir = args.out_root / verb
                n = record_webcam_clip(capture, out_dir, verb, clip_id,
                                       seconds=args.seconds)
                _append_clip_row(csv_path, {
                    "verb": verb, "clip_id": clip_id, "directory": str(out_dir),
                    "n_frames": n, "source": "webcam",
                    "recorded_at": datetime.now().isoformat(timespec="seconds")})
                print(f"  saved {n} jpgs -> {out_dir}")
    print(f"clips index: {csv_path}")


if __name__ == "__main__":
    main()
