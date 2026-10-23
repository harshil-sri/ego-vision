"""Cut 5 annotated segments out of a real downloaded video for eyeballing.

Purpose (WP1 acceptance): produce short mp4s the owner can WATCH next to the
annotation rows, to confirm the stop_frame-inclusive convention and the
frame->second arithmetic on real data before training anything.

Method per segment:
    start_sec    = start_frame / fps
    duration_sec = (stop_frame - start_frame + 1) / fps   # +1 = INCLUSIVE
    ffmpeg -ss <start_sec> -i <video> -t <duration_sec> -c:v libx264 ...
-ss BEFORE -i uses input seeking with ffmpeg's default ACCURATE seek (it
seeks to the previous keyframe and decodes forward, discarding until the
exact timestamp — we do NOT pass -noaccurate_seek). Segments are re-encoded
(no -c copy) because stream-copying from mid-GOP is keyframe-inaccurate; no
-r flag, so fps is preserved.

Outputs (reports/annotation_check/):
    <narration_id>_<verb>.mp4   the cut clip
    labels.csv                  narration_id, video_id, verb, noun,
                                start_frame, stop_frame, frames_expected,
                                start_sec, duration_sec, frames_actual,
                                out_path
    NOT_RUN.md                  written only when no complete video exists.

Usage:
    python -m scripts.annotation_check            # smallest complete video
    python -m scripts.annotation_check --video P01_107
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

from src.data.annotations import load_video_info
from src.data.video_io import probe_fps

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "annotation_check"

LABEL_COLUMNS = ["narration_id", "video_id", "verb", "noun", "start_frame",
                 "stop_frame", "frames_expected", "start_sec",
                 "duration_sec", "frames_actual", "out_path"]

CANDIDATE_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]


def ffprobe_nb_frames(path: Path) -> int:
    """Frame count of a small clip via ffprobe nb_frames (0 on failure).

    Input: Path to a (short) video. Output: int.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return int(out.stdout.strip())
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return 0


def smallest_complete_video(videos_root: Path) -> Path | None:
    """Smallest fully-downloaded .MP4 under videos_root, or None.

    Inputs: videos_root with <Pxx>/videos/*.MP4 layout.
    Output: Path to the smallest file > 10 MB (bulk-download complete files
    are hundreds of MB; anything tiny is suspect), or None if there is none.
    """
    if not videos_root.exists():
        return None
    candidates = [p for p in videos_root.glob("P*/videos/*.MP4")
                  if ".incomplete" not in p.name and p.stat().st_size > 10_000_000]
    return min(candidates, key=lambda p: p.stat().st_size) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations",
                        default="/mnt/ssd/epic-tools/epic-kitchens-100-annotations")
    parser.add_argument("--videos-root",
                        default="/mnt/ssd/epic-kitchens/EPIC-KITCHENS")
    parser.add_argument("--video", default=None,
                        help="override, e.g. P01_107 (else smallest complete)")
    parser.add_argument("--n", type=int, default=5,
                        help="segments to cut (default 5)")
    args = parser.parse_args()

    annotations = Path(args.annotations)
    videos_root = Path(args.videos_root)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = REPORT_DIR / "labels.csv"

    if args.video is not None:
        src = videos_root / args.video.split("_")[0] / "videos" / f"{args.video}.MP4"
        if not src.exists():
            src = None
    else:
        src = smallest_complete_video(videos_root)

    if src is None:
        msg = ("NOT RUN: no complete video exists under "
               f"{videos_root} (bulk download still in progress; check "
               "with: find <root> -name '*.MP4' -size +10M)")
        print(msg)
        pd.DataFrame(columns=LABEL_COLUMNS).to_csv(labels_path, index=False)
        (REPORT_DIR / "NOT_RUN.md").write_text(
            f"# annotation_check: NOT RUN\n\n{time.strftime('%Y-%m-%d %H:%M')} "
            f"— {msg}\n", encoding="utf-8")
        return 0

    video_id = src.stem
    print(f"source video: {src} ({src.stat().st_size / 1e6:.0f} MB)")

    # fps from the file itself (ground truth), cross-checked against the
    # official metadata when the video is listed there.
    fps = probe_fps(str(src))
    info = load_video_info(annotations)
    meta_fps = float(info.loc[video_id, "fps"]) if video_id in info.index else float("nan")
    print(f"fps: file={fps:.4f}  video_info={meta_fps:.4f}")

    df = pd.read_csv(annotations / "EPIC_100_train.csv")
    rows = df[df["video_id"] == video_id]
    # Prefer candidate verbs (the classes we train); fall back to any verb.
    cand = rows[rows["verb"].isin(CANDIDATE_VERBS)]
    pool = cand if len(cand) >= args.n else rows
    # Spread the picks across the video instead of the first n neighbours.
    picks = pool.iloc[sorted(set(
        round(i * (len(pool) - 1) / max(1, args.n - 1)) for i in range(args.n)
    ))] if len(pool) > 1 else pool
    print(f"segments: {len(picks)} of {len(rows)} rows "
          f"({len(cand)} candidate-verb rows)")

    records = []
    for _, row in picks.iterrows():
        start_sec = float(row["start_frame"]) / fps
        duration_sec = (float(row["stop_frame"]) - float(row["start_frame"]) + 1.0) / fps
        # sanitize the verb for the filename (verbs like "turn-on" are fine;
        # slashes/space would not be)
        safe_verb = str(row["verb"]).replace("/", "-").replace(" ", "_")
        out = REPORT_DIR / f"{row['narration_id']}_{safe_verb}.mp4"

        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{start_sec:.6f}", "-i", str(src),
               "-t", f"{duration_sec:.6f}",
               "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
               "-an", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"ffmpeg failed on {row['narration_id']}: {result.stderr[-300:]}")
            continue

        frames_actual = ffprobe_nb_frames(out)
        records.append({
            "narration_id": row["narration_id"],
            "video_id": video_id,
            "verb": row["verb"],
            "noun": row["noun"],
            "start_frame": int(row["start_frame"]),
            "stop_frame": int(row["stop_frame"]),
            "frames_expected": int(row["stop_frame"]) - int(row["start_frame"]) + 1,
            "start_sec": round(start_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "frames_actual": frames_actual,
            "out_path": str(out),
        })
        print(f"cut {row['narration_id']} verb={row['verb']} "
              f"noun={row['noun']} frames {row['start_frame']}..{row['stop_frame']} "
              f"-> {out.name} (expected "
              f"{int(row['stop_frame']) - int(row['start_frame']) + 1}, "
              f"got {frames_actual})")

    pd.DataFrame(records, columns=LABEL_COLUMNS).to_csv(labels_path, index=False)
    print(f"wrote {labels_path} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
