"""Downscale complete EPIC videos to short-side 256, preserving every frame.

Why: decode cost at 1080p dominates the data loader (D8). The downscaled
copies keep the SAME fps and frame count as the originals (no -r flag, no
frame dropping), so annotation start/stop frame indices remain valid against
either root; only pixel dimensions change (1920x1080 -> 456x256).

Idempotent: existing outputs are skipped, so the script can be rerun as the
bulk download lands more videos. Only files matching P*/videos/*.MP4 are
considered; the downloader's partial files (.incomplete, under .cache/) never
match that pattern, and --only-complete additionally ffprobe-checks each
input before spending encode time on it.

Usage (from repo root):
    python -m scripts.downscale \
        --videos-root /mnt/ssd/epic-kitchens/EPIC-KITCHENS \
        --out-root    /mnt/ssd/epic-kitchens/EPIC-256 \
        --crf 23 --jobs 2 --only-complete

Verification: for the FIRST video transcoded in a run, fps, frame count
(ffprobe nb_frames, with a -count_frames fallback when the header lies) and
size are compared before/after and appended to reports/downscale_check.md.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "reports" / "downscale_check.md"


def ffprobe_stream_field(path: Path, field: str, count_flag: str = "") -> str:
    """One ffprobe stream field as a stripped string ("" on failure).

    Inputs: path to a media file; field like "nb_frames" or "r_frame_rate";
        count_flag "" (header metadata) or "-count_frames"/"-count_packets"
        (slower, demuxes the file for the truth).
    Output: str, first line of ffprobe stdout.
    """
    cmd = ["ffprobe", "-v", "error", count_flag, "-select_streams", "v:0",
           "-show_entries", f"stream={field}",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    cmd = [c for c in cmd if c]  # drop the empty count_flag slot
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def frame_count(path: Path) -> int:
    """Best-effort frame count: header nb_frames, else -count_frames.

    Input: Path to a video. Output: int (0 if undeterminable).
    -count_frames demuxes/decodes the whole file (minutes for 1080p), so it
    is only the fallback for containers whose headers lie.
    """
    raw = ffprobe_stream_field(path, "nb_frames")
    try:
        n = int(raw)
        if n > 0:
            return n
    except ValueError:
        pass
    raw = ffprobe_stream_field(path, "nb_frames", count_flag="-count_frames")
    try:
        return int(raw)
    except ValueError:
        return 0


def fps_of(path: Path) -> str:
    """r_frame_rate as a string like '30000/1001' ("" on failure)."""
    return ffprobe_stream_field(path, "r_frame_rate")


def transcode(src: Path, dst: Path, crf: int) -> float:
    """ffmpeg transcode one video to short-side 256; return wall seconds.

    Inputs: src (existing .MP4), dst (output path, parents created), crf.
    Output: float seconds the ffmpeg subprocess took.

    The command deliberately has NO -r flag: fps is inherited from the
    source. scale=-2:256 sets the height to 256 and rounds the WIDTH to a
    multiple of 2 (libx264 needs even dimensions); -tune/preset veryfast
    trades compression ratio for encode speed. Audio (EPIC videos have
    none) would be stream-copied.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", "scale=-2:256",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-c:a", "copy",
        str(dst),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        # Remove partial output so a rerun retries this video.
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src}: {result.stderr[-500:]}")
    return elapsed


def append_check_row(video_id: str, src: Path, dst: Path,
                     seconds: float, crf: int) -> None:
    """Append one before/after verification row to reports/downscale_check.md.

    Inputs: the video id, original and downscaled paths, transcode wall
    time, crf used. Output: None (file created if absent).
    """
    src_fps, dst_fps = fps_of(src), fps_of(dst)
    src_n, dst_n = frame_count(src), frame_count(dst)
    ok = "OK" if (src_n == dst_n and src_n > 0 and src_fps == dst_fps) else "MISMATCH"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text(
            "# Downscale verification (first video of each run)\n\n"
            "Frame count via ffprobe nb_frames header (fallback\n"
            "-count_frames). Requirement: fps and frame count identical to\n"
            "the original so annotation indices stay valid.\n\n"
            "| date | video | crf | orig fps | new fps | orig frames | new "
            "frames | orig MB | new MB | encode s | status |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    row = (f"| {time.strftime('%Y-%m-%d %H:%M')} | {video_id} | {crf} "
           f"| {src_fps} | {dst_fps} | {src_n} | {dst_n} "
           f"| {src.stat().st_size / 1e6:.0f} | {dst.stat().st_size / 1e6:.0f} "
           f"| {seconds:.0f} | {ok} |")
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", required=True,
                        help="root with <Pxx>/videos/<id>.MP4 originals")
    parser.add_argument("--out-root", required=True,
                        help="destination root (same layout is recreated)")
    parser.add_argument("--crf", type=int, default=23,
                        help="libx264 CRF (default 23; higher = smaller)")
    parser.add_argument("--jobs", type=int, default=2,
                        help="parallel ffmpeg processes (default 2)")
    parser.add_argument("--only-complete", action="store_true",
                        help="ffprobe each input first and skip anything "
                             "that does not parse (guards against files "
                             "still being written)")
    args = parser.parse_args()

    videos_root, out_root = Path(args.videos_root), Path(args.out_root)
    # Smallest first: the verification row and the first usable outputs
    # appear as early as possible, and a partial run leaves most work done.
    sources = sorted(videos_root.glob("P*/videos/*.MP4"), key=lambda p: p.stat().st_size)

    todo: list[tuple[str, Path, Path]] = []
    for src in sources:
        if ".incomplete" in src.name:
            continue  # never happens for this glob, but be explicit
        video_id = src.stem
        dst = out_root / src.relative_to(videos_root)  # same layout
        if dst.exists():
            print(f"skip (exists): {video_id}")
            continue
        if args.only_complete and frame_count(src) <= 0:
            print(f"skip (unparsable/incomplete): {src}")
            continue
        todo.append((video_id, src, dst))

    if not todo:
        print(f"nothing to do: {len(sources)} sources found, all outputs "
              f"exist or were skipped ({videos_root})")
        return 0

    print(f"transcoding {len(todo)} video(s) with jobs={args.jobs}, "
          f"crf={args.crf}")
    checked = False
    failures: list[str] = []

    def job(item):
        video_id, src, dst = item
        seconds = transcode(src, dst, args.crf)
        return video_id, src, dst, seconds

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(job, item): item for item in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                video_id, src, dst, seconds = fut.result()
            except Exception as exc:  # one bad video must not kill the run
                failures.append(f"{item[0]}: {exc}")
                print(f"FAILED {item[0]}: {exc}")
                continue
            print(f"done {video_id}: {seconds:.0f}s, "
                  f"{dst.stat().st_size / 1e6:.0f} MB")
            if not checked:
                # Verify ONLY the first completed video (cheap enough to
                # repeat per run; the rest share the same ffmpeg command).
                append_check_row(video_id, src, dst, seconds, args.crf)
                checked = True

    if failures:
        print(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
