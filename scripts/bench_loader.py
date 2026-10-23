"""Benchmark the two WP1 loading strategies on real downloaded videos (D8).

(a) decode-on-the-fly: EpicVerbDataset over downscaled (or original) mp4s —
    cv2 decodes T frames per __getitem__ (sequential grab() walk, see
    src/data/video_io.py).
(b) pre-extracted frames: dump each segment's T frames to JPEG once, then
    measure re-loading the JPEGs (cv2.imread) — the disk-cost trade-off.

Both modes run over the SAME --clips segments taken from --videos (or every
complete video under the chosen root). Output: reports/loader_benchmark.md
with clips/sec, MB/s and the disk cost of pre-extraction.

All numbers are measured on this machine at run time; nothing is estimated
except the decode-mode bytes-read (sequential decode touches the file from
byte 0 up to the last needed frame; per-clip bytes are approximated as
file_size * max_sampled_index / total_frames and labelled ESTIMATE).
A bulk download running concurrently inflates both paths; the report says
so when that is detected is NOT attempted — read the caveat line instead.

Usage:
    python -m scripts.bench_loader --clips 16
    python -m scripts.bench_loader --clips 32 --videos P01_107 P01_108
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src import registry
from src.data.annotations import load_segments, load_video_info, verb_label_map
from src.data.dataset import EpicVerbDataset
from src.data.splits import video_path
from src.data.video_io import decode_frames, probe_frame_count

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "data.yaml"
REPORT = REPO_ROOT / "reports" / "loader_benchmark.md"


def complete_videos(root: Path) -> list[str]:
    """Video ids of complete .MP4s under root (P*/videos layout).

    Input: root Path. Output: list[str] sorted, e.g. ["P01_107", ...].
    Complete = final-name file > 10 MB (the downloader's partials never
    appear under their final name).
    """
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("P*/videos/*.MP4")
                  if ".incomplete" not in p.name and p.stat().st_size > 10_000_000)


def time_dataset(dataset: EpicVerbDataset, n: int) -> tuple[float, int]:
    """Time n dataset items (1 uncounted warmup call first).

    Inputs: dataset, n items to time. Output: (elapsed_seconds, n_ok);
    items that raise are counted in n_ok's shortfall, not silently skipped.
    """
    dataset[0]  # warmup: page in cv2 libs, open one video, prime caches
    n_ok = 0
    t0 = time.perf_counter()
    for i in range(n):
        try:
            dataset[i % len(dataset)]
            n_ok += 1
        except Exception as exc:
            print(f"decode failed on item {i}: {exc}")
    return time.perf_counter() - t0, n_ok


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=int, default=16,
                        help="segments to measure per mode (default 16)")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="restrict to these video ids (default: all "
                             "complete videos under the chosen root)")
    parser.add_argument("--root", default=None,
                        help="video root override (default: downscaled root "
                             "if it has complete videos, else original)")
    parser.add_argument("--T", type=int, default=8,
                        help="frames per clip (default 8, D5 default)")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()

    downscaled_root = Path(cfg["downscaled_root"])
    videos_root = Path(cfg["videos_root"])

    # Prefer the downscaled copies (that is the D8 question being measured);
    # fall back to originals with a loud note when downscales do not exist.
    if args.root is not None:
        root = Path(args.root)
        root_kind = "explicit --root"
    elif complete_videos(downscaled_root):
        root, root_kind = downscaled_root, "downscaled (EPIC-256)"
    elif complete_videos(videos_root):
        root, root_kind = videos_root, "ORIGINAL 1080p (no downscaled videos yet)"
    else:
        msg = ("NOT RUN (no complete videos yet) under "
               f"{downscaled_root} or {videos_root}; rerun once the bulk "
               "download lands files (find <root> -name '*.MP4' -size +10M).")
        print(msg)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            f"# Loader benchmark\n\n{time.strftime('%Y-%m-%d %H:%M')} — {msg}\n",
            encoding="utf-8")
        return 0

    available = complete_videos(root)
    chosen = [v for v in (args.videos or available) if v in set(available)]
    if not chosen:
        print(f"none of --videos {args.videos} are complete under {root}")
        return 1

    # Same segment table for both modes: candidate-verb rows of the chosen
    # videos, first --clips of them.
    segments = load_segments(Path(cfg["annotations_dir"]) / "EPIC_100_train.csv",
                             verbs=cfg["candidate_verbs"],
                             video_ids=set(chosen))
    segments = segments.head(args.clips).reset_index(drop=True)
    n = len(segments)
    vmap = verb_label_map(list(cfg["candidate_verbs"]))
    sampler = registry.get("frame_sampler")
    info = load_video_info(cfg["annotations_dir"])

    lines = [
        "# Loader benchmark (D8 evidence: decode-on-the-fly vs pre-extracted JPEGs)",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- command: python -m scripts.bench_loader --clips {n} "
        f"--videos {' '.join(chosen) if args.videos else '(all complete)'} "
        f"--T {args.T}",
        f"- root: {root} ({root_kind})",
        f"- videos used: {', '.join(chosen)}",
        f"- segments measured: {n} (candidate verbs "
        f"{cfg['candidate_verbs']}), T={args.T}, single process "
        f"(dataset[i] loop, num_workers=0), 1 warmup call excluded",
        f"- caveat: numbers reflect this machine's load at run time (a "
        f"concurrent bulk download was active when this benchmark was "
        f"first run; CPU+disk contention inflates both modes).",
        "",
    ]

    # ---- mode (a): decode on the fly -------------------------------------
    dataset = EpicVerbDataset(cfg["annotations_dir"], root, segments, vmap, T=args.T)
    elapsed_a, ok_a = time_dataset(dataset, n)
    clips_per_sec_a = ok_a / elapsed_a if elapsed_a > 0 else float("nan")

    # ESTIMATED bytes touched (sequential decode walks the file from 0):
    # file_size * max_sampled_index / total_frames, summed over segments.
    est_bytes = 0.0
    for _, row in segments.iterrows():
        vid = row["video_id"]
        vfile = video_path(root, vid)
        total = probe_frame_count(vfile)
        fps = float(info.loc[vid, "fps"]) if vid in info.index else 59.94
        idx = sampler.sample_indices(int(row["start_frame"]),
                                     int(row["stop_frame"]), args.T, fps)
        frac = (float(idx.max()) + 1.0) / total if total > 0 else 1.0
        est_bytes += vfile.stat().st_size * min(1.0, frac)
    est_mb_s = est_bytes / 1e6 / elapsed_a if elapsed_a > 0 else float("nan")

    # ---- mode (b): pre-extract JPEGs, then reload -------------------------
    import tempfile
    t0 = time.perf_counter()
    jpg_dirs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="epic_bench_jpg_") as tmp:
        tmp = Path(tmp)
        for _, row in segments.iterrows():
            vid = row["video_id"]
            vfile = video_path(root, vid)
            fps = float(info.loc[vid, "fps"]) if vid in info.index else 59.94
            idx = sampler.sample_indices(int(row["start_frame"]),
                                         int(row["stop_frame"]), args.T, fps)
            frames = decode_frames(str(vfile), idx)          # (T,H,W,3) RGB
            d = tmp / str(row["narration_id"])
            d.mkdir(parents=True, exist_ok=True)
            for t in range(args.T):
                # imwrite expects BGR; quality 95 = typical training JPEG.
                cv2.imwrite(str(d / f"{t:02d}.jpg"),
                            cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            jpg_dirs.append(d)
        extract_s = time.perf_counter() - t0
        jpg_bytes = sum(p.stat().st_size for d in jpg_dirs for p in d.iterdir())

        # Timed reload: imread T jpgs + stack (what a pre-extracted loader does).
        t0 = time.perf_counter()
        ok_b = 0
        for d in jpg_dirs:
            try:
                imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                        for p in sorted(d.glob("*.jpg"))]
                np.stack(imgs)  # (T,H,W,3) uint8, same as decode output
                ok_b += 1
            except Exception as exc:
                print(f"jpg load failed on {d.name}: {exc}")
        elapsed_b = time.perf_counter() - t0

    clips_per_sec_b = ok_b / elapsed_b if elapsed_b > 0 else float("nan")
    jpg_mb_s = (jpg_bytes * (ok_b / max(1, n))) / 1e6 / elapsed_b if elapsed_b > 0 else float("nan")

    lines += [
        "| mode | clips ok | clips/sec | disk MB/s | disk cost/clip |",
        "|---|---|---|---|---|",
        f"| (a) decode-on-the-fly ({root_kind.split(' ')[0].lower()}) "
        f"| {ok_a}/{n} | {clips_per_sec_a:.2f} | {est_mb_s:.1f} (ESTIMATE, "
        f"see note) | 0 extra |",
        f"| (b) pre-extracted JPEGs (reload only) | {ok_b}/{n} | "
        f"{clips_per_sec_b:.2f} | {jpg_mb_s:.1f} (measured jpg bytes read) "
        f"| {jpg_bytes / max(1, ok_b) / 1e6:.2f} MB |",
        "",
        f"- pre-extraction one-time cost: {extract_s:.1f} s for {n} clips, "
        f"{jpg_bytes / 1e6:.1f} MB total ({jpg_bytes / max(1, n) / 1e6:.2f} "
        f"MB/clip, i.e. {jpg_bytes / max(1, n) / 1e6:.2f} GB per 1000 clips)",
        f"- decode-mode MB/s is an ESTIMATE: sequential decode reads the "
        f"file from byte 0 up to the last needed frame, so bytes ~= "
        f"file_size * max_index / total_frames ({est_bytes / 1e6:.0f} MB "
        f"touched over {ok_a} clips); it is not instrumented with iostat.",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"(a) decode-on-the-fly : {clips_per_sec_a:.2f} clips/s ({ok_a}/{n})")
    print(f"(b) jpg reload        : {clips_per_sec_b:.2f} clips/s ({ok_b}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
