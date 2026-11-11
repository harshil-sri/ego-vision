"""Extract frozen VideoMAE features for SLIDING windows over whole videos.

scripts.extract_features cuts one clip per annotated action. The live app
never sees such clips: it sees "the last W seconds" at every tick. This
script produces features for exactly those windows, so a head can be
trained and tested on what the app will show it ("train like you serve").

For every video, every tick and every window length in --windows:
    frames  16 frames spread over the last W seconds, picked by the
            registry frame sampler (same rule as SlidingWindowBuffer)
    feature mean of the VideoMAE vectors of the left, centre and right
            224-wide crops (same as VideoMAEEncoder.encode_window)
Ticks: one every --tick seconds on training videos, one every
--heldout-tick seconds on the held-out participants (denser, for
evaluation). All window lengths share the same ticks.

Decoding: each worker process walks its videos once from the first frame.
Frames that no window needs are only grab()bed (cheap); needed frames are
retrieve()d and resized.

Outputs in --out:
    feats_<W>.npy  (N, 768) float16, one file per window length
    meta.csv       N rows: video_id, participant_id, frame (last frame of
                   the window), fps
    info.json      windows, ticks, counts, wall time

Usage:
    python -m scripts.extract_sliding --windows 2.0 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import yaml

from src.paths import resolve_data_path

T = 16


class WindowStream(torch.utils.data.IterableDataset):
    """Yields (clips uint8 (W, 16, 224, 392, 3) RGB, video number, frame) per tick,
    one clip per window length."""

    def __init__(self, videos: list[Path], fps: list[float], windows: list[float],
                 ticks: list[float], hw: tuple[int, int]) -> None:
        """ticks: seconds between windows, one value PER VIDEO."""
        self.videos, self.fps, self.windows, self.ticks, self.hw = videos, fps, windows, ticks, hw

    def __iter__(self):
        import cv2
        from src import registry
        sampler = registry.get("frame_sampler")
        info = torch.utils.data.get_worker_info()
        mine = range(len(self.videos)) if info is None else \
            range(info.id, len(self.videos), info.num_workers)
        h, w = self.hw
        for v in mine:
            fps = self.fps[v]
            wlens = [int(round(w_ * fps)) for w_ in self.windows]
            wlen, step = max(wlens), int(round(self.ticks[v] * fps))
            cap = cv2.VideoCapture(str(self.videos[v]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ticks = list(range(wlen, n, step))                     # last frame of each window
            plans = [[sampler.sample_indices(f - wl, f, T, fps, stop_inclusive=True) for wl in wlens]
                     for f in ticks]                               # per tick, per window: (16,) indices
            needed = set(int(i) for per_tick in plans for p in per_tick for i in p)
            keep: dict[int, np.ndarray] = {}
            k = 0
            for f in range(n):
                if not cap.grab():
                    break
                if f in needed:
                    ok, bgr = cap.retrieve()
                    if not ok:
                        break
                    keep[f] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1]
                while k < len(ticks) and ticks[k] == f:
                    if all(int(i) in keep for p in plans[k] for i in p):
                        clips = np.stack([np.stack([keep[int(i)] for i in p]) for p in plans[k]])  # (W,16,h,w,3)
                        yield torch.from_numpy(np.ascontiguousarray(clips)), v, f
                    k += 1
                    lo = ticks[k] - wlen if k < len(ticks) else n               # drop frames no later window needs
                    for old in [i for i in keep if i < lo]:
                        del keep[old]
            cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="+", default=[2.0, 1.0], help="window lengths in seconds")
    ap.add_argument("--tick", type=float, default=1.0, help="seconds between windows, training videos")
    ap.add_argument("--heldout-tick", type=float, default=0.5, help="the same for held-out participants")
    ap.add_argument("--participants", nargs="+", default=None, help="default: every participant on disk")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from src.data.annotations import load_video_info
    from src.models.frozen import VideoMAEEncoder, target_size

    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    ann = Path(resolve_data_path(cfg["annotations_dir"]))
    root = Path(resolve_data_path(cfg["downscaled_root"]))
    out = Path(resolve_data_path(args.out or "/mnt/ssd/epic-kitchens/features/videomae_ssv2_sliding"))
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(root.glob("P*/videos/*.MP4"))
    if args.participants:
        videos = [p for p in videos if p.stem.split("_")[0] in args.participants]
    fps_of = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    fps = [fps_of[p.stem] for p in videos]
    hw = target_size(256, 456)
    crops = [0, (hw[1] - 224) // 2, hw[1] - 224]
    held = set(cfg["heldout_participants"])
    ticks = [args.heldout_tick if p.stem.split("_")[0] in held else args.tick for p in videos]
    W = len(args.windows)
    print(f"{len(videos)} videos, windows {args.windows} s, tick {args.tick} s "
          f"(held-out {args.heldout_tick} s)", flush=True)

    enc = VideoMAEEncoder(device=args.device)
    loader = torch.utils.data.DataLoader(WindowStream(videos, fps, args.windows, ticks, hw),
                                         batch_size=args.batch, num_workers=args.workers)
    feats, vid, frame = [], [], []
    t0 = time.time()
    for clips, v, f in loader:                                    # clips (B, W, 16, 224, 392, 3)
        B = clips.shape[0]
        flat = clips.reshape(B * W, *clips.shape[2:])             # (B*W, 16, 224, 392, 3)
        x = torch.stack([enc.encode(flat[:, :, :, c:c + 224]) for c in crops]).mean(0)   # (B*W, 768)
        feats.append(x.reshape(B, W, -1).numpy().astype(np.float16)); vid += v.tolist(); frame += f.tolist()
        if len(frame) % 2000 < len(f):
            print(f"{len(frame)} ticks  {len(frame) / (time.time() - t0):.1f}/s", flush=True)
    order = np.lexsort((np.array(frame), np.array(vid)))          # workers interleave videos; sort back
    feats = np.concatenate(feats)[order]                          # (N, W, 768)
    meta = pd.DataFrame({"video_id": [videos[vid[i]].stem for i in order],
                         "frame": [frame[i] for i in order]})
    meta["participant_id"] = meta["video_id"].str.split("_").str[0]
    meta["fps"] = meta["video_id"].map(fps_of)
    for j, wl in enumerate(args.windows):
        np.save(out / f"feats_{wl}.npy", feats[:, j])
    meta.to_csv(out / "meta.csv", index=False)
    info = {"model": "videomae_base_ssv2", "feature": "mean of left/centre/right crop vectors",
            "windows_s": args.windows, "tick_s": args.tick, "heldout_tick_s": args.heldout_tick,
            "ticks": int(len(meta)),
            "videos": len(videos), "wall_seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
