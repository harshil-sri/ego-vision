"""Frozen VideoMAE features for the downloaded HoloAssist videos.

One feature per second of video: the last --window seconds (16 frames,
three 224-wide crops averaged), exactly as scripts.extract_sliding does
for EPIC. These features are the "eyes" for pilot part 5: can a model SEE
the hand state that the pilot so far read from human annotations?

Outputs in <root>/features_w<window>/:
    feats.npy   (N, 768) float16
    meta.csv    N rows: video_name, frame, fps, t_end (s), t_mid (s)
                t_end = time of the window's last frame; t_mid = its centre
    info.json
    parts/<video>.npz   per-video cache; a repeated run only does new videos

Usage:
    python -m scripts.holoassist_video_features --window 2.0 --tick 1.0
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

from scripts.extract_sliding import WindowStream
from src.paths import resolve_data_path

ROOT = "/mnt/ssd/holoassist"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--tick", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--root", default=ROOT, help="dataset folder with videos/ (also used for Ego-EXTRA)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N videos, output in features_w<W>_smoke")
    args = ap.parse_args()

    import cv2
    from src.models.frozen import VideoMAEEncoder, target_size

    root = Path(resolve_data_path(args.root))
    out = root / (f"features_w{args.window}" + ("_smoke" if args.limit else ""))
    parts = out / "parts"                                 # one file per finished video: the run can be repeated
    parts.mkdir(parents=True, exist_ok=True)
    everything = sorted((root / "videos").rglob("*.mp4"))[:args.limit or None]
    videos = [p for p in everything if not (parts / f"{p.stem}.npz").exists()]
    t0 = time.time()
    if videos:
        fps, sizes = [], set()
        for p in videos:
            cap = cv2.VideoCapture(str(p))
            fps.append(float(cap.get(cv2.CAP_PROP_FPS)))
            sizes.add((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
            cap.release()
        if len(sizes) != 1:
            raise SystemExit(f"videos have different frame sizes: {sizes}")
        hw = target_size(*sizes.pop())
        crops = sorted({0, (hw[1] - 224) // 2, hw[1] - 224})
        print(f"{len(videos)} new videos ({len(everything) - len(videos)} done before), input {hw}, "
              f"fps {min(fps):.2f}..{max(fps):.2f}", flush=True)

        enc = VideoMAEEncoder(device=args.device)
        loader = torch.utils.data.DataLoader(
            WindowStream(videos, fps, [args.window], [args.tick] * len(videos), hw),
            batch_size=args.batch, num_workers=args.workers)
        feats, vid, frame = [], [], []
        for clips, v, f in loader:                                    # clips (B, 1, 16, h, w, 3)
            x = clips[:, 0]
            e = torch.stack([enc.encode(x[:, :, :, c:c + 224]) for c in crops]).mean(0)   # (B, 768)
            feats.append(e.numpy().astype(np.float16)); vid += v.tolist(); frame += f.tolist()
            if len(frame) % 4000 < len(f):
                print(f"{len(frame)} ticks  {len(frame) / (time.time() - t0):.1f}/s", flush=True)
        feats, vid, frame = np.concatenate(feats), np.array(vid), np.array(frame)
        for i, p in enumerate(videos):                    # a video with no tick still gets a (empty) part file
            m = vid == i
            o = np.argsort(frame[m])
            np.savez(parts / f"{p.stem}.npz", feats=feats[m][o], frame=frame[m][o], fps=fps[i])

    # merge all parts into one table, sorted by video then frame
    F, rows = [], []
    for p in sorted(parts.glob("*.npz")):
        z = np.load(p)
        F.append(z["feats"])
        rows += [(p.stem, int(fr), float(z["fps"])) for fr in z["frame"]]
    np.save(out / "feats.npy", np.concatenate(F))
    meta = pd.DataFrame(rows, columns=["video_name", "frame", "fps"])
    meta["t_end"] = meta["frame"] / meta["fps"]
    meta["t_mid"] = meta["t_end"] - args.window / 2
    meta.to_csv(out / "meta.csv", index=False)
    info = {"model": "videomae_base_ssv2", "window_s": args.window, "tick_s": args.tick,
            "ticks": int(len(meta)), "videos": int(meta["video_name"].nunique()),
            "wall_seconds_this_run": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
