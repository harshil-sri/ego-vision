"""Extract frozen DINOv2 features once for every annotated segment on disk.

For each action segment of each downloaded video:
    sample T frames (registry frame sampler, same as training)
    -> resize short side to 224 -> DINOv2 ViT-S/14 -> (T, 768)
Also stores the horizontally flipped version (augmentation for the head).

ALL verbs are extracted (not only the 8 demo verbs), so later experiments
(background class, more verbs) need no second pass over the videos.

Outputs in --out:
    feats.npy       (N, T, 768) float16   original frames
    feats_flip.npy  (N, T, 768) float16   mirrored frames
    meta.csv        N rows: narration_id, video_id, participant_id,
                    start_frame, stop_frame, verb, verb_class, noun, ok
    info.json       T, model, counts, wall time

Usage (Windows, from the repo root):
    python -m scripts.extract_features --device cuda
    python -m scripts.extract_features --encoder videomae --device cuda
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

META_COLUMNS = ["narration_id", "video_id", "participant_id", "start_frame",
                "stop_frame", "verb", "verb_class", "noun"]


class SegmentFrames(torch.utils.data.Dataset):
    """One item = the T resized frames of one annotated segment.

    __getitem__(i) -> (frames uint8 (T, h, w, 3) RGB, i, ok)
    A segment whose frames cannot be decoded returns zeros and ok = 0
    (the run never dies on one bad file; the row is flagged in meta.csv).
    """

    def __init__(self, df: pd.DataFrame, videos_root: str, fps: dict, T: int,
                 out_hw: tuple[int, int]) -> None:
        self.df, self.root, self.fps, self.T = df, Path(videos_root), fps, T
        self.out_hw = out_hw

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from src import registry
        from src.data.video_io import decode_frames
        from src.models.frozen import resize_frames

        row = self.df.iloc[i]
        vid = str(row["video_id"])
        path = self.root / vid.split("_")[0] / "videos" / f"{vid}.MP4"
        try:
            idx = registry.get("frame_sampler").sample_indices(
                int(row["start_frame"]), int(row["stop_frame"]), self.T,
                self.fps.get(vid, 50.0), stop_inclusive=True)   # (T,) int64
            frames = decode_frames(str(path), idx, fast_seek=True)  # (T,H,W,3)
            frames = resize_frames(frames)                       # (T,h,w,3)
            if frames.shape[1:3] != self.out_hw:                 # odd aspect
                import cv2
                frames = np.stack([cv2.resize(f, self.out_hw[::-1]) for f in frames])
            return torch.from_numpy(frames), i, 1
        except Exception as exc:  # noqa: BLE001 - flag and continue
            print(f"[skip] {row['narration_id']}: {exc}", flush=True)
            h, w = self.out_hw
            return torch.zeros((self.T, h, w, 3), dtype=torch.uint8), i, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2", choices=["dinov2", "videomae"])
    ap.add_argument("--out", default=None, help="default: features/<encoder>_T16 on the data disk")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4,
                    help="decode processes; each one costs about 1.5 GB of commit memory on Windows")
    ap.add_argument("--batch", type=int, default=8, help="segments per batch")
    ap.add_argument("--limit", type=int, default=None, help="first N segments (smoke test)")
    args = ap.parse_args()

    from src.data.annotations import load_video_info
    from src.models.frozen import DinoFrameEncoder, VideoMAEEncoder, target_size

    if args.out is None:
        name = "dinov2_vits14_T16" if args.encoder == "dinov2" else "videomae_ssv2_T16"
        args.out = f"/mnt/ssd/epic-kitchens/features/{name}"

    data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    ann = Path(resolve_data_path(data_cfg["annotations_dir"]))
    root = resolve_data_path(data_cfg["downscaled_root"])
    out = Path(resolve_data_path(args.out))
    out.mkdir(parents=True, exist_ok=True)

    on_disk = {p.stem for p in Path(root).glob("P*/videos/*.MP4")}
    df = pd.read_csv(ann / "EPIC_100_train.csv")
    df = df[df["video_id"].isin(on_disk)].dropna(subset=["start_frame", "stop_frame"])
    df = df.sort_values(["video_id", "start_frame"]).reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    fps = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    N, T = len(df), args.T
    hw = target_size(256, 456)   # EPIC-256 frames are 256x456 -> (224, 392)
    print(f"{N} segments from {df['video_id'].nunique()} videos, T={T}, input {hw}", flush=True)

    video = args.encoder == "videomae"
    if video:
        # One vector per clip and per crop (left, centre, right 224-wide views).
        enc = VideoMAEEncoder(device=args.device)
        crops = [0, (hw[1] - 224) // 2, hw[1] - 224]
        shape = (N, len(crops), enc.dim)
    else:
        enc = DinoFrameEncoder(device=args.device)
        shape = (N, T, enc.dim)
    feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=shape)
    flips = np.lib.format.open_memmap(out / "feats_flip.npy", mode="w+", dtype=np.float16, shape=shape)
    ok_flags = np.zeros(N, dtype=np.int8)

    loader = torch.utils.data.DataLoader(
        SegmentFrames(df, root, fps, T, hw), batch_size=args.batch,
        shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0)
    t0, done = time.time(), 0
    for frames, idx, ok in loader:                 # frames (B, T, h, w, 3) uint8
        B = frames.shape[0]
        if video:
            mirrored = torch.flip(frames, dims=[3])              # flip width axis
            f = torch.stack([enc.encode(frames[:, :, :, x:x + 224]) for x in crops], dim=1)   # (B,3,768)
            g = torch.stack([enc.encode(mirrored[:, :, :, x:x + 224]) for x in crops], dim=1)
        else:
            flat = frames.reshape(B * T, *frames.shape[2:])      # (B*T, h, w, 3)
            f = enc.encode(flat).reshape(B, T, enc.dim)          # (B, T, 768)
            g = enc.encode(torch.flip(flat, dims=[2])).reshape(B, T, enc.dim)
        ii = idx.numpy()
        feats[ii] = f.numpy().astype(np.float16)
        flips[ii] = g.numpy().astype(np.float16)
        ok_flags[ii] = ok.numpy()
        done += B
        if done % 400 < B:
            rate = done / (time.time() - t0)
            print(f"{done}/{N}  {rate:.1f} seg/s  eta {(N - done) / rate / 60:.1f} min", flush=True)
    feats.flush(); flips.flush()

    meta = df[META_COLUMNS].copy()
    meta["ok"] = ok_flags
    meta.to_csv(out / "meta.csv", index=False)
    bad = int(N - ok_flags.sum())
    if bad > 0.01 * N:   # a few broken clips are normal; many means the run itself failed
        raise SystemExit(f"{bad} of {N} segments failed to decode. Features NOT usable. "
                         "Common cause on Windows: too many workers for the free memory "
                         "(each worker loads the CUDA libraries). Retry with fewer --workers.")
    info = {"model": "videomae_base_ssv2" if video else "dinov2_vits14",
            "feature": "mean_token per crop (left, centre, right)" if video else "cls|patch_mean",
            "dim": enc.dim,
            "T": T, "input_hw": list(hw), "segments": int(N),
            "ok": int(ok_flags.sum()), "videos": int(df["video_id"].nunique()),
            "wall_seconds": round(time.time() - t0, 1), "videos_root": root}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
