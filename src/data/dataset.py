"""EpicVerbDataset: EPIC-KITCHENS segments -> (clip tensor, verb label) pairs.

Batching shape contract (what a DataLoader yields from this Dataset):
    single item:   clip torch.float32 (T, C, H, W), values in [0, 1], RGB
                   channel order; label python int (verb_map[verb])
    DataLoader:    stacks items into (B, T, C, H, W) float32 and (B,) int64
                   (with the default collate and a transform that returns a
                   fixed-size tensor — use img_size or a resize in transform
                   for that; without img_size, H/W are the video's native
                   size and all frames of one video match, but different
                   videos can differ, which breaks default collate).

Data path: <videos_root>/<Pxx>/videos/<video_id>.MP4 — pass either the
original root or the downscaled root (identical layout, identical frame
count, so annotation indices are valid against both).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data

from src import registry
from src.data.annotations import load_video_info
from src.data.video_io import decode_frames, probe_fps

REQUIRED_COLUMNS = ["video_id", "narration_id", "start_frame", "stop_frame",
                    "verb", "verb_class"]


def default_transform(img_size: int | None = None) -> Callable[[np.ndarray], torch.Tensor]:
    """Build the default frame-array -> clip-tensor transform.

    Input: img_size (int or None). If int, the clip's SHORT side is resized
        to img_size with cv2 (aspect preserved, e.g. 456x256 -> 256-side
        target 224 gives 224x398); if None, frames keep their native size.

    Output: a callable taking an np.ndarray (T, H, W, 3) uint8 RGB and
        returning torch.float32 (T, 3, H', W') in [0, 1] (RGB, CHW).
    """
    import cv2  # local import so importing this module stays cheap

    def _apply(frames: np.ndarray) -> torch.Tensor:
        # frames: (T, H, W, 3) uint8 RGB
        if img_size is not None:
            h, w = frames.shape[1], frames.shape[2]
            if h <= w:  # portrait/square: short side is h
                new_h, new_w = img_size, max(1, round(w * img_size / h))
            else:
                new_h, new_w = max(1, round(h * img_size / w)), img_size
            resized = np.empty((frames.shape[0], new_h, new_w, 3), dtype=np.uint8)
            for t in range(frames.shape[0]):  # plain loop: cv2 has no batch API
                resized[t] = cv2.resize(frames[t], (new_w, new_h),
                                        interpolation=cv2.INTER_AREA)
            frames = resized  # (T, new_h, new_w, 3) uint8

        clip = torch.from_numpy(frames.copy())       # (T, H, W, 3) uint8
        clip = clip.permute(0, 3, 1, 2).contiguous() # (T, 3, H, W) uint8
        return clip.float().div_(255.0)              # (T, 3, H, W) float32 [0,1]

    return _apply


class EpicVerbDataset(torch.utils.data.Dataset):
    """Torch Dataset over annotated EPIC action segments.

    __init__ inputs:
        annotations_dir: str/Path with EPIC_100_video_info.csv (fps source).
        videos_root: root with <Pxx>/videos/<id>.MP4 (original or downscaled;
            both roots share the layout and frame counts).
        segments_df: pd.DataFrame from src.data.splits.make_splits (columns
            video_id, narration_id, start_frame, stop_frame INCLUSIVE int64,
            verb str, verb_class int64).
        verb_map: dict[str, int] from verb_label_map (alphabetical ids;
            save the SAME map into every run dir for inference reuse).
        T: frames sampled per clip (default 8, D5 default).
        stop_inclusive: stop_frame convention (True matches EPIC, S1).
        transform: callable np (T,H,W,3) uint8 -> torch (T,C,H,W); overrides
            the default HWC->CHW + /255 pipeline. With a custom transform,
            the [0,1] float contract is the transform's responsibility.
        img_size: optional int; short-side resize applied in the default
            transform (ignored when transform is given).

    __getitem__(i) outputs:
        clip: torch.float32 (T, C, H, W) in [0, 1], RGB (default transform).
        label: python int = verb_map[segments_df[i].verb].

    Missing videos: rows whose video file does not exist are dropped AT
    INIT TIME with a single warning listing how many rows and which videos —
    the Dataset never crashes mid-epoch when a download was incomplete.
    Per-video fps comes from video_info (cached dict); a video absent from
    video_info is probed from the file once with a warning.
    """

    def __init__(self,
                 annotations_dir: str | Path,
                 videos_root: str | Path,
                 segments_df,
                 verb_map: dict[str, int],
                 T: int = 8,
                 stop_inclusive: bool = True,
                 transform: Callable[[np.ndarray], torch.Tensor] | None = None,
                 img_size: int | None = None,
                 fast_seek: bool = False):
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in segments_df.columns]
        if missing_cols:
            raise ValueError(
                f"segments_df is missing columns {missing_cols}; expected "
                f"the output of src.data.splits.make_splits."
            )
        unknown_verbs = sorted(set(segments_df["verb"]) - set(verb_map))
        if unknown_verbs:
            raise ValueError(
                f"segments_df contains verbs not in verb_map: {unknown_verbs}; "
                f"rebuild the splits with the same verb list as the map."
            )

        self.videos_root = Path(videos_root)
        self.verb_map = dict(verb_map)
        self.T = int(T)
        self.stop_inclusive = bool(stop_inclusive)
        self.transform = transform if transform is not None else default_transform(img_size)
        # fast_seek: CAP_PROP_POS_FRAMES seek instead of grab()-from-zero.
        # Default False (annotation-exact by construction). True is safe on
        # OUR EPIC-256 re-encodes — audited pixel-exact on 40 late segments,
        # 21.5x faster (STATUS.md 2026-10-24); other files: audit first.
        self.fast_seek = bool(fast_seek)

        # Frame sampler comes from the registry (reference or owner's core/).
        self.sampler = registry.get("frame_sampler")

        # Per-video fps cache: video_id -> float fps.
        info = load_video_info(annotations_dir)
        self._fps_cache: dict[str, float] = {str(v): float(f)
                                             for v, f in info["fps"].items()}

        # Resolve every referenced video's path ONCE; drop rows whose video
        # is missing (download incomplete) with a warning, never mid-epoch.
        self._paths: dict[str, Path] = {}
        missing_videos: list[str] = []
        for vid in segments_df["video_id"].astype(str).unique():
            p = self.videos_root / vid.split("_")[0] / "videos" / f"{vid}.MP4"
            if p.exists():
                self._paths[vid] = p
            else:
                missing_videos.append(vid)

        if missing_videos:
            df = segments_df[~segments_df["video_id"].astype(str).isin(set(missing_videos))]
            n_dropped = len(segments_df) - len(df)
            warnings.warn(
                f"EpicVerbDataset: dropped {n_dropped}/{len(segments_df)} "
                f"segment rows whose videos are missing on disk "
                f"({len(missing_videos)} videos: {sorted(missing_videos)}). "
                f"Rerun make_splits once the download completes.",
                UserWarning, stacklevel=2,
            )
        else:
            df = segments_df
        self.segments_df = df.reset_index(drop=True)

    def __len__(self) -> int:
        """Number of usable segments (rows). Scalar int."""
        return len(self.segments_df)

    def __getitem__(self, i: int):
        """Return (clip (T,C,H,W) float32, label int) for segment i."""
        row = self.segments_df.iloc[i]
        vid = str(row["video_id"])
        path = self._paths[vid]

        fps = self._fps_cache.get(vid, 0.0)
        if fps <= 0:  # not in video_info: probe the file once, cache, warn
            fps = probe_fps(str(path))
            self._fps_cache[vid] = fps
            warnings.warn(
                f"EpicVerbDataset: video {vid} not in EPIC_100_video_info.csv; "
                f"probed fps={fps:.4f} from the file.", UserWarning,
                stacklevel=2,
            )

        # (T,) int64 sorted frame indices inside start_frame..stop_frame.
        indices = self.sampler.sample_indices(
            int(row["start_frame"]), int(row["stop_frame"]),
            self.T, fps, stop_inclusive=self.stop_inclusive,
        )
        frames = decode_frames(str(path), indices,
                               fast_seek=self.fast_seek)  # (T,H,W,3) uint8 RGB

        clip = self.transform(frames)                # (T,C,H,W) by contract
        label = int(self.verb_map[str(row["verb"])])
        return clip, label
