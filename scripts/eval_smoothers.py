"""Offline smoother bake-off (decision D6) on REAL validation data.

Replays every validation segment through the trained checkpoint once
(fast_seek decode — audited pixel-exact), caches the probability series per
video in annotation order, then scores each smoother
(raw / ema / majority / hysteresis) the way the live engine uses them:

  - tick accuracy: smoothed argmax vs the segment's true verb per tick
  - mean class accuracy over verbs present in val
  - fire analysis: the engine CONSUMES verb EDGES (a stabilized verb that
    differs from the last fired one) -> precision of fires vs the true verb
    of the tick they fired on, and how many fires each smoother produces.

Only numbers from commands actually run land in the report (anti-fabrication:
the cache and report are regenerated together or not at all).

Usage:
    python3 -m scripts.eval_smoothers --run-dir runs/baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_val_segments(run_dir: Path) -> pd.DataFrame:
    """Val split for the run's videos, ordered as a per-video time series."""
    from src.data.splits import make_splits
    data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    verb_map = json.loads((run_dir / "verb_map.json").read_text())
    _, val_df, _, _ = make_splits(
        data_cfg["annotations_dir"], data_cfg["downscaled_root"],
        heldout_participants=data_cfg["heldout_participants"],
        verbs=sorted(verb_map), split_csv=None)
    return val_df.sort_values(["video_id", "start_frame"]).reset_index(drop=True)


def _build_cache(run_dir: Path, val_df: pd.DataFrame, out: Path):
    """Decode + predict every val segment once; cache probs/labels/order."""
    from src.data.dataset import load_video_info
    from src.data.video_io import decode_frames
    from src.pipeline.model_source import ModelVerbSource
    from src.pipeline.window import SlidingWindowBuffer  # noqa: F401 (doc)

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    src = ModelVerbSource(str(run_dir), device=cfg.get("device", "cpu"))
    data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    fps = {v: float(f) for v, f in
           load_video_info(data_cfg["annotations_dir"])["fps"].items()}
    sampler = __import__("src.registry", fromlist=["get"]).get("frame_sampler")
    T = int(cfg.get("T", 8))

    from src.capture import FramePacket
    rows = []
    t0 = time.time()
    for i, row in val_df.iterrows():
        vid = row.video_id
        path = Path(data_cfg["downscaled_root"]) / vid.split("_")[0] / "videos" / f"{vid}.MP4"
        idx = sampler.sample_indices(int(row.start_frame), int(row.stop_frame),
                                     T, fps[vid], stop_inclusive=True)
        frames = decode_frames(str(path), idx, fast_seek=True)
        packets = [FramePacket(frame_id=int(f), timestamp_ns=k, image=frames[k])
                   for k, f in enumerate(idx)]
        probs = src.predict(packets)
        rows.append((vid, row.verb, probs.astype(np.float32)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_df)} segments ({time.time() - t0:.0f}s)",
                  flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        video=np.array([r[0] for r in rows]),
        verb=np.array([r[1] for r in rows]),
        probs=np.stack([r[2] for r in rows]),   # (N, C) float32
        verbs=np.array(sorted(json.loads(
            (run_dir / "verb_map.json").read_text()))))
    print(f"cached {len(rows)} predictions -> {out}")


def _score(name, series_iter, verbs, smoother_factory, min_conf):
    """Replay each video's prob series; return per-smoother metrics dict."""
    from reference.metrics_ref import mean_class_accuracy
    tick_ok, tick_true, tick_pred, fires, fire_ok = 0, [], [], 0, 0
    conf_true, conf_pred = [], []      # confident ticks only (for mca)
    for probs, labels in series_iter:           # one video at a time
        sm = smoother_factory() if smoother_factory else None
        last_fired = None
        verb_idx = {v: i for i, v in enumerate(verbs)}
        for p, true in zip(probs, labels):
            if sm is None:                      # raw baseline
                i = int(np.argmax(p)); conf = float(p[i]); out = verbs[i]
            else:
                out, conf = sm.update(p, 0)
            pred = out if (out is not None and conf >= min_conf) else None
            tick_pred.append(verb_idx[pred] if pred else -1)
            tick_true.append(verb_idx[true])
            if pred == true:
                tick_ok += 1
            if pred is not None:
                conf_true.append(verb_idx[true]); conf_pred.append(verb_idx[pred])
            if pred is not None and pred != last_fired:
                fires += 1
                fire_ok += int(pred == true)
                last_fired = pred
    n = len(tick_true)
    present = sorted(set(conf_true)) if conf_true else [0]
    return {
        "smoother": name,
        "coverage": round(len(conf_true) / n, 4),
        "tick_acc": round(tick_ok / n, 4),
        "tick_mca_conf": round(mean_class_accuracy(
            np.array(conf_pred), np.array(conf_true),
            max(present) + 1), 4) if conf_true else None,
        "fires": fires,
        "fire_precision": round(fire_ok / fires, 4) if fires else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/baseline")
    ap.add_argument("--cache", default="runs/eval_cache/val_preds.npz")
    ap.add_argument("--out", default="reports/smoother_eval.md")
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cache = Path(args.cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        video, verb, probs, verbs = (z["video"], z["verb"], z["probs"],
                                     [str(v) for v in z["verbs"]])
        print(f"loaded cache {cache} ({len(video)} segments)")
    else:
        val_df = _load_val_segments(run_dir)
        _build_cache(run_dir, val_df, cache)
        z = np.load(cache, allow_pickle=True)
        video, verb, probs, verbs = (z["video"], z["verb"], z["probs"],
                                     [str(v) for v in z["verbs"]])

    # group into per-video ordered series
    by_video: dict[str, list] = {}
    for v, lb, p in zip(video, verb, probs):
        by_video.setdefault(str(v), []).append((p, str(lb)))
    series = [(np.stack([x[0] for x in rows]),
               [x[1] for x in rows]) for rows in by_video.values()]

    from src.pipeline.smoothers import make_smoother
    variants = [
        ("raw", None),
        ("ema", lambda: make_smoother("ema", verbs, alpha=0.3,
                                       min_conf=args.min_conf)),
        ("majority", lambda: make_smoother("majority", verbs, k=5,
                                           min_conf=args.min_conf)),
        ("hysteresis", lambda: make_smoother("hysteresis", verbs, margin=0.1,
                                             min_conf=args.min_conf)),
    ]
    results = [_score(n, series, verbs, f, args.min_conf)
               for n, f in variants]

    out = Path(args.out)
    lines = [
        "# Smoother bake-off (D6) — offline on real val segments", "",
        f"Model: `{run_dir}` over {len(video)} val segments "
        f"({len(by_video)} videos), per-video replay in annotation order,",
        f"min_conf={args.min_conf}. Fires = stabilized verb edges (what the "
        "engine actually consumes). tick_acc counts 'no confident verb' as a "
        "miss; tick_mca_conf is over confident ticks only.", "",
        "| smoother | coverage | tick acc | tick mca (confident) | fires | fire precision |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['smoother']} | {r['coverage']} | {r['tick_acc']} "
                     f"| {r['tick_mca_conf']} | {r['fires']} | "
                     f"{r['fire_precision']} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
