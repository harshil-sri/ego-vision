"""Sliding-window evaluation: the verb model under LIVE conditions.

Why: the head was trained and validated on clips cut exactly at the
annotated start and stop of each action. The live app does not know where
actions start. It looks at the last `window` seconds every tick. Such a
window can hold half an action, two actions, or none. This script measures
the model on exactly those windows, on the held-out participants.

Stage 1 (cache, slow, GPU): every held-out video is decoded once from the
first frame to the last. Every --tick seconds, for each window length, 16
frames are sampled across the last `window` seconds with the registry frame
sampler (the same rule as src.pipeline.window.SlidingWindowBuffer) and
passed through the frozen encoder and the head.
    -> runs/<id>/sliding_cache.npz: video, frame, probs_<window>

Stage 2 (score, fast): each tick gets a label from the annotations at one
reference moment of its window (the centre by default; rule and reasons in
src/data/window_labels.py):
    a demo verb   an action with one of the 8 verbs covers that moment
    other         only actions with other verbs cover it
    background    no annotated action covers it
Reported per window length:
    tick accuracy on demo-verb ticks (argmax, no threshold)
    how often the model confidently names a demo verb on other /
        background ticks. Every such tick is a possible false fire.
    with each smoother, replayed like the engine: number of fires (verb
        edges), strict precision (fired verb = tick label), lenient
        precision (an action with the fired verb overlaps the window),
        false fires per minute (fires that fail the lenient test), and
        action recall (share of annotated demo-verb actions during which
        the stabilised verb was correct at least once).
A head with a "none" class (scripts.train_window_head) is handled: "none"
never fires, and it lets the same verb fire again afterwards.

Usage:
    python -m scripts.eval_sliding --run-dir runs/videomae_mlp
Outputs: reports/sliding_eval.md (+ .json)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.paths import resolve_data_path

T = 16


def _data():
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    ann = Path(resolve_data_path(cfg["annotations_dir"]))
    root = Path(resolve_data_path(cfg["downscaled_root"]))
    return cfg, ann, root


def build_cache(run_dir: Path, windows: list[float], tick: float, device: str,
                out: Path) -> None:
    """Decode each held-out video once and predict every tick."""
    import cv2
    import torch

    from src import registry
    from src.data.annotations import load_video_info
    from src.models.frozen import target_size
    from src.pipeline.model_source import FrozenVerbSource

    cfg, ann, root = _data()
    src = FrozenVerbSource(str(run_dir), device=device)
    if src.kind != "frozen_videomae":
        raise SystemExit("eval_sliding currently supports frozen_videomae runs")
    enc, head = src.encoder, src.head
    sampler = registry.get("frame_sampler")
    fps_of = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    videos = sorted(p for part in cfg["heldout_participants"]
                    for p in (root / part / "videos").glob("*.MP4"))
    h, w = target_size(256, 456)                       # (224, 392)
    crops = [0, (w - 224) // 2, w - 224]

    def predict(clips: list[np.ndarray]) -> np.ndarray:
        """clips: list of (16, 224, 392, 3) uint8 RGB -> (n, V) probabilities."""
        x = torch.from_numpy(np.stack(clips))                                  # (n,16,224,392,3)
        f = torch.stack([enc.encode(x[:, :, :, c:c + 224]) for c in crops]).mean(0)   # (n, 768)
        with torch.no_grad():
            return torch.softmax(head(f.to(device)).float(), dim=1).cpu().numpy()

    rows_v, rows_f, probs = [], [], {wl: [] for wl in windows}
    t0 = time.time()
    for path in videos:
        vid, fps = path.stem, fps_of[path.stem]
        step, longest = int(round(tick * fps)), int(round(max(windows) * fps))
        ring: deque = deque(maxlen=longest + 2)        # (frame index, resized RGB frame)
        cap = cv2.VideoCapture(str(path))
        pend = {wl: [] for wl in windows}
        f = -1
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            f += 1
            ring.append((f, cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1]))
            if f < longest or (f - longest) % step:
                continue
            by_idx = dict(ring)
            for wl in windows:
                idx = sampler.sample_indices(f - int(round(wl * fps)), f, T, fps, stop_inclusive=True)
                pend[wl].append(np.stack([by_idx[int(i)] for i in idx]))
            rows_v.append(vid); rows_f.append(f)
            if len(pend[windows[0]]) == 8:
                for wl in windows:
                    probs[wl].append(predict(pend[wl])); pend[wl] = []
        cap.release()
        if pend[windows[0]]:
            for wl in windows:
                probs[wl].append(predict(pend[wl]))
        print(f"  {vid}: {f + 1} frames, {len(rows_f)} ticks so far ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, video=np.array(rows_v), frame=np.array(rows_f),
                        verbs=np.array(src.verbs), windows=np.array(windows), tick=tick,
                        **{f"probs_{wl}": np.concatenate(probs[wl]).astype(np.float32) for wl in windows})
    print(f"cached {len(rows_f)} ticks -> {out}")


def probs_from_features(run_dir: Path, feats_dir: Path, window: float, device: str, out: Path) -> None:
    """Fast path: sliding-window FEATURES already exist (scripts.extract_sliding).
    Apply the run's head to the held-out rows and write the same cache format."""
    import torch
    from src.models.frozen import FeatureHead
    cfg, _, _ = _data()
    meta = pd.read_csv(feats_dir / "meta.csv")
    keep = meta["participant_id"].isin(cfg["heldout_participants"]).to_numpy()
    x = torch.from_numpy(np.load(feats_dir / f"feats_{window}.npy")[keep].astype(np.float32)).to(device)
    ck = next(run_dir / n for n in ("last.pt", "best.pt") if (run_dir / n).exists())
    state = torch.load(ck, map_location=device)
    head = FeatureHead(**state["head"]); head.load_state_dict(state["model_state_dict"]); head.to(device).eval()
    with torch.no_grad():
        probs = torch.softmax(head(x).float(), dim=1).cpu().numpy()
    verbs = sorted(state["verb_map"], key=state["verb_map"].get)
    info = json.loads((feats_dir / "info.json").read_text())
    np.savez_compressed(out, video=meta["video_id"].to_numpy(str)[keep], frame=meta["frame"].to_numpy()[keep],
                        verbs=np.array(verbs), windows=np.array([window]), tick=info["heldout_tick_s"],
                        **{f"probs_{window}": probs.astype(np.float32)})
    print(f"cached {keep.sum()} ticks -> {out}")


def score(cache: Path, min_conf: float, ref: float) -> dict:
    """Score a probability cache. Handles heads with and without a "none" class."""
    from src.data.annotations import load_video_info
    from src.data.window_labels import NONE, load_actions, point_labels
    from src.pipeline.smoothers import make_smoother
    _, ann, _ = _data()
    fps_of = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    z = np.load(cache, allow_pickle=True)
    video = z["video"].astype(str); frame = z["frame"]
    all_verbs = [str(v) for v in z["verbs"]]            # the head's classes
    verbs = [v for v in all_verbs if v != NONE]         # the demo verbs
    V, has_none = len(verbs), NONE in all_verbs
    actions = load_actions(ann, verbs)
    minutes = len(frame) * float(z["tick"]) / 60
    res = {"ticks": int(len(frame)), "tick_seconds": float(z["tick"]), "min_conf": min_conf,
           "ref": ref, "verbs": verbs, "has_none_class": has_none, "windows": {}}
    for wl in [float(x) for x in z["windows"]]:
        probs = z[f"probs_{wl}"]
        label, any_verb, segs = point_labels(video, frame, fps_of, wl, verbs, actions, ref)
        # column of the head -> demo verb id (0..V-1), or -1 for "none"
        col = np.array([verbs.index(v) if v != NONE else -1 for v in all_verbs])
        pred, conf = col[probs.argmax(1)], probs.max(1)
        sure = (pred >= 0) & (conf >= min_conf)         # a confident demo-verb prediction
        demo, other, bg = label < V, label == V, label == V + 1
        per_class = [float((pred[label == k] == k).mean()) if (label == k).any() else None for k in range(V)]
        r = {"demo_ticks": int(demo.sum()), "other_ticks": int(other.sum()), "background_ticks": int(bg.sum()),
             "demo_tick_acc": round(float((pred[demo] == label[demo]).mean()), 4),
             "demo_tick_mca": round(float(np.mean([x for x in per_class if x is not None])), 4),
             "demo_tick_recall": {v: (None if x is None else round(x, 3)) for v, x in zip(verbs, per_class)},
             "demo_acc_when_confident": round(float((pred[demo & sure] == label[demo & sure]).mean()), 4),
             "demo_confident_share": round(float(sure[demo].mean()), 4),
             "other_confident_share": round(float(sure[other].mean()), 4),
             "background_confident_share": round(float(sure[bg].mean()), 4),
             "smoothers": {}}
        factories = {
            "raw": None,
            "majority_k3": lambda: make_smoother("majority", all_verbs, k=3, min_conf=min_conf),
            "majority_k5": lambda: make_smoother("majority", all_verbs, k=5, min_conf=min_conf),
            "ema_0.5": lambda: make_smoother("ema", all_verbs, alpha=0.5, min_conf=min_conf),
            "ema_0.3": lambda: make_smoother("ema", all_verbs, alpha=0.3, min_conf=min_conf),
            "hysteresis": lambda: make_smoother("hysteresis", all_verbs, margin=0.1, min_conf=min_conf),
        }
        for name, fac in factories.items():
            fires = strict = lenient = 0
            hit = set()                                     # demo-verb actions recalled
            for v in np.unique(video):
                m = np.flatnonzero(video == v)
                sm = fac() if fac else None
                last = None
                vsegs = [sg for sg in segs if sg[0] == v]
                wlen = int(round(wl * fps_of[v]))
                for i in m:
                    if sm is None:
                        out, c = (verbs[int(pred[i])] if pred[i] >= 0 else NONE), float(conf[i])
                    else:
                        out, c = sm.update(probs[i].astype(np.float64), 0)
                    if out is None or c < min_conf:
                        continue                            # unsure: the engine keeps its last verb
                    if out == NONE:
                        last = None                         # "nothing": the same verb may fire again
                        continue
                    k = verbs.index(out)
                    f1, f0 = int(frame[i]), int(frame[i]) - wlen
                    for sg in vsegs:                        # stabilised verb correct during this action?
                        if sg[3] == k and sg[1] <= f1 and sg[2] >= f0:
                            hit.add(sg)
                    if out != last:
                        fires += 1; last = out
                        strict += int(label[i] == k); lenient += int(any_verb[i, k])
            r["smoothers"][name] = {
                "fires": fires, "fires_per_min": round(fires / minutes, 2),
                "false_fires_per_min": round((fires - lenient) / minutes, 2),
                "precision_strict": round(strict / max(fires, 1), 4),
                "precision_lenient": round(lenient / max(fires, 1), 4),
                "segment_recall": round(len(hit) / max(len(segs), 1), 4)}
        r["demo_segments"] = len(segs)
        r["demo_segments_per_min"] = round(len(segs) / minutes, 2)
        res["windows"][str(wl)] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/videomae_mlp")
    ap.add_argument("--windows", type=float, nargs="+", default=[2.0, 1.0])
    ap.add_argument("--tick", type=float, default=0.5, help="seconds between predictions")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--ref", type=float, default=0.5,
                    help="where in the window the label is read: 0.5 = centre, 0 = the newest frame")
    ap.add_argument("--feats", default=None,
                    help="sliding-window feature directory from scripts.extract_sliding (fast path)")
    ap.add_argument("--window", type=float, default=2.0, help="window length to use from --feats")
    ap.add_argument("--rebuild", action="store_true", help="recompute the probability cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/sliding_eval.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cache = run_dir / "sliding_cache.npz"
    if args.rebuild or not cache.exists():
        if args.feats:
            fd = Path(resolve_data_path(args.feats))
            probs_from_features(run_dir, fd, float(args.window), args.device, cache)
        else:
            build_cache(run_dir, args.windows, args.tick, args.device, cache)
    res = score(cache, args.min_conf, args.ref)
    res["run_dir"] = str(run_dir)
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    L = ["# Sliding-window evaluation (live conditions, held-out participants)", "",
         f"Generated by `python -m scripts.eval_sliding --run-dir {run_dir.as_posix()}`. "
         f"{res['ticks']} ticks, one every {res['tick_seconds']} s, over the full held-out videos. "
         "Each tick is labelled by the annotated action at one reference moment of its window "
         f"(ref = {args.ref}; 0.5 is the window centre). "
         f"Confident = a demo verb is the top class with probability >= {args.min_conf}. "
         f"The head {'has a' if res['has_none_class'] else 'has NO'} 'none' class. "
         "See the script docstring for the definitions.", ""]
    for wl, r in res["windows"].items():
        L += [f"## Window {wl} s", "",
              f"- Ticks: {r['demo_ticks']} demo verb, {r['other_ticks']} other verb, {r['background_ticks']} background.",
              f"- Demo-verb ticks: accuracy {r['demo_tick_acc']}, mean-class accuracy {r['demo_tick_mca']}; "
              f"confident on {r['demo_confident_share']:.0%} of them, accuracy when confident {r['demo_acc_when_confident']}.",
              f"- Per-verb recall on demo ticks: {r['demo_tick_recall']}",
              f"- The model is confident on {r['other_confident_share']:.0%} of other-verb ticks and "
              f"{r['background_confident_share']:.0%} of background ticks (each one a possible false fire).", "",
              f"- Annotated demo-verb actions: {r['demo_segments']} ({r['demo_segments_per_min']} per minute).", "",
              "| smoother | fires | fires per minute | false fires per minute | precision (strict) | "
              f"precision (lenient) | action recall (of {r['demo_segments']}) |", "|---|---|---|---|---|---|---|"]
        for name, s in r["smoothers"].items():
            L.append(f"| {name} | {s['fires']} | {s['fires_per_min']} | {s['false_fires_per_min']} | "
                     f"{s['precision_strict']} | {s['precision_lenient']} | {s['segment_recall']} |")
        L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
