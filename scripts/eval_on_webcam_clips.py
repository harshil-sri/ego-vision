"""Run a trained checkpoint over recorded webcam clips (domain-gap check).

One prediction per clip: the clip's saved JPGs (~10 fps) are sampled down
to T frames with the registry frame sampler, preprocessed exactly like
training (src.data.dataset.default_transform: short-side resize + RGB +
[0,1]), passed through VerbModel.predict_probs, and scored with the
REGISTRY metrics module. Reports per-verb top-1 + overall to
reports/webcam_eval.md. These clips are evaluation only, never training.

If there are no clips or no checkpoint, writes "NOT RUN" plus the reason
instead of crashing.

Usage:
    python -m scripts.eval_on_webcam_clips --run-dir runs/baseline
    python -m scripts.eval_on_webcam_clips        # writes NOT RUN + reason
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS_ROOT = Path("data/webcam_eval")
DEFAULT_OUT = REPO_ROOT / "reports" / "webcam_eval.md"


def _write_not_run(out_path: Path, reason: str) -> None:
    """Write a NOT RUN report with the reason.

    Inputs: out_path Path; reason str (factual). Output: None (file
    written; nothing fabricated — explicitly not run).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Webcam-clip verb evaluation\n\nNOT RUN\n\nReason: " + reason + "\n",
        encoding="utf-8")
    print(f"NOT RUN — {reason}\nwrote {out_path}")


def _load_clip_tensor(directory: Path, T: int, img_size: int,
                      sampler) -> tuple:
    """Load one clip's JPGs and build the model input.

    Inputs:
        directory: folder of <anything>_%03d.jpg frames (sorted by name).
        T: frames the model needs; sampled with the registry sampler over
            the clip's frame range at the recording save rate (10 fps —
            the fps argument is unused by the index arithmetic, like the
            dataset path).
        img_size: short-side resize, same as training preprocessing.
        sampler: registry frame_sampler module.

    Output: (clip (T, 3, H, W) float32 in [0,1], n_files int) or
        (None, n) when the folder has no readable frames.
    """
    import cv2
    import numpy as np
    import torch
    from src.data.dataset import default_transform

    files = sorted(directory.glob("*.jpg"))
    if not files:
        return None, 0

    frames = []
    for fp in files:
        img = cv2.imread(str(fp))            # (H, W, 3) uint8 BGR
        if img is None:                      # unreadable file: skip it
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB (training order)
        frames.append(img)
    if not frames:
        return None, len(files)

    stack = np.stack(frames)                 # (N, H, W, 3) uint8 RGB
    indices = sampler.sample_indices(0, stack.shape[0] - 1, T,
                                     fps=10.0, stop_inclusive=True)  # (T,)
    selected = stack[indices]                # (T, H, W, 3) uint8
    transform = default_transform(img_size)  # same preprocessing as training
    return transform(selected), stack.shape[0]  # ((T,3,H',W') float32, int)


def main() -> None:
    """CLI entry point. Outputs: reports/webcam_eval.md (+ console table),
    or a NOT RUN report when clips/checkpoint are missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="training run dir with config.yaml, "
                             "verb_map.json, best.pt")
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    csv_path = args.clips_root / "clips.csv"
    if not csv_path.exists():
        _write_not_run(args.out, f"no clips index at {csv_path} "
                                 f"(run scripts/record_verb_clips.py first)")
        return
    if args.run_dir is None or not (args.run_dir / "best.pt").exists():
        _write_not_run(args.out, "no checkpoint given (pass --run-dir "
                                 "runs/<id> with a trained best.pt)")
        return

    # Lazy heavy imports after the NOT RUN gates.
    import pandas as pd
    import torch
    from src import registry
    from src.models.variants import VerbModel
    from src.train.config import load_config

    run_dir: Path = args.run_dir
    cfg = load_config(run_dir / "config.yaml")
    with open(run_dir / "verb_map.json", "r", encoding="utf-8") as f:
        verb_map: dict = json.load(f)  # {"verb": id}
    id_to_verb = {v: k for k, v in verb_map.items()}  # id -> verb name
    class_names = sorted(verb_map, key=lambda v: verb_map[v])
    sampler = registry.get("frame_sampler")
    metrics = registry.get("metrics")

    device = torch.device(args.device)
    model = VerbModel(cfg.backbone, len(verb_map), cfg.variant,
                      pretrained=False)  # weights come from the checkpoint
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    clips_df = pd.read_csv(csv_path)  # rows from scripts/record_verb_clips.py
    y_true, y_pred, skipped = [], [], 0
    for _idx, row in clips_df.iterrows():
        clip, n_files = _load_clip_tensor(Path(row["directory"]), cfg.T,
                                          cfg.img_size, sampler)
        if clip is None:
            skipped += 1
            continue
        probs = model.predict_probs(clip.to(device))  # (num_verbs,) float32
        y_pred.append(int(probs.argmax().item()))
        y_true.append(int(verb_map[str(row["verb"])]))

    if not y_true:
        _write_not_run(args.out, f"clips index exists at {csv_path} but no "
                                 f"clip had readable frames")
        return

    import numpy as np
    y_true_arr = np.asarray(y_true, dtype=np.int64)   # (N,) int64
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)   # (N,) int64
    num_classes = len(verb_map)
    overall = metrics.topk_accuracy(
        np.eye(num_classes, dtype=np.float64)[y_pred_arr], y_true_arr, k=1)
    per_class = metrics.per_class_accuracy(y_pred_arr, y_true_arr,
                                           num_classes)  # (C,) float
    baseline = metrics.majority_baseline(y_true_arr)

    lines = [
        "# Webcam-clip verb evaluation (domain-gap check)",
        "",
        f"Checkpoint: `{run_dir / 'best.pt'}` (backbone {cfg.backbone}, "
        f"variant {cfg.variant}, T {cfg.T}, img {cfg.img_size}).",
        f"Clips evaluated: {len(y_true)} ({skipped} skipped: no readable "
        f"frames). Evaluation-only data; never used for training.",
        "",
        "| verb | webcam top-1 | majority baseline |",
        "|---|---:|---:|",
    ]
    for cid, name in enumerate(class_names):
        lines.append(f"| {name} | {per_class[cid]:.4f} | {baseline:.4f} |")
    lines += ["",
              f"Overall top-1: {overall:.4f} (majority baseline "
              f"{baseline:.4f}).",
              "",
              "These are the owner's own kitchen and camera; a gap to the "
              "EPIC validation numbers is expected (domain gap) — measure, "
              "do not assume.",
              ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"overall top-1 {overall:.4f} vs majority baseline {baseline:.4f} "
          f"over {len(y_true)} clips; wrote {args.out}")


if __name__ == "__main__":
    main()
