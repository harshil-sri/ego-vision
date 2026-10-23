"""Evaluate a trained run's best checkpoint on the validation split.

Rebuilds the exact validation dataset from the run's saved config + verb
map (same split policy as training: held-out participants, availability on
the downscaled root), loads <run-dir>/best.pt, evaluates, writes
<run-dir>/metrics.json + <run-dir>/confusion.png, and prints a table with
the majority-class baseline beside every number.

Usage:
    python -m scripts.eval_verb_model --run-dir runs/baseline
    python -m scripts.eval_verb_model --run-dir runs/baseline --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """CLI entry point. Outputs: metrics.json + confusion.png in the run
    dir; a printed table (baseline beside every number); exit code 2 if the
    run dir or its artifacts are missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="training output dir (has config.yaml, "
                             "verb_map.json, best.pt)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="override cfg.device for this evaluation")
    parser.add_argument("--annotations-dir", type=str, default=None,
                        help="override cfg.annotations_dir (Kaggle mounts)")
    parser.add_argument("--videos-root", type=str, default=None,
                        help="override cfg.videos_root (downscaled layout)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (default 2)")
    args = parser.parse_args()

    # Lazy imports (same convention as scripts/train.py).
    import torch
    import torch.utils.data
    from src.data.dataset import EpicVerbDataset
    from src.data.splits import make_splits
    from src.models.variants import VerbModel
    from src.train.config import load_config
    from src.train.eval_model import evaluate, save_confusion_png

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    verb_map_path = run_dir / "verb_map.json"
    ckpt_path = run_dir / "best.pt"
    for p, what in [(config_path, "config.yaml"), (verb_map_path, "verb_map.json"),
                    (ckpt_path, "best.pt")]:
        if not p.exists():
            raise SystemExit(f"missing {what} in run dir: {p}")

    cfg = load_config(config_path)  # exact cfg the run trained with
    if args.device is not None:
        cfg.device = args.device
    if args.annotations_dir is not None:
        cfg.annotations_dir = args.annotations_dir
    if args.videos_root is not None:
        cfg.videos_root = args.videos_root

    with open(verb_map_path, "r", encoding="utf-8") as f:
        verb_map: dict = json.load(f)  # {"verb": id} saved by train_run
    verbs = sorted(verb_map, key=lambda v: verb_map[v])  # id 0..C-1 order
    class_names = verbs

    # Same split construction as scripts/train.py (V1: held-out participants).
    _train_df, val_df, _tids, _vids = make_splits(
        cfg.annotations_dir, cfg.videos_root,
        heldout_participants=list(cfg.heldout_participants),
        verbs=list(verb_map), downscaled_root=None)
    if len(val_df) == 0:
        raise SystemExit("validation split is empty — no held-out-participant "
                         f"videos under {cfg.videos_root}; cannot evaluate.")

    val_ds = EpicVerbDataset(cfg.annotations_dir, cfg.videos_root, val_df,
                             verb_map, T=cfg.T, img_size=cfg.img_size)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.workers)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cfg.device='cuda' but CUDA unavailable; "
                         "pass --device cpu.")

    # Rebuild the model exactly as trained (random init + load; no download).
    model = VerbModel(cfg.backbone, cfg.num_verbs, cfg.variant,
                      pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])  # (B,T,C,H,W)->(B,C)
    model.to(device)

    results = evaluate(model, val_loader, device, cfg.num_verbs)

    # Persist metrics + confusion PNG inside the run dir.
    out = dict(results)
    out["run_dir"] = str(run_dir)
    out["checkpoint"] = str(ckpt_path)
    out["num_verbs"] = cfg.num_verbs
    out["class_names"] = class_names
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    save_confusion_png(results["confusion"], class_names,
                       run_dir / "confusion.png")

    # Printed table: majority baseline BESIDE every number (brief rule).
    baseline = results["majority_baseline"]
    print(f"\nevaluation of {ckpt_path} on {len(val_df)} val segments "
          f"(device {device})")
    print(f"{'metric':<28s}{'value':>10s}{'majority baseline':>20s}")
    print("-" * 58)
    print(f"{'top-1 accuracy':<28s}{results['top1']:>10.4f}{baseline:>20.4f}")
    print(f"{'top-5 accuracy':<28s}{results['top5']:>10.4f}{baseline:>20.4f}"
          + (f"   [{results['note']}]" if "note" in results else ""))
    print(f"{'mean-class accuracy':<28s}{results['mean_class']:>10.4f}"
          f"{baseline:>20.4f}")
    print("\nper-class accuracy (baseline in last column):")
    for name, acc in zip(class_names, results["per_class"]):
        print(f"  {name:<12s}{acc:>10.4f}{baseline:>20.4f}")
    print(f"\nwrote {run_dir / 'metrics.json'} and {run_dir / 'confusion.png'}")


if __name__ == "__main__":
    main()
