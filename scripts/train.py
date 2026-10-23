"""Train a verb model from a YAML config (WP3 CLI).

Builds the splits + datasets from the config's data fields (lazy imports:
nothing heavy loads until after argument parsing), runs train_run, prints
the run directory.

Usage:
    python -m scripts.train --config configs/train_baseline.yaml
    python -m scripts.train --config configs/train_overfit32.yaml --device cpu
    # quick smoke without touching real data availability assumptions:
    python -m scripts.train --config configs/train_baseline.yaml \
        --train-limit 32 --overfit-batch 32 --no-pretrained

Verb classes: read from configs/data.yaml `candidate_verbs` (or --verbs);
ids are assigned alphabetically by verb_label_map, matching the dataset
contract. The SAME map is saved into the run dir (verb_map.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def _candidate_verbs(cli_verbs: str | None) -> list:
    """Resolve the verb list: CLI --verbs wins, else configs/data.yaml.

    Input: cli_verbs str|None (comma-separated) from argparse.
    Output: list[str] of verb names (never empty; errors exit with code 2).
    """
    if cli_verbs:
        return [v.strip() for v in cli_verbs.split(",") if v.strip()]

    import yaml  # lazy: only needed for the default path

    with open(DATA_CONFIG, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    verbs = data_cfg.get("candidate_verbs")
    if not verbs:
        sys.exit(f"no candidate_verbs in {DATA_CONFIG} and no --verbs given")
    return list(verbs)


def main() -> None:
    """CLI entry point. Outputs: prints the run dir; exit code 2 on setup
    problems (missing data), 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="path to a train config YAML (configs/train_*.yaml)")
    parser.add_argument("--overfit-batch", type=int, default=None,
                        help="override cfg.overfit_batch (quick gate run)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="override cfg.train_limit (cap train segments)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="override cfg.device (Kaggle: --device cuda)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override cfg.epochs (smoke runs)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (default 2; 0 = in-process)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random backbone init (no ImageNet download)")
    parser.add_argument("--verbs", type=str, default=None,
                        help="comma-separated verb list overriding data.yaml")
    args = parser.parse_args()

    # Lazy imports: keep --help and argparse errors fast; torch/data stack
    # only loads once we actually run.
    from src.data.annotations import verb_label_map
    from src.data.dataset import EpicVerbDataset
    from src.data.splits import make_splits
    from src.train.config import load_config
    from src.train.loop import train_run

    cfg = load_config(args.config)

    # CLI overrides (None = keep the config value).
    if args.overfit_batch is not None:
        cfg.overfit_batch = args.overfit_batch
    if args.train_limit is not None:
        cfg.train_limit = args.train_limit
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs

    verbs = _candidate_verbs(args.verbs)
    verb_map = verb_label_map(verbs)  # {verb: id}, ids ALPHABETICAL
    if len(verb_map) != cfg.num_verbs:
        sys.exit(f"config num_verbs={cfg.num_verbs} but the verb list has "
                 f"{len(verb_map)} entries: {sorted(verb_map)}")

    # Splits from the SAME data fields the dataset uses (V1 internal split:
    # held-out participants; availability checked on the downscaled root).
    train_df, val_df, train_ids, val_ids = make_splits(
        cfg.annotations_dir, cfg.videos_root,
        heldout_participants=list(cfg.heldout_participants),
        verbs=verbs, downscaled_root=None)

    # train_limit caps TRAIN segments only (quick-run knob; val untouched).
    if cfg.train_limit is not None:
        train_df = train_df.head(cfg.train_limit).reset_index(drop=True)

    if len(train_df) == 0:
        sys.exit(f"no train segments found under cfg.videos_root="
                 f"{cfg.videos_root} (annotations: {cfg.annotations_dir}). "
                 f"Download/downscale videos first, or check the config paths.")
    if len(val_df) == 0:
        print("WARNING: validation split is empty (no held-out-participant "
              "videos on disk); val metrics will be 0.0.", file=sys.stderr)

    print(f"train segments: {len(train_df)} ({len(train_ids)} videos) | "
          f"val segments: {len(val_df)} ({len(val_ids)} videos)")

    train_ds = EpicVerbDataset(cfg.annotations_dir, cfg.videos_root,
                               train_df, verb_map, T=cfg.T,
                               img_size=cfg.img_size)
    val_ds = EpicVerbDataset(cfg.annotations_dir, cfg.videos_root,
                             val_df, verb_map, T=cfg.T,
                             img_size=cfg.img_size)

    best_path = train_run(cfg, train_ds, val_ds, verb_map=verb_map,
                          pretrained=not args.no_pretrained,
                          num_workers=args.workers)
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
