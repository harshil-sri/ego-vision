"""Train a temporal head on cached DINOv2 features (seconds per run).

Reads the output of scripts.extract_features, keeps the 8 demo verbs BY
OFFICIAL VERB CLASS (so "pick-up" counts as take, "put-down" as put, ...),
holds out whole participants for validation (V1 split: P22 + P30) and
trains src.models.frozen.FeatureHead.

Why by class: the earlier pipeline matched the raw narration word, which
kept 2,995 of the 5,435 usable segments on the same 55 videos.

Augmentation (features are fixed, so it is cheap):
    - mirrored frames (pre-extracted) with probability 0.5
    - temporal jitter: the cache has 16 frames per clip; training takes 8
      by picking one frame at random from each consecutive pair
    - dropout inside the head, label smoothing

Evaluation uses frames 0,2,4,...,14 of the un-mirrored features, the metrics
module from the registry, and prints the majority baseline beside every
number.

Outputs in --out (a run directory the app can load):
    best.pt, verb_map.json, config.yaml, log.csv, metrics.json

Usage:
    python -m scripts.train_head --kind tfm --out runs/dino_tfm
    python -m scripts.train_head --sweep        # compare head kinds
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src import registry
from src.paths import resolve_data_path

DEMO_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]
DEFAULT_FEATS = "/mnt/ssd/epic-kitchens/features/dinov2_vits14_T16"


def load_data(feat_dir: Path, ann_dir: Path, heldout: list[str],
              raw_match: bool = False):
    """Load cached features and build the 8-verb train/val index sets.

    Outputs:
        feats, flips: np.memmap (N, 16, 768) float16
        labels: np.ndarray (N,) int64, -1 for segments outside the 8 verbs
        tr, va: np.ndarray of row indices (train / validation)
        verb_map: dict verb -> label id (alphabetical, same rule as
            src.data.annotations.verb_label_map)
        meta: pd.DataFrame, one row per cached segment
    """
    feats = np.load(feat_dir / "feats.npy", mmap_mode="r")
    flips = np.load(feat_dir / "feats_flip.npy", mmap_mode="r")
    meta = pd.read_csv(feat_dir / "meta.csv")
    classes = pd.read_csv(ann_dir / "EPIC_100_verb_classes.csv")
    key_of = dict(zip(classes["id"].astype(int), classes["key"].astype(str)))
    verb_map = {v: i for i, v in enumerate(sorted(DEMO_VERBS))}
    if raw_match:   # old behaviour, kept only to measure what it cost
        canon = meta["verb"].astype(str)
    else:
        canon = meta["verb_class"].astype(int).map(key_of)
    labels = canon.map(verb_map).fillna(-1).astype(np.int64).to_numpy()
    usable = (labels >= 0) & (meta["ok"].to_numpy() == 1)
    is_val = meta["participant_id"].isin(heldout).to_numpy()
    tr = np.flatnonzero(usable & ~is_val)
    va = np.flatnonzero(usable & is_val)
    return feats, flips, labels, tr, va, verb_map, meta


def evaluate(model, X: torch.Tensor, y: np.ndarray, num_classes: int) -> dict:
    """X (N, 8, F) on device, y (N,) -> metrics dict + probs (N, C)."""
    m = registry.get("metrics")
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    pred = probs.argmax(1)
    return {"acc": float((pred == y).mean()),
            "mca": float(m.mean_class_accuracy(pred, y, num_classes)),
            "top3": float(m.topk_accuracy(probs, y, k=3)),
            "per_class": m.per_class_accuracy(pred, y, num_classes).tolist(),
            "confusion": m.confusion_matrix(pred, y, num_classes).tolist(),
            "probs": probs}


def train_one(args, kind: str, motion: bool, seed: int, data, out: Path | None):
    """One training run. Returns the best-epoch metrics dict."""
    from src.models.frozen import FeatureHead

    feats, flips, labels, tr, va, verb_map, _ = data
    C = len(verb_map)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = args.device

    Xtr = torch.from_numpy(np.asarray(feats[tr], dtype=np.float32)).to(dev)   # (Ntr,16,F)
    Xfl = torch.from_numpy(np.asarray(flips[tr], dtype=np.float32)).to(dev)
    ytr = torch.from_numpy(labels[tr]).to(dev)
    Xva = torch.from_numpy(np.asarray(feats[va], dtype=np.float32))[:, 0::2].to(dev)  # (Nva,8,F)
    yva = labels[va]

    counts = np.bincount(labels[tr], minlength=C).astype(np.float64)
    weight = None
    if args.class_weights:   # inverse frequency, softened by a square root
        w = (counts.sum() / (C * np.maximum(counts, 1))) ** 0.5
        weight = torch.tensor(w, dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    model = FeatureHead(Xtr.shape[-1], C, kind=kind, hidden=args.hidden,
                        motion=motion, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    majority = float(registry.get("metrics").majority_baseline(yva))
    best, rows = None, []
    pair = torch.arange(0, 16, 2, device=dev)                 # (8,) start of each pair
    for epoch in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(tr))).to(dev)
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            use_flip = torch.rand(len(b), device=dev) < 0.5
            x16 = torch.where(use_flip[:, None, None], Xfl[b], Xtr[b])   # (B,16,F)
            pick = pair[None, :] + torch.randint(0, 2, (len(b), 8), device=dev)  # (B,8)
            x = torch.gather(x16, 1, pick[:, :, None].expand(-1, -1, x16.shape[-1]))  # (B,8,F)
            loss = loss_fn(model(x), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(b)
        ev = evaluate(model, Xva, yva, C)
        rows.append({"epoch": epoch, "train_loss": round(total / len(tr), 4),
                     "val_acc": round(ev["acc"], 4), "val_mca": round(ev["mca"], 4)})
        if best is None or ev["mca"] > best["mca"]:
            best = dict(ev, epoch=epoch)
            if out is not None:
                torch.save({"model_state_dict": model.state_dict(),
                            "head": model.config(), "epoch": epoch,
                            "verb_map": verb_map, "kind": "frozen_dino",
                            "encoder": "dinov2_vits14", "T": 8}, out / "best.pt")
    best.update(majority=majority, n_train=int(len(tr)), n_val=int(len(va)),
                kind=kind, motion=motion, seed=seed)
    if out is not None:
        with open(out / "log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        (out / "verb_map.json").write_text(json.dumps(verb_map, indent=2))
        (out / "config.yaml").write_text(yaml.safe_dump(
            {"model": "frozen_dino", "encoder": "dinov2_vits14", "T": 8,
             "head": model.config(), "seed": seed, "epochs": args.epochs,
             "lr": args.lr, "heldout_participants": args.heldout,
             "label_match": "raw" if args.raw_match else "verb_class"}))
        np.savez(out / "val_preds.npz", probs=best["probs"], labels=yva, rows=va)
        slim = {k: v for k, v in best.items() if k != "probs"}
        slim["verbs"] = sorted(verb_map, key=verb_map.get)
        slim["train_counts"] = counts.astype(int).tolist()
        (out / "metrics.json").write_text(json.dumps(slim, indent=2))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=DEFAULT_FEATS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kind", default="tfm", choices=["mean", "attn", "tfm"])
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--heldout", nargs="+", default=["P22", "P30"])
    ap.add_argument("--raw-match", action="store_true",
                    help="old label rule (raw narration word) for comparison")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=3, help="seeds per row in --sweep")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    ann = Path(resolve_data_path(data_cfg["annotations_dir"]))
    data = load_data(Path(resolve_data_path(args.feats)), ann, args.heldout, args.raw_match)
    _, _, labels, tr, va, verb_map, _ = data
    verbs = sorted(verb_map, key=verb_map.get)
    print(f"train {len(tr)}  val {len(va)}  verbs {verbs}")
    print("train counts", np.bincount(labels[tr], minlength=len(verbs)).tolist())
    print("val counts  ", np.bincount(labels[va], minlength=len(verbs)).tolist())

    if args.sweep:
        print(f"\n{'head':<6}{'motion':<8}{'val_acc':>16}{'val_mca':>16}{'top3':>8}  (mean +- std over {args.seeds} seeds)")
        for kind in ("mean", "attn", "tfm"):
            for motion in (False, True):
                rs = [train_one(args, kind, motion, args.seed + s, data, None)
                      for s in range(args.seeds)]
                a = np.array([r["acc"] for r in rs]); m = np.array([r["mca"] for r in rs])
                t = np.mean([r["top3"] for r in rs])
                print(f"{kind:<6}{str(motion):<8}{a.mean():>9.4f} +-{a.std():.3f}"
                      f"{m.mean():>9.4f} +-{m.std():.3f}{t:>8.4f}")
        print(f"majority baseline (val): {rs[0]['majority']:.4f}")
        return

    out = Path(args.out or f"runs/dino_{args.kind}")
    out.mkdir(parents=True, exist_ok=True)
    best = train_one(args, args.kind, not args.no_motion, args.seed, data, out)
    print(f"\nbest epoch {best['epoch']}: val_acc {best['acc']:.4f}  "
          f"val_mca {best['mca']:.4f}  top3 {best['top3']:.4f}  "
          f"(majority baseline {best['majority']:.4f})")
    for v, a in zip(verbs, best["per_class"]):
        print(f"  {v:<6} recall {a:.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
