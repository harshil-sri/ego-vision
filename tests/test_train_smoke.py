"""CPU smoke tests for src/train (config loader + train_run).

Two gates in one file:
  1. a 2-epoch synthetic run: loss decreases, and log.csv / best.pt /
     metrics.json land in out_dir with the documented contents;
  2. the overfit_batch gate on the separable brightness dataset: training
     accuracy reaches >= 0.99 and stops early (the WP3 sanity gate).

Everything is deliberately tiny (resnet18, 32x32, T=2) so the whole file
runs on CPU in well under a minute. No network access: pretrained=False.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.data.synthetic import SyntheticClipDataset
from src.train.config import TrainConfig, load_config
from src.train.loop import train_run


def _tiny_cfg(out_dir: Path, **overrides) -> TrainConfig:
    """Build a minimal CPU TrainConfig for synthetic data.

    Input: out_dir Path; overrides applied last (kwargs win).
    Output: TrainConfig (validated by __post_init__).
    """
    base = dict(
        backbone="resnet18", variant="tsn_avg", num_verbs=8, T=2,
        img_size=32, optimizer="adamw", lr=1e-3, weight_decay=0.0,
        lr_schedule="cosine", epochs=2, batch_size=8, amp=False,
        class_weights=False, seed=0, grad_clip=None,
        out_dir=str(out_dir), overfit_batch=None, device="cpu",
        data_root="/tmp", annotations_dir="/tmp", videos_root="/tmp",
        heldout_participants=["P22"], train_limit=None,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_two_epoch_run_learns_and_logs(tmp_path: Path):
    """2 epochs: train loss decreases; log.csv/best.pt/metrics.json exist."""
    torch.manual_seed(0)
    train_ds = SyntheticClipDataset(num_classes=8, length=32, T=2,
                                    H=32, W=32, seed=0)
    val_ds = SyntheticClipDataset(num_classes=8, length=16, T=2,
                                  H=32, W=32, seed=1)
    cfg = _tiny_cfg(tmp_path / "run1")

    best = train_run(cfg, train_ds, val_ds, verb_map={"take": 0},
                     pretrained=False)
    run = tmp_path / "run1"

    # deliverables on disk
    assert Path(best) == run / "best.pt"
    assert (run / "best.pt").exists()
    assert (run / "last.pt").exists()
    assert (run / "log.csv").exists()
    assert (run / "config.yaml").exists()
    assert (run / "verb_map.json").exists()
    assert (run / "metrics.json").exists()

    # log.csv: header + exactly `epochs` rows, documented column names
    with open(run / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert rows, "log.csv has no epoch rows"
        assert set(rows[0]) == {"epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc",
                                "val_mean_class_acc", "lr"}
    losses = [float(r["train_loss"]) for r in rows]
    assert len(losses) == cfg.epochs
    assert losses[-1] < losses[0], (
        f"train loss did not decrease over {cfg.epochs} epochs: {losses}")

    # metrics.json: exactly the documented keys, sane values
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics) == {"best_epoch", "best_val_mean_class_acc",
                            "majority_baseline_val"}
    assert 0 <= metrics["best_val_mean_class_acc"] <= 1.0
    assert metrics["majority_baseline_val"] == pytest.approx(
        2 / 8)  # synthetic labels are i % 8 -> uniform, 2 samples per class

    # checkpoint loads (weights_only: no pickle surprises) with full metadata
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) >= {"model_state_dict", "epoch", "backbone",
                         "variant", "num_verbs", "verb_map"}
    assert ckpt["backbone"] == "resnet18" and ckpt["variant"] == "tsn_avg"
    assert any(k.startswith("backbone.") for k in ckpt["model_state_dict"])
    assert any(k.startswith("head.") for k in ckpt["model_state_dict"])

    # config.yaml round-trips through the strict loader
    cfg2 = load_config(run / "config.yaml")
    assert cfg2.out_dir == cfg.out_dir and cfg2.seed == cfg.seed


def test_overfit_batch_gate(tmp_path: Path):
    """overfit_batch on the separable brightness dataset reaches >= 0.99
    train accuracy and stops before the epoch budget runs out."""
    torch.manual_seed(0)
    # channel-0 shift dataset: classes linearly separable by brightness
    train_ds = SyntheticClipDataset(num_classes=8, length=16, T=2,
                                    H=32, W=32, seed=0)
    cfg = _tiny_cfg(tmp_path / "overfit", epochs=30, lr=3e-3,
                    overfit_batch=16, batch_size=8)

    train_run(cfg, train_ds, train_ds, pretrained=False)
    run = tmp_path / "overfit"

    with open(run / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_accs = [float(r["train_acc"]) for r in rows]
    assert max(train_accs) >= 0.99, (
        f"overfit gate FAILED: train accs {train_accs}")
    assert rows[-1]["train_acc"] == f"{train_accs[-1]:.6f}"
    # early stop must actually fire (well under the 30-epoch budget)
    assert len(rows) < cfg.epochs, (
        f"no early stop: ran all {cfg.epochs} epochs (accs {train_accs})")
    # metrics.json still written with the gate run's best epoch
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0 <= metrics["best_epoch"] < len(rows)


def test_class_weights_change_training_but_run_valid(tmp_path: Path):
    """class_weights=True path executes (formula well-defined on uniform
    synthetic labels) and still writes every artefact."""
    torch.manual_seed(0)
    train_ds = SyntheticClipDataset(num_classes=4, length=16, T=2,
                                    H=32, W=32, seed=0)
    val_ds = SyntheticClipDataset(num_classes=4, length=8, T=2,
                                  H=32, W=32, seed=2)
    cfg = _tiny_cfg(tmp_path / "weighted", num_verbs=4, epochs=2,
                    class_weights=True)
    best = train_run(cfg, train_ds, val_ds, pretrained=False)
    assert Path(best).exists() and (tmp_path / "weighted" / "log.csv").exists()


def test_grad_clip_runs(tmp_path: Path):
    """grad_clip set: the loop unscales and clips without error."""
    torch.manual_seed(0)
    train_ds = SyntheticClipDataset(num_classes=4, length=8, T=2,
                                    H=32, W=32, seed=0)
    cfg = _tiny_cfg(tmp_path / "clipped", num_verbs=4, epochs=1,
                    grad_clip=1.0)
    train_run(cfg, train_ds, train_ds, pretrained=False)
    assert (tmp_path / "clipped" / "best.pt").exists()


def test_sgd_step_schedule_run(tmp_path: Path):
    """optimizer=sgd + lr_schedule=step combination executes one epoch."""
    torch.manual_seed(0)
    train_ds = SyntheticClipDataset(num_classes=4, length=8, T=2,
                                    H=32, W=32, seed=0)
    cfg = _tiny_cfg(tmp_path / "sgdstep", num_verbs=4, epochs=3,
                    optimizer="sgd", lr=1e-2, lr_schedule="step")
    train_run(cfg, train_ds, train_ds, pretrained=False)
    with open(tmp_path / "sgdstep" / "log.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # step schedule: LR halves at epoch max(1, 3//3)=1 -> logged lr drops
    lrs = [float(r["lr"]) for r in rows]
    assert lrs[0] == pytest.approx(1e-2) and lrs[-1] < lrs[0]


# ---- config loader validation (field-naming errors must be loud) ----------


def test_load_config_rejects_unknown_field(tmp_path: Path):
    """A typo'd field name fails with the valid field list, not silently."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("backbone: resnet18\nvariant: tsn_avg\nlr: 0.1\n"
                   "epochs: 1\nbatch_size: 4\nout_dir: runs/x\n"
                   "epoks: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config field"):
        load_config(bad)


def test_load_config_rejects_missing_required(tmp_path: Path):
    """Missing required field (lr) fails naming it."""
    bad = tmp_path / "bad2.yaml"
    bad.write_text("backbone: resnet18\nvariant: tsn_avg\n"
                   "epochs: 1\nbatch_size: 4\nout_dir: runs/x\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_config(bad)


def test_load_config_rejects_bad_choice(tmp_path: Path):
    """optimizer outside {adamw, sgd} fails validation."""
    bad = tmp_path / "bad3.yaml"
    bad.write_text("backbone: resnet18\nvariant: tsn_avg\nlr: 0.1\n"
                   "epochs: 1\nbatch_size: 4\nout_dir: runs/x\n"
                   "optimizer: rmsprop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer"):
        load_config(bad)


def test_repo_configs_load_and_validate():
    """Every shipped configs/train_*.yaml parses into a valid TrainConfig."""
    for name in ["train_baseline", "train_overfit32", "train_tsn",
                 "train_temporal_attn"]:
        cfg = load_config(Path("configs") / f"{name}.yaml")
        assert cfg.epochs >= 1 and cfg.num_verbs == 8
