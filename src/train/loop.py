"""The training loop: train_run(cfg, train_ds, val_ds) -> best checkpoint path.

Everything knob-like lives in TrainConfig (configs/*.yaml); this module has
no decisions of its own beyond documented constants (SGD momentum, the step
schedule's step/gamma), which are called out below and in the config
docstring.

Class-weight formula (cfg.class_weights = true):
    n_c = number of train samples with label c, N = total train samples,
    C = num_verbs.
    w_c = N / (C * n_c)          (inverse frequency, normalised so that the
                                  mean weight over present classes is 1)
    n_c = 0      -> w_c = 1.0    (class never seen: gets no gradient anyway;
                                  1.0 keeps the tensor well-defined)

Step-LR constants (lr_schedule = 'step'; a documented default, owner may
change): StepLR(step_size = max(1, epochs // 3), gamma = 0.5), i.e. the LR
halves every third of the run.

overfit_batch gate (cfg.overfit_batch = N > 0):
    train AND validate on the FIRST N samples of train_ds (Subset indices
    0..N-1 — deterministic, no shuffle in the subset choice), stop early the
    epoch training accuracy reaches >= 0.99 (or when epochs run out). Driving
    this to 100% on real data proves the loop CAN learn; failing it means a
    bug, not "hard data".

Outputs written into cfg.out_dir:
    config.yaml   — copy of the cfg (dataclasses.asdict via save_config)
    verb_map.json — {"verb": id} (only when the verb_map argument is given)
    log.csv       — per epoch: epoch,train_loss,train_acc,val_loss,val_acc,
                    val_mean_class_acc,lr
    best.pt       — checkpoint dict (see _save_checkpoint) with the best
                    val_mean_class_acc seen so far
    last.pt       — same structure, final epoch (so a crashed run is usable)
    metrics.json  — {best_epoch, best_val_mean_class_acc,
                    majority_baseline_val}
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from src import registry
from src.models.variants import VerbModel
from src.train.config import TrainConfig, save_config

# Fixed, documented constants (not config fields; changing them is an owner
# decision — listed in this module docstring so they are easy to find).
SGD_MOMENTUM = 0.9      # standard SGD momentum; Nesterov off
STEP_GAMMA = 0.5        # step schedule multiplier
OVERFIT_TARGET_ACC = 0.99  # early-stop threshold for overfit_batch mode


def set_seed(seed: int) -> None:
    """Seed every RNG the loop touches.

    Input: seed int. Output: None. Seeds python.random, numpy, torch (and
    torch.cuda when available) so a rerun with the same seed reproduces.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # no-op on CPU-only hosts
        torch.cuda.manual_seed_all(seed)


def _dataset_labels(ds) -> np.ndarray:
    """Collect the int labels of a dataset WITHOUT decoding clips.

    Input: ds — a torch Dataset yielding (clip, int_label) pairs.
    Output: np.ndarray (N,) int64.
    Fast paths (no __getitem__ call, so no video decode):
        Subset -> labels of the parent, indexed by subset indices;
        EpicVerbDataset -> segments_df['verb'].map(verb_map);
    Generic fallback iterates ds (fine for synthetic sets; never reached
    for EpicVerbDataset because it exposes the attributes above).
    """
    if isinstance(ds, torch.utils.data.Subset):
        base = _dataset_labels(ds.dataset)              # (N_parent,) int64
        return base[np.asarray(list(ds.indices), dtype=np.int64)]
    verb_map = getattr(ds, "verb_map", None)
    segments_df = getattr(ds, "segments_df", None)
    if verb_map is not None and segments_df is not None:
        # (N,) int64: verb strings mapped through the alphabetical-id map
        return (segments_df["verb"].map(verb_map)
                .to_numpy(dtype=np.int64))
    # Generic path: index explicitly. NEVER `for ... in ds` — python's
    # legacy sequence-protocol iteration only stops at IndexError, and
    # Dataset classes without a raising __getitem__ (SyntheticClipDataset)
    # would iterate forever.
    return np.asarray([int(ds[i][1]) for i in range(len(ds))],
                      dtype=np.int64)


def _class_weights(labels: np.ndarray, num_classes: int,
                   device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights (formula in the module docstring).

    Inputs: labels (N,) int64; num_classes int C; device torch.device.
    Output: torch.float32 (C,) on `device`, mean 1 over present classes.
    """
    counts = np.bincount(labels, minlength=num_classes)  # (C,) int64
    n_total = int(counts.sum())
    weights = np.ones(num_classes, dtype=np.float64)     # (C,) default 1.0
    present = counts > 0
    # w_c = N / (C * n_c); n_c == 0 keeps the default 1.0
    weights[present] = n_total / (num_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)  # (C,)


def _build_optimizer(cfg: TrainConfig, model: torch.nn.Module
                     ) -> torch.optim.Optimizer:
    """Optimizer per cfg.optimizer (documented constants in module docstring).

    Input: cfg TrainConfig; model nn.Module (parameters() consumed).
    Output: torch.optim.Optimizer (AdamW or SGD(momentum=0.9)).
    """
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                           momentum=SGD_MOMENTUM,
                           weight_decay=cfg.weight_decay)


def _build_scheduler(cfg: TrainConfig,
                     optimizer: torch.optim.Optimizer
                     ) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-epoch LR scheduler per cfg.lr_schedule (None if unknown-off).

    Input: cfg; optimizer to schedule. Output: scheduler stepping once per
    EPOCH. cosine -> CosineAnnealingLR(T_max=epochs); step -> halve every
    max(1, epochs // 3) epochs (STEP_GAMMA).
    """
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs)
    if cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.epochs // 3), gamma=STEP_GAMMA)
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")


def _evaluate_loss_acc(model: torch.nn.Module,
                       loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       metrics,
                       num_classes: int,
                       criterion: torch.nn.Module | None = None
                       ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Eval-mode pass over a loader.

    Inputs:
        model: VerbModel (set to .eval() here — BN running stats, no dropout).
        loader: yields (clips (B,T,C,H,W) float32, labels (B,) int64).
        device, metrics (registry metrics module), num_classes int.
        criterion: optional unweighted-loss fn(logits (B,C), y (B,)) -> scalar
            tensor; None skips loss computation.
    Outputs: (loss float or nan, top1_acc float, mean_class_acc float,
              pred (N,) int64, y (N,) int64).
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_pred, all_y = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)              # (B,T,C,H,W) float32
            labels = labels.to(device)            # (B,) int64
            logits = model(clips)                 # (B, num_classes) float32
            if criterion is not None:
                total_loss += float(criterion(logits, labels))  # scalar
                n_batches += 1
            all_pred.append(logits.argmax(dim=1).cpu().numpy())  # (B,) int64
            all_y.append(labels.cpu().numpy())                    # (B,) int64

    pred = np.concatenate(all_pred) if all_pred else np.zeros(0, np.int64)
    y = np.concatenate(all_y) if all_y else np.zeros(0, np.int64)
    loss = total_loss / n_batches if n_batches else float("nan")
    if pred.size:
        top1 = metrics.topk_accuracy(logits_scores(pred, num_classes), y, k=1)
        mean_class = metrics.mean_class_accuracy(pred, y, num_classes)
    else:  # empty loader (e.g. no validation videos on disk): well-defined 0s
        top1, mean_class = 0.0, 0.0
    return loss, top1, mean_class, pred, y


def logits_scores(pred: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot 'scores' so topk_accuracy(argmax==truth) == plain accuracy.

    Inputs: pred (N,) int64 class ids; num_classes int C.
    Output: np.ndarray (N, C) float64, 1.0 at the predicted class.
    """
    scores = np.zeros((pred.shape[0], num_classes), dtype=np.float64)  # (N,C)
    scores[np.arange(pred.shape[0]), pred] = 1.0
    return scores


def _save_checkpoint(path: Path, model: torch.nn.Module, epoch: int,
                     cfg: TrainConfig, verb_map: dict | None) -> None:
    """Write one checkpoint dict.

    Input: path; model; epoch int; cfg; verb_map dict|None.
    Output: file with {'model_state_dict', 'epoch', 'backbone', 'variant',
        'num_verbs', 'verb_map'} — everything eval_verb_model needs to
        rebuild the model without the config file.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "backbone": cfg.backbone,
        "variant": cfg.variant,
        "num_verbs": cfg.num_verbs,
        "verb_map": dict(verb_map) if verb_map else None,
    }, path)


def train_run(cfg: TrainConfig, train_ds, val_ds,
              verb_map: dict | None = None, pretrained: bool = True,
              num_workers: int = 0) -> str:
    """Train one VerbModel run and return the BEST checkpoint path.

    Inputs:
        cfg: TrainConfig (validated; see src/train/config.py).
        train_ds, val_ds: torch Datasets yielding (clip (T,C,H,W) float32,
            label int) — EpicVerbDataset or SyntheticClipDataset.
        verb_map: optional {verb: id} saved into the run dir (verb_map.json)
            and into checkpoints; None skips the file.
        pretrained: backbone ImageNet init (False for synthetic smoke tests).
        num_workers: DataLoader workers (0 = in-process; deterministic).
    Output: str path of <out_dir>/best.pt (best val mean-class accuracy).

    Raises RuntimeError when cfg.device='cuda' but CUDA is unavailable, and
    ValueError when train_ds is empty (nothing to learn from).
    """
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.device='cuda' but torch.cuda.is_available() is False; "
            "set device: cpu in the config or fix the CUDA install.")

    # Metrics through the registry so the owner's core/ drop-in applies here.
    metrics = registry.get("metrics")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")  # copy of the exact run cfg
    if verb_map is not None:
        with open(out_dir / "verb_map.json", "w", encoding="utf-8") as f:
            json.dump(dict(verb_map), f, indent=2, sort_keys=True)

    # overfit gate: first N samples as BOTH train and val (module docstring).
    if cfg.overfit_batch is not None and cfg.overfit_batch > 0:
        n = min(int(cfg.overfit_batch), len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, range(n))
        val_ds = train_ds  # same first-N subset object; no hidden divergence
    if len(train_ds) == 0:
        raise ValueError("train_ds is empty — nothing to train on "
                         "(check that videos exist under cfg.videos_root).")

    # Seeded shuffling: the loader's generator, not global state, drives it.
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)

    model = VerbModel(cfg.backbone, cfg.num_verbs, cfg.variant,
                      pretrained=pretrained).to(device)

    # Class weights from TRAIN labels only (val must never influence them).
    train_labels = _dataset_labels(train_ds)  # (N,) int64
    if cfg.class_weights:
        weight = _class_weights(train_labels, cfg.num_verbs, device)  # (C,)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    val_criterion = torch.nn.CrossEntropyLoss()  # val loss: UNWEIGHTED

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    # AMP only when cfg.amp AND cuda (fp16 autocast on CPU is not a speedup).
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "val_mean_class_acc",
                                "lr"])

    val_labels = _dataset_labels(val_ds)  # (N_val,) int64
    majority_baseline_val = (
        metrics.majority_baseline(val_labels) if val_labels.size else 0.0)

    best_path = str(out_dir / "best.pt")
    best_score, best_epoch = -1.0, -1
    overfitting = bool(cfg.overfit_batch and cfg.overfit_batch > 0)

    for epoch in range(cfg.epochs):
        lr_now = optimizer.param_groups[0]["lr"]  # LR in force this epoch

        # ---- train one epoch ------------------------------------------------
        model.train()
        running_loss, n_batches = 0.0, 0
        train_pred, train_y = [], []
        for clips, labels in train_loader:
            clips = clips.to(device)      # (B,T,C,H,W) float32
            labels = labels.to(device)    # (B,) int64
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(clips)     # (B, num_verbs)
                loss = criterion(logits, labels)  # scalar
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:  # unscale first so the norm is true
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach())
            n_batches += 1
            train_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
            train_y.append(labels.detach().cpu().numpy())

        train_loss = running_loss / max(1, n_batches)
        train_pred = np.concatenate(train_pred) if train_pred else np.zeros(0, np.int64)
        train_y = np.concatenate(train_y) if train_y else np.zeros(0, np.int64)
        train_acc = (metrics.topk_accuracy(
            logits_scores(train_pred, cfg.num_verbs), train_y, k=1)
            if train_pred.size else 0.0)

        # ---- validate (metrics via registry; BN in eval mode) ----------------
        val_loss, val_acc, val_mean_class, _, _ = _evaluate_loss_acc(
            model, val_loader, device, metrics, cfg.num_verbs,
            criterion=val_criterion)

        with open(log_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}",
                                    f"{train_acc:.6f}", f"{val_loss:.6f}",
                                    f"{val_acc:.6f}",
                                    f"{val_mean_class:.6f}", f"{lr_now:.6g}"])
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} "
              f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} "
              f"val_acc {val_acc:.4f} val_mca {val_mean_class:.4f} "
              f"| lr {lr_now:.3g}", flush=True)

        # ---- checkpointing: best by val mean-class accuracy -----------------
        if val_mean_class > best_score:
            best_score, best_epoch = val_mean_class, epoch
            _save_checkpoint(out_dir / "best.pt", model, epoch, cfg, verb_map)
        _save_checkpoint(out_dir / "last.pt", model, epoch, cfg, verb_map)

        scheduler.step()

        if overfitting and train_acc >= OVERFIT_TARGET_ACC:
            print(f"overfit gate reached train_acc={train_acc:.4f} "
                  f">= {OVERFIT_TARGET_ACC} at epoch {epoch}; stopping early.",
                  flush=True)
            break

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": int(best_epoch),
                   "best_val_mean_class_acc": float(best_score),
                   "majority_baseline_val": float(majority_baseline_val)},
                  f, indent=2)
    print(f"run dir: {out_dir} (best epoch {best_epoch}, "
          f"val_mean_class_acc {best_score:.4f})", flush=True)
    return best_path
