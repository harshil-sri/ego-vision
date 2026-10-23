"""TrainConfig dataclass + strict YAML loader.

One config file fully describes one training run (no decisions baked into
the loop). load_config validates FIELD NAMES (typos must fail loudly, not
silently fall back to defaults) and every choice/type constraint.

YAML null maps to None for the optional fields (grad_clip, overfit_batch,
train_limit). Unknown keys are rejected with the list of valid field names;
missing required fields (no safe default: backbone, variant, lr, epochs,
batch_size, out_dir) are rejected the same way.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: variant names allowed in TrainConfig.variant (mirrors src/models/variants.py)
VARIANT_NAMES = ("single_frame", "tsn_avg", "temporal_attn")
#: optimizers the training loop knows how to build
OPTIMIZERS = ("adamw", "sgd")
#: LR schedule families the loop knows how to build
LR_SCHEDULES = ("cosine", "step")
#: allowed device strings (cuda is resolved lazily by the loop, not here)
DEVICES = ("cpu", "cuda")

#: fields without a safe default — a config missing one of these is a typo or
#: an incomplete file, and fails with a clear message.
REQUIRED_FIELDS = ("backbone", "variant", "lr", "epochs", "batch_size",
                   "out_dir")


@dataclass
class TrainConfig:
    """All knobs of one training run. Types in brackets; see load_config.

    backbone: str, one of src.models.backbones.supported_backbones().
    variant: str, one of VARIANT_NAMES.
    num_verbs: int >= 1 (8 for the demo verbs).
    T: int >= 1, frames per clip (D5: 4/8/16; default 8).
    img_size: int >= 1, short-side resize for the dataset transform.
    optimizer: 'adamw' | 'sgd'.
    lr: float > 0, peak learning rate.
    weight_decay: float >= 0.
    lr_schedule: 'cosine' | 'step'
        (step = halve LR every max(1, epochs // 3) epochs; documented in
        src/train/loop.py — changing it is an owner decision).
    epochs: int >= 1.
    batch_size: int >= 1.
    amp: bool, mixed precision (only ever ACTIVE on device='cuda').
    class_weights: bool, inverse-frequency class weights (formula in loop.py).
    seed: int, seeds torch/numpy/random + the DataLoader shuffler.
    grad_clip: float > 0 | None, max-norm for clip_grad_norm_ (None = off).
    out_dir: str/Path, run directory (created by train_run).
    overfit_batch: int > 0 | None — if set, train AND validate on the FIRST
        N dataset samples only, stop early at train_acc >= 0.99 (gate mode).
    device: 'cpu' | 'cuda'.
    data_root: str/Path, parent of the EPIC data roots (informational; the
        three fields below are what the loaders actually use).
    annotations_dir: str/Path, epic-kitchens-100-annotations clone.
    videos_root: str/Path, root the Dataset reads videos from — point this
        at the DOWNSCALED root (EPIC-256) for training; layout and frame
        counts match the original root.
    heldout_participants: list[str], e.g. ["P22", "P30"] (V1 internal split).
    train_limit: int > 0 | None, cap on train SEGMENTS for quick runs.
    """

    backbone: str
    variant: str
    num_verbs: int = 8
    T: int = 8
    img_size: int = 224
    optimizer: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    epochs: int = 12
    batch_size: int = 16
    amp: bool = False
    class_weights: bool = True
    seed: int = 42
    grad_clip: float | None = None
    out_dir: str = "runs/baseline"
    overfit_batch: int | None = None
    device: str = "cpu"
    data_root: str = "/mnt/ssd/epic-kitchens"
    annotations_dir: str = "/mnt/ssd/epic-tools/epic-kitchens-100-annotations"
    videos_root: str = "/mnt/ssd/epic-kitchens/EPIC-256"
    heldout_participants: list = field(default_factory=lambda: ["P22", "P30"])
    train_limit: int | None = None

    def __post_init__(self) -> None:
        """Validate types/choices (field names are validated by load_config;
        direct construction gets the same checks here so nothing slips in)."""
        _validate(self)


def _validate(cfg: TrainConfig) -> None:
    """Raise ValueError on any invalid field value.

    Input: cfg (TrainConfig, fields already assigned).
    Output: None (raises on the first problem found, with the field name).
    """
    from src.models.backbones import supported_backbones

    problems: list[str] = []

    if cfg.backbone not in supported_backbones():
        problems.append(f"backbone {cfg.backbone!r} not in {supported_backbones()}")
    if cfg.variant not in VARIANT_NAMES:
        problems.append(f"variant {cfg.variant!r} not in {list(VARIANT_NAMES)}")
    if cfg.optimizer not in OPTIMIZERS:
        problems.append(f"optimizer {cfg.optimizer!r} not in {list(OPTIMIZERS)}")
    if cfg.lr_schedule not in LR_SCHEDULES:
        problems.append(f"lr_schedule {cfg.lr_schedule!r} not in {list(LR_SCHEDULES)}")
    if cfg.device not in DEVICES:
        problems.append(f"device {cfg.device!r} not in {list(DEVICES)}")
    for name, value in [("num_verbs", cfg.num_verbs), ("T", cfg.T),
                        ("img_size", cfg.img_size), ("epochs", cfg.epochs),
                        ("batch_size", cfg.batch_size)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"{name} must be an int >= 1, got {value!r}")
    if not (isinstance(cfg.lr, (int, float)) and not isinstance(cfg.lr, bool)
            and cfg.lr > 0):
        problems.append(f"lr must be a number > 0, got {cfg.lr!r}")
    if not (isinstance(cfg.weight_decay, (int, float))
            and not isinstance(cfg.weight_decay, bool) and cfg.weight_decay >= 0):
        problems.append(f"weight_decay must be a number >= 0, got {cfg.weight_decay!r}")
    for name in ("amp", "class_weights"):
        if not isinstance(getattr(cfg, name), bool):
            problems.append(f"{name} must be a bool, got {getattr(cfg, name)!r}")
    if cfg.grad_clip is not None and not (isinstance(cfg.grad_clip, (int, float))
                                          and not isinstance(cfg.grad_clip, bool)
                                          and cfg.grad_clip > 0):
        problems.append(f"grad_clip must be a number > 0 or null, got {cfg.grad_clip!r}")
    if cfg.overfit_batch is not None and (not isinstance(cfg.overfit_batch, int)
                                          or isinstance(cfg.overfit_batch, bool)
                                          or cfg.overfit_batch < 1):
        problems.append(f"overfit_batch must be an int >= 1 or null, "
                        f"got {cfg.overfit_batch!r}")
    if cfg.train_limit is not None and (not isinstance(cfg.train_limit, int)
                                        or isinstance(cfg.train_limit, bool)
                                        or cfg.train_limit < 1):
        problems.append(f"train_limit must be an int >= 1 or null, "
                        f"got {cfg.train_limit!r}")
    if not isinstance(cfg.heldout_participants, list) or not all(
            isinstance(p, str) and p for p in cfg.heldout_participants):
        problems.append("heldout_participants must be a list of non-empty "
                        f"strings, got {cfg.heldout_participants!r}")
    for name in ("backbone", "variant", "optimizer", "lr_schedule", "device",
                 "out_dir", "data_root", "annotations_dir", "videos_root"):
        if not isinstance(getattr(cfg, name), str) or not getattr(cfg, name):
            problems.append(f"{name} must be a non-empty string, "
                            f"got {getattr(cfg, name)!r}")
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool):
        problems.append(f"seed must be an int, got {cfg.seed!r}")

    if problems:
        raise ValueError("invalid TrainConfig:\n  - " + "\n  - ".join(problems))


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from YAML with strict field-name validation.

    Input: path to a YAML file whose top-level keys are TrainConfig field
        names (values as documented on the dataclass).
    Output: TrainConfig (validated; unknown/missing/ill-typed fields raise
        ValueError naming the offending field(s)).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"train config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a YAML mapping at top level, "
                         f"got {type(raw).__name__}")

    valid = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"unknown config field(s) {unknown} in {path}; valid field "
            f"names are {sorted(valid)} (check for typos)")

    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"config {path} is missing required field(s) {missing}")

    kwargs: dict[str, Any] = dict(raw)
    # YAML null -> None for the optional numeric fields (already the sentinel
    # we want). An explicit null for heldout_participants means "use the
    # dataclass default" (the V1 split from configs/data.yaml).
    if kwargs.get("heldout_participants", None) is None:
        kwargs.pop("heldout_participants", None)

    try:
        cfg = TrainConfig(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive, same class as above
        raise ValueError(f"bad field types in {path}: {exc}") from exc
    return cfg


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Write a cfg back to YAML (train_run stores a copy per run dir).

    Input: TrainConfig; output path (parents created).
    Output: None (file contains every field, alphabetically by key).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
