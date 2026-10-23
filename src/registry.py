"""Implementation registry: chooses reference/ or core/ for the five pieces.

Reads configs/impl.yaml, which maps each piece name to "reference" or "core":

    metrics: reference
    frame_sampler: reference
    enrollment: reference
    temporal_attention: reference
    state_machine: reference

"reference" loads reference/<name>_ref.py (written by the agent, study
material). "core" loads core/<name>.py (the owner's hand-written version;
the folder starts empty, so a clear error is raised until he writes it).

The rest of the codebase must fetch these pieces ONLY through this module,
never by importing reference/... or core/... directly. That keeps the owner's
drop-in replacements effective everywhere at once (app, training, tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPL_CONFIG = REPO_ROOT / "configs" / "impl.yaml"

# piece -> (reference module, reference factory, core module, core factory)
# Factories exist so callers get a uniform constructor surface even when the
# two implementations structure their code slightly differently.
_PIECES = {
    "metrics": ("reference.metrics_ref", None, "core.metrics", None),
    "frame_sampler": ("reference.frame_sampler_ref", None, "core.frame_sampler", None),
    "enrollment": ("reference.enrollment_ref", "IngredientMatcher",
                   "core.enrollment", "IngredientMatcher"),
    "temporal_attention": ("reference.temporal_attention_ref", "TemporalAttention",
                            "core.temporal_attention", "TemporalAttention"),
    "state_machine": ("reference.state_machine_ref", "RecipeStateMachine",
                      "core.state_machine", "RecipeStateMachine"),
}


@dataclass
class ImplChoice:
    """One resolved implementation choice."""
    piece: str
    impl: str            # "reference" or "core"
    module_name: str


def _load_config() -> dict[str, str]:
    with open(IMPL_CONFIG, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(_PIECES)
    if unknown:
        raise KeyError(f"unknown impl keys in {IMPL_CONFIG}: {sorted(unknown)}")
    bad = {k: v for k, v in raw.items() if v not in ("reference", "core")}
    if bad:
        raise ValueError(f"impl values must be 'reference' or 'core': {bad}")
    # Environment override, used by the equivalence test harness:
    # EGO_IMPL=core switches EVERY piece to core for this process
    # (pytest tests/equivalence --impl core sets exactly this).
    import os
    env = os.environ.get("EGO_IMPL")
    if env is not None:
        if env not in ("reference", "core"):
            raise ValueError(f"EGO_IMPL must be 'reference' or 'core', got {env}")
        raw = {piece: env for piece in _PIECES}
    return raw


def _import_piece(piece: str) -> Any:
    """Import the selected module for one piece and return the module."""
    cfg = _load_config()
    impl = cfg.get(piece, "reference")
    ref_mod, ref_fac, core_mod, core_fac = _PIECES[piece]

    if impl == "reference":
        module_name, factory = ref_mod, ref_fac
    else:
        module_name, factory = core_mod, core_fac

    import importlib
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if impl == "core":
            raise ModuleNotFoundError(
                f"impl.yaml selects core for '{piece}' but {core_mod}.py does "
                f"not exist yet (core/ starts empty; the owner writes it). "
                f"Set {piece}: reference in configs/impl.yaml to fall back."
            ) from exc
        raise

    # If the module exposes a factory wrapper, remember it for get().
    module.__dict__.setdefault("_registry_factory", factory)
    return module


def get(piece: str) -> Any:
    """Return the module implementing `piece` under the current config.

    For "metrics" and "frame_sampler" the module itself carries the
    functions; for class pieces use get_class().
    """
    if piece not in _PIECES:
        raise KeyError(f"unknown piece: {piece}; valid: {sorted(_PIECES)}")
    return _import_piece(piece)


def get_class(piece: str) -> type:
    """Return the main class for a class-style piece (enrollment,
    temporal_attention, state_machine)."""
    module = get(piece)
    factory = module._registry_factory
    if factory is None:
        raise ValueError(f"piece {piece} has no class factory")
    return getattr(module, factory)


def active_impls() -> dict[str, str]:
    """Return the currently selected impl per piece (for logging/UI)."""
    cfg = _load_config()
    return {piece: cfg.get(piece, "reference") for piece in _PIECES}
