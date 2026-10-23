"""Verb sources for the engine: trained model (Tiers 2/3) or keyboard
(Tier 1). Lazy model loading so the app starts without a checkpoint.

Contract: predict(window: list[FramePacket]) -> np.ndarray (num_verbs,)
float64 softmax probabilities, or None when no model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class KeyboardVerbSource:
    """Tier 1: no model. Verbs arrive via engine.inject_verb() from the UI
    (number keys / buttons / REST). predict() always returns None."""

    name = "keyboard"

    def predict(self, window) -> np.ndarray | None:
        return None


class ModelVerbSource:
    """Tiers 2/3: a trained VerbModel checkpoint over the sliding window.

    Loads src.models.variants.VerbModel lazily (keeps the app importable
    before the training code exists) plus the verb_map.json saved with the
    run. Inference is eval-mode, no-grad, softmaxed (BatchNorm trap
    covered by tests on the model side).
    """

    name = "model"

    def __init__(self, run_dir, device: str = "cpu") -> None:
        """run_dir: a runs/<id>/ directory holding config.yaml copy,
        best.pt (or checkpoint.pt) and verb_map.json."""
        import torch, yaml
        from src.models.variants import VerbModel

        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        verb_map_path = run_dir / "verb_map.json"
        if not verb_map_path.exists():
            raise FileNotFoundError(f"no verb_map.json in {run_dir}")
        import json
        self.verb_map: dict[str, int] = json.loads(verb_map_path.read_text())
        self.verbs = sorted(self.verb_map, key=self.verb_map.get)  # id -> name
        self.model = VerbModel(backbone_name=cfg["backbone"],
                               num_verbs=len(self.verbs),
                               variant=cfg["variant"])
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(device).eval()
        self.device = device
        self.img_size = int(cfg.get("img_size", 224))

    def predict(self, window) -> np.ndarray | None:
        """window: list[FramePacket] (T frames, BGR uint8) -> (num_verbs,)
        probabilities; None for an empty window."""
        if not window:
            return None
        import cv2, torch
        frames = np.stack([
            cv2.resize(p.image, (self.img_size, self.img_size))
            for p in window])                                     # (T,S,S,3)
        t = torch.from_numpy(frames[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
        t = t.float().div_(255.0).to(self.device)                 # (1,3,T,S,S)
        # VerbModel expects (B,T,C,H,W); we built (1,C,T,S,S) -> fix order.
        t = t.permute(0, 2, 1, 3, 4).contiguous()                 # (1,T,C,S,S)
        with torch.no_grad():
            logits = self.model(t)                                # (1, V)
            probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu().numpy().astype(np.float64)             # (V,)


def make_verb_source(mode: str, run_dir: str | None = None,
                     device: str = "cpu"):
    """mode 'keyboard' -> KeyboardVerbSource; 'clip'/'live' -> ModelVerbSource.

    Missing checkpoint in model modes raises FileNotFoundError with a clear
    message (the app surfaces it, never a crash)."""
    if mode == "keyboard":
        return KeyboardVerbSource()
    if run_dir is None:
        raise FileNotFoundError(
            f"--mode {mode} needs --run-dir runs/<id> with a trained "
            f"checkpoint (Tier 1 keyboard mode needs no model)")
    return ModelVerbSource(run_dir, device=device)
