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
        state = torch.load(ckpt_path, map_location=device)
        # checkpoint fields are authoritative; config.yaml is the fallback
        # (train saves {'model_state_dict', 'epoch', 'backbone', 'variant',
        #  'num_verbs', 'verb_map'} — see src/train/loop.py save_checkpoint)
        backbone = state.get("backbone", cfg["backbone"])
        variant = state.get("variant", cfg["variant"])
        self.model = VerbModel(backbone_name=backbone,
                               num_verbs=len(self.verbs),
                               variant=variant)
        sd = state.get("model_state_dict", state.get("model", state))
        self.model.load_state_dict(sd)
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
            for p in window])                                     # (T,S,S,3) BGR
        t = torch.from_numpy(frames[..., ::-1].copy())            # BGR -> RGB
        t = t.permute(0, 3, 1, 2).contiguous()                    # (T,3,S,S)
        t = t.unsqueeze(0).float().div_(255.0).to(self.device)    # (1,T,3,S,S)
        with torch.no_grad():
            logits = self.model(t)                                # (1, V)
            probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu().numpy().astype(np.float64)             # (V,)


class FrozenVerbSource:
    """Tiers 2/3 with a frozen encoder + small trained head.

    Loads a run directory written by scripts.train_head:
        config.yaml   model: frozen_dino | frozen_videomae, T, head config
        last.pt       head weights of the final epoch (best.pt is the
                      epoch picked on the validation set; pass
                      checkpoint="best.pt" to use it)
        verb_map.json verb -> label id

    Per tick:
        frozen_dino      T frames -> DINOv2 -> (T, 768) -> head -> (V,)
        frozen_videomae  16 frames -> VideoMAE, 3 crops averaged -> (768,)
                         -> head -> (V,)
    `T` is the number of frames the engine must sample from its window.
    `window_s` (heads from scripts.train_window_head) is the window length
    in seconds the head was trained on, else None. Such heads also have a
    "none" class; the engine never fires it.
    """

    name = "model"

    def __init__(self, run_dir, device: str = "cpu",
                 checkpoint: str | None = None) -> None:
        import json
        import torch
        import yaml
        from src.models.frozen import DinoFrameEncoder, FeatureHead, VideoMAEEncoder

        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        names = [checkpoint] if checkpoint else ["last.pt", "best.pt"]
        ckpt_path = next((run_dir / n for n in names if (run_dir / n).exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"no checkpoint ({names}) in {run_dir}")
        self.verb_map: dict[str, int] = json.loads((run_dir / "verb_map.json").read_text())
        self.verbs = sorted(self.verb_map, key=self.verb_map.get)  # id -> name
        state = torch.load(ckpt_path, map_location=device)
        self.kind = state.get("kind", cfg["model"])
        self.T = int(state.get("T", cfg.get("T", 8)))
        # heads trained on sliding windows record the window length they expect
        self.window_s = state.get("window_s", cfg.get("window_s"))
        self.head = FeatureHead(**state["head"])
        self.head.load_state_dict(state["model_state_dict"])
        self.head.to(device).eval()
        if self.kind == "frozen_videomae":
            self.encoder = VideoMAEEncoder(device=device)
        else:
            self.encoder = DinoFrameEncoder(device=device)
        self.device = device
        # VideoMAE vector (768,) of the newest window; src.pipeline.hand_state reads it
        self.last_feature: np.ndarray | None = None

    def predict(self, window) -> np.ndarray | None:
        """window: list[FramePacket] (BGR uint8) -> (num_verbs,) float64
        probabilities; None for an empty window."""
        if not window:
            return None
        import torch
        frames = np.stack([p.image for p in window])[..., ::-1]   # (T,H,W,3) RGB
        frames = np.ascontiguousarray(frames)
        if self.kind == "frozen_videomae":
            n = self.encoder.frames                               # 16
            if len(frames) != n:   # repeat or thin frames to the count the encoder needs
                frames = frames[np.linspace(0, len(frames) - 1, n).round().astype(int)]
            self.last_feature = self.encoder.encode_window(frames)        # (768,)
            x = torch.from_numpy(self.last_feature)[None]                    # (1, 768)
        else:
            x = torch.from_numpy(self.encoder.encode_numpy(frames))[None]    # (1, T, 768)
        with torch.no_grad():
            probs = torch.softmax(self.head(x.to(self.device)).float(), dim=1)[0]
        return probs.cpu().numpy().astype(np.float64)             # (V,)


def _is_frozen_run(run_dir) -> bool:
    """True when run_dir/config.yaml was written by scripts.train_head."""
    import yaml
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    return str(cfg.get("model", "")).startswith("frozen")


def make_verb_source(mode: str, run_dir: str | None = None,
                     device: str = "cpu"):
    """mode 'keyboard' -> KeyboardVerbSource; 'clip'/'live' -> a model source
    (FrozenVerbSource for scripts.train_head runs, else ModelVerbSource).

    Missing checkpoint in model modes raises FileNotFoundError with a clear
    message (the app surfaces it, never a crash)."""
    if mode == "keyboard":
        return KeyboardVerbSource()
    if run_dir is None:
        raise FileNotFoundError(
            f"--mode {mode} needs --run-dir runs/<id> with a trained "
            f"checkpoint (Tier 1 keyboard mode needs no model)")
    if _is_frozen_run(run_dir):
        return FrozenVerbSource(run_dir, device=device)
    return ModelVerbSource(run_dir, device=device)
