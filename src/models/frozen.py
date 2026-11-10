"""Frozen-feature verb model: DINOv2 per-frame features + a small head.

Why this exists: fine-tuning ResNet-34 on ~2,400 clips overfit after one
epoch (val mean-class acc 0.22). Here the image backbone is NEVER trained.
Each frame becomes one fixed vector, computed once and cached; only a small
temporal head is trained on those vectors.

Two pieces:
    DinoFrameEncoder   frames -> per-frame features, frozen, no gradients
    FeatureHead        (B, T, F) features -> (B, num_verbs) logits

Feature layout per frame (F = 768 for ViT-S/14):
    [0:384]    CLS token       global summary of the frame
    [384:768]  mean patch token average over all image patches
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src import registry

#: ImageNet statistics DINOv2 was trained with (RGB order, 0..1 scale)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14          # ViT-S/14 patch size; input sides must be multiples of it
SHORT_SIDE = 224    # 16 patches

HEAD_KINDS = ("mean", "attn", "tfm", "mlp")


def target_size(h: int, w: int, short: int = SHORT_SIDE) -> tuple[int, int]:
    """Resize target that keeps aspect ratio and is divisible by the patch.

    Input: native frame height and width (ints).
    Output: (new_h, new_w) with the short side = `short` and the long side
        rounded to the nearest multiple of 14, e.g. (256, 456) -> (224, 392),
        (480, 640) -> (224, 294).
    """
    if h <= w:
        new_h = short
        new_w = max(PATCH, int(round(w * short / h / PATCH)) * PATCH)
    else:
        new_w = short
        new_h = max(PATCH, int(round(h * short / w / PATCH)) * PATCH)
    return new_h, new_w


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Resize a stack of frames for the encoder (CPU, cv2).

    Input: frames (N, H, W, 3) uint8, any channel order.
    Output: (N, h, w, 3) uint8 with (h, w) = target_size(H, W).
    """
    import cv2
    h, w = target_size(frames.shape[1], frames.shape[2])
    out = np.empty((frames.shape[0], h, w, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
    return out


class DinoFrameEncoder:
    """Frozen DINOv2 ViT-S/14. Turns RGB frames into 768-d vectors."""

    dim = 768

    def __init__(self, device: str = "cpu", half: bool | None = None) -> None:
        """device: 'cpu' or 'cuda'. half: fp16 on GPU (default: on for cuda)."""
        self.device = device
        self.half = (device == "cuda") if half is None else bool(half)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.to(device).eval()
        if self.half:
            self.model.half()
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames_rgb: torch.Tensor) -> torch.Tensor:
        """Encode already-resized frames.

        Input: frames_rgb uint8 tensor (N, h, w, 3), RGB, h and w multiples
            of 14 (use resize_frames first).
        Output: float32 tensor (N, 768) on CPU: [CLS | mean patch token].
        """
        x = frames_rgb.to(self.device).permute(0, 3, 1, 2).float() / 255.0  # (N,3,h,w)
        x = (x - self._mean) / self._std
        if self.half:
            x = x.half()
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]                   # (N, 384)
        patch = out["x_norm_patchtokens"].mean(dim=1)  # (N, P, 384) -> (N, 384)
        return torch.cat([cls, patch], dim=1).float().cpu()  # (N, 768)

    def encode_numpy(self, frames_rgb: np.ndarray, batch: int = 64) -> np.ndarray:
        """Resize + encode a numpy stack. (N, H, W, 3) uint8 RGB -> (N, 768) float32."""
        small = torch.from_numpy(resize_frames(frames_rgb))
        parts = [self.encode(small[i:i + batch]) for i in range(0, len(small), batch)]
        return torch.cat(parts).numpy()


class VideoMAEEncoder:
    """Frozen VideoMAE ViT-B, fine-tuned on Something-Something v2.

    Why this model: SSv2 labels are motion-defined ("opening something",
    "putting something into something"), so its features separate verbs
    that look identical in single frames. It has never seen EPIC-KITCHENS,
    so the held-out participants stay unseen.

    One clip of 16 frames becomes ONE 768-d vector (the mean of all
    space-time tokens after the final norm).
    """

    dim = 768
    frames = 16
    size = 224

    def __init__(self, device: str = "cpu", half: bool | None = None,
                 name: str = "MCG-NJU/videomae-base-finetuned-ssv2") -> None:
        from transformers import VideoMAEForVideoClassification
        self.device = device
        self.half = (device == "cuda") if half is None else bool(half)
        self.model = VideoMAEForVideoClassification.from_pretrained(name).to(device).eval()
        if self.half:
            self.model.half()
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, clips_rgb: torch.Tensor) -> torch.Tensor:
        """clips_rgb uint8 (B, 16, 224, 224, 3) RGB -> float32 (B, 768) on CPU."""
        x = clips_rgb.to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0  # (B,16,3,224,224)
        x = (x - self._mean) / self._std
        if self.half:
            x = x.half()
        tokens = self.model.videomae(x)[0]             # (B, 1568, 768)
        pooled = tokens.mean(dim=1)                    # (B, 768)
        if getattr(self.model, "fc_norm", None) is not None:
            pooled = self.model.fc_norm(pooled)
        return pooled.float().cpu()

    def encode_window(self, frames_rgb: np.ndarray) -> np.ndarray:
        """Live use: (16, H, W, 3) uint8 RGB -> (768,) float32.

        Resizes the short side to 224 and averages three 224-wide crops
        (left, centre, right), the same views the cached features hold.
        """
        small = torch.from_numpy(resize_frames(frames_rgb))        # (16, 224, w, 3)
        w = small.shape[2]
        xs = sorted({0, (w - self.size) // 2, w - self.size})
        clips = torch.stack([small[:, :, x:x + self.size] for x in xs])  # (C,16,224,224,3)
        return self.encode(clips).mean(dim=0).numpy()


class FeatureHead(nn.Module):
    """Temporal head over cached per-frame features.

    Input:  x float tensor (B, T, F)   F = feature dim (768)
    Output: logits float tensor (B, num_verbs)

    Steps shared by every kind:
        1. optional motion channel: d_t = x_t - x_(t-1) (zeros at t = 0),
           concatenated to x_t                     (B, T, F) -> (B, T, 2F)
        2. project to a smaller width H            (B, T, 2F) -> (B, T, H)
        3. pool over time (the part that differs)  (B, T, H) -> (B, H)
        4. linear classifier                       (B, H) -> (B, num_verbs)

    kind:
        'mean'  average over T. Order-blind: "open" and "close" look alike
                unless the motion channel separates them.
        'attn'  the registry's TemporalAttention (reference/ or core/):
                a learned weighted average over T. Still order-blind.
        'tfm'   small Transformer encoder with learned position embeddings
                and a class token. Sees frame order.
        'mlp'   for clip-level features (x is (B, F), no time axis): one
                hidden layer, then the classifier. No motion channel.
    """

    def __init__(self, in_dim: int, num_verbs: int, kind: str = "tfm",
                 hidden: int = 256, motion: bool = True, dropout: float = 0.3,
                 layers: int = 2, heads: int = 4, max_T: int = 32) -> None:
        super().__init__()
        if kind not in HEAD_KINDS:
            raise ValueError(f"kind must be one of {HEAD_KINDS}, got {kind!r}")
        self.kind, self.motion = kind, bool(motion) and kind != "mlp"
        self.in_dim, self.num_verbs, self.hidden = in_dim, num_verbs, hidden
        proj_in = in_dim * 2 if self.motion else in_dim
        self.norm_in = nn.LayerNorm(proj_in)
        self.proj = nn.Linear(proj_in, hidden)
        self.drop = nn.Dropout(dropout)
        if kind == "attn":
            self.attn = registry.get_class("temporal_attention")(hidden)
        elif kind == "tfm":
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            self.pos = nn.Parameter(torch.zeros(1, max_T + 1, hidden))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm_out = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, num_verbs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, F) -> logits (B, num_verbs). For kind 'mlp': x (B, F)."""
        if self.kind == "mlp":
            h = self.drop(torch.relu(self.proj(self.norm_in(x))))  # (B, H)
            return self.fc(self.drop(self.norm_out(h)))
        if self.motion:
            diff = torch.zeros_like(x)
            diff[:, 1:] = x[:, 1:] - x[:, :-1]          # (B, T, F)
            x = torch.cat([x, diff], dim=-1)            # (B, T, 2F)
        h = self.drop(torch.relu(self.proj(self.norm_in(x))))  # (B, T, H)
        if self.kind == "mean":
            pooled = h.mean(dim=1)                      # (B, H)
        elif self.kind == "attn":
            pooled, _ = self.attn(h)                    # (B, H)
        else:
            B, T, _ = h.shape
            tokens = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)  # (B, T+1, H)
            tokens = tokens + self.pos[:, :T + 1]
            pooled = self.encoder(tokens)[:, 0]         # class token (B, H)
        return self.fc(self.drop(self.norm_out(pooled)))  # (B, num_verbs)

    def config(self) -> dict:
        """Constructor arguments, saved in the checkpoint for reloading."""
        return {"in_dim": self.in_dim, "num_verbs": self.num_verbs,
                "kind": self.kind, "hidden": self.hidden, "motion": self.motion}
