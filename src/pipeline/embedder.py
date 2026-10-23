"""Embedder wrapper with switchable backbones (decision D7): DINOv2
ViT-S/14 (default), CLIP, ResNet features, and a deterministic toy
embedder for offline tests.

Contract (brief section 5.2):  __call__(images: list[HxWx3 uint8])
    -> np.ndarray (N, D) float32, rows L2-normalized.

Models load lazily on first use; the toy embedder needs no network and is
what the smoke tests use. CPU device is the default (matcher runs at a few
Hz; GPU is not required for the scan phase).
"""

from __future__ import annotations

import numpy as np


class ToyEmbedder:
    """Mean-colour embedder (deterministic, no deps). Output (N, 3).

    NOT a real model — its only job is making the matcher testable
    offline. Two visually different ingredients separate well if their
    mean colours differ.
    """

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """images: list of HxWx3 uint8 -> (N, 3) float32 unit rows."""
        rows = np.stack([np.asarray(img, np.uint8).reshape(-1, 3).mean(0)
                         for img in images]).astype(np.float32)   # (N, 3)
        rows /= 255.0
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        return (rows / norms).astype(np.float32)                  # (N, 3)


class DinoEmbedder:
    """DINOv2 ViT-S/14 via torch.hub (384-dim). Default for D7.

    License note (README): repo weights listed CC-BY-NC in one source,
    HF `dinov2-small` Apache 2.0 in another — fine for a personal demo.
    """

    def __init__(self, device: str = "cpu") -> None:
        import torch  # lazy: hub load only when actually selected
        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2",
                                    "dinov2_vits14").to(device).eval()
        self.dim = 384

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """images: list of HxWx3 uint8 (resized to 224) -> (N, 384) unit rows."""
        import cv2
        batch = np.stack([cv2.resize(np.asarray(img, np.uint8), (224, 224))
                          for img in images])                     # (N,224,224,3)
        batch = batch[..., ::-1]                                  # BGR->RGB
        t = self.torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).float() / 255.0
        t = t.to(self.device)                                     # (N,3,224,224)
        with self.torch.no_grad():
            feats = self.model(t)                                 # (N, 384)
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        return feats / norms                                      # (N, 384)


class ResnetEmbedder:
    """ImageNet ResNet-34 pooled features (512-dim) — baseline for D7."""

    def __init__(self, device: str = "cpu") -> None:
        import torch, torchvision
        self.torch = torch
        self.device = device
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        full = torchvision.models.resnet34(weights=weights)
        self.model = torch.nn.Sequential(*list(full.children())[:-1]).to(device).eval()
        self.dim = 512

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """images: list of HxWx3 uint8 BGR -> (N, 512) unit rows."""
        import cv2
        mean = np.array([0.485, 0.456, 0.406], np.float32) * 255
        std = np.array([0.229, 0.224, 0.225], np.float32) * 255
        batch = np.stack([cv2.resize(np.asarray(img, np.uint8), (224, 224))
                          for img in images])[..., ::-1]           # (N,...,3) RGB
        t = (batch.astype(np.float32) - mean) / std
        t = self.torch.from_numpy(t.copy()).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            feats = self.model(t).flatten(1)                      # (N, 512)
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        return feats / norms


class ClipEmbedder:
    """OpenAI CLIP image encoder via torch.hub (512-dim ViT-B/16)."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        self.torch = torch
        self.device = device
        self.model = torch.hub.load("openai/clip", "clip_vitb16",
                                    device=device).eval()
        self.dim = 512

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2
        batch = np.stack([cv2.resize(np.asarray(img, np.uint8), (224, 224))
                          for img in images])[..., ::-1]
        t = self.torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).float() / 255.0
        mean = self.torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = self.torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        t = ((t.to(self.device) - mean) / std)
        with self.torch.no_grad():
            feats = self.model.encode_image(t)                    # (N, 512)
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        return feats / norms


def make_embedder(name: str, device: str = "cpu"):
    """Factory: 'toy' | 'dinov2' | 'resnet' | 'clip' -> callable (config D7)."""
    table = {"toy": ToyEmbedder, "dinov2": DinoEmbedder,
             "resnet": ResnetEmbedder, "clip": ClipEmbedder}
    if name not in table:
        raise ValueError(f"unknown embedder '{name}'; valid: {sorted(table)}")
    cls = table[name]
    return cls() if name == "toy" else cls(device=device)
