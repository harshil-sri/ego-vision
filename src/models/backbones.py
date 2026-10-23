"""Backbone factory: torchvision image classifiers -> pooled feature extractors.

Every backbone returned here maps a batch of frames
    input  (B, 3, H, W) float32, RGB, roughly ImageNet-normalised scale
    output (B, feature_dim) float32 pooled features (post-global-average-pool,
    pre-classifier), so the verb variants only ever see (B, T, D) features.

`pretrained=True` loads ImageNet weights through the torchvision weights
ENUM (not the deprecated `pretrained=True` bool). `pretrained=False` builds
randomly-initialised weights (tests, latency probing, or training from
scratch) and never touches the network.
"""

from __future__ import annotations

import torch.nn as nn

# name -> (feature_dim, torchvision weights enum class, constructor)
# feature_dim is the width of the pooled features each backbone emits.
_SUPPORTED: dict[str, tuple[int, str]] = {
    "resnet18": (512, "ResNet18_Weights.IMAGENET1K_V1"),
    "resnet34": (512, "ResNet34_Weights.IMAGENET1K_V1"),
    "mobilenet_v2": (1280, "MobileNet_V2_Weights.IMAGENET1K_V1"),
}


def make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Build one pooled-feature backbone by name.

    Inputs:
        name: one of 'resnet18', 'resnet34', 'mobilenet_v2' (case-insensitive).
        pretrained: True -> ImageNet weights via the torchvision weights enum
            (downloads to ~/.cache/torch once); False -> random init, no
            network access.

    Output: (module, feature_dim) where
        module: nn.Module with the classifier head REPLACED by nn.Identity,
            so module(x) with x (B, 3, H, W) float32 returns pooled features
            (B, feature_dim) float32 (trainable; BatchNorm layers stay
            train/eval-aware exactly as torchvision defines them).
        feature_dim: int, 512 for the ResNets, 1280 for MobileNetV2.

    Raises ValueError for an unknown name (listing the supported ones).
    """
    from torchvision import models  # local import: keeps module import cheap

    key = str(name).lower()
    if key not in _SUPPORTED:
        raise ValueError(
            f"unknown backbone {name!r}; supported: {sorted(_SUPPORTED)}")

    feature_dim, weights_enum_path = _SUPPORTED[key]

    # Resolve the weights enum lazily from torchvision (avoids importing every
    # enum at module import time). None -> random initialisation.
    weights = None
    if pretrained:
        obj: object = models
        for attr in weights_enum_path.split("."):  # e.g. models.ResNet18_Weights
            obj = getattr(obj, attr)
        weights = obj  # torchvision Weights enum member (or None above)

    if key == "resnet18":
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()          # drop the 1000-way head; keep (B,512) pooled
    elif key == "resnet34":
        net = models.resnet34(weights=weights)
        net.fc = nn.Identity()          # (B,512) pooled features out
    else:  # mobilenet_v2 (only remaining key, checked above)
        net = models.mobilenet_v2(weights=weights)
        net.classifier = nn.Identity()  # drop Dropout+Linear; (B,1280) pooled out

    return net, feature_dim


def supported_backbones() -> list[str]:
    """Return the supported backbone names. Output: list[str], sorted.

    Pure query helper so CLIs can print valid choices without poking the
    private table above.
    """
    return sorted(_SUPPORTED)
