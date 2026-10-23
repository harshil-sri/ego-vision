"""VerbModel: the three clip-classification variants compared in WP3.

All variants share the same skeleton:

    clips (B, T, C, H, W) float32
      -> backbone on the flattened frames (B*T, C, H, W) -> (B*T, D)
      -> reshape to per-frame features (B, T, D)
      -> pool over T (variant-specific)
      -> Linear head -> logits (B, num_verbs) float32

Variants (D1/D5 comparisons; the choice is the owner's, config-driven):
    'single_frame': use ONLY the middle frame's features (index T//2).
        Cheapest at inference (the backbone runs on 1 frame). Which single
        frame to use is an open owner decision (middle frame is the default
        here); the alternative "average of per-frame logits" is NOT what this
        variant does -- it is listed in DECISIONS_NEEDED.md terms as D-list
        subject to owner decision.
    'tsn_avg': mean of per-frame features (B,T,D) -> (B,D), then the head
        (TSN-style consensus, but on features not logits).
    'temporal_attn': attention pooling over T via the registry's
        TemporalAttention(D) (reference/ or core/, per configs/impl.yaml),
        then the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src import registry
from src.models.backbones import make_backbone

#: valid variant names (kept module-level so CLIs/tests can enumerate them)
VARIANTS = ("single_frame", "tsn_avg", "temporal_attn")


class VerbModel(nn.Module):
    """Clip -> verb-logits model with a swappable temporal variant.

    __init__ inputs:
        backbone_name: str passed to make_backbone ('resnet18', 'resnet34',
            'mobilenet_v2').
        num_verbs: int C, number of output classes (8 for the demo verbs).
        variant: one of VARIANTS ('single_frame' | 'tsn_avg' |
            'temporal_attn'); anything else raises ValueError.
        pretrained: bool, forwarded to make_backbone (ImageNet weights).

    Attributes:
        backbone: nn.Module mapping (B,C,H,W)->(B, feature_dim) float32.
        feature_dim: int D (512 ResNet / 1280 MobileNetV2).
        head: nn.Linear(D, num_verbs).
        attn: TemporalAttention (temporal_attn variant only), built through
            registry.get_class so the owner's core/ version drops in.

    forward input/output:
        clips: float tensor (B, T, C, H, W) (any T >= 1).
        returns: logits float32 (B, num_verbs).
    """

    def __init__(self, backbone_name: str, num_verbs: int,
                 variant: str, pretrained: bool = True) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if num_verbs < 1:
            raise ValueError(f"num_verbs must be >= 1, got {num_verbs}")

        self.backbone_name = str(backbone_name)
        self.num_verbs = int(num_verbs)
        self.variant = str(variant)

        self.backbone, self.feature_dim = make_backbone(backbone_name, pretrained)
        self.head = nn.Linear(self.feature_dim, self.num_verbs)  # (D -> C)

        if self.variant == "temporal_attn":
            # Registry fetch: reference/ implementation now, core/ later.
            attention_cls = registry.get_class("temporal_attention")
            self.attn = attention_cls(self.feature_dim)  # (B,T,D)->( (B,D),(B,T) )

    def extract_features(self, clips: torch.Tensor) -> torch.Tensor:
        """Per-frame backbone features.

        Input: clips float tensor (B, T, C, H, W).
        Output: float tensor (B, T, D) with D = self.feature_dim.
        """
        B, T, C, H, W = clips.shape           # scalars
        frames = clips.reshape(B * T, C, H, W)  # (B*T, C, H, W) backbone batch
        feats = self.backbone(frames)           # (B*T, D)
        return feats.reshape(B, T, self.feature_dim)  # (B, T, D)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Clip batch -> verb logits.

        Input: clips float tensor (B, T, C, H, W), T >= 1.
        Output: logits float32 (B, num_verbs) (raw scores; softmax happens
            in predict_probs / the loss, never here).
        """
        feats = self.extract_features(clips)  # (B, T, D)

        if self.variant == "single_frame":
            # Middle frame ONLY (documented default; owner may change which
            # frame). Backbone still trains through this single-frame path.
            pooled = feats[:, feats.shape[1] // 2]        # (B, D)
        elif self.variant == "tsn_avg":
            pooled = feats.mean(dim=1)                    # (B, D) mean over T
        else:  # temporal_attn (validated in __init__)
            pooled, _weights = self.attn(feats)           # (B, D), (B, T)
        return self.head(pooled)                          # (B, num_verbs)

    @torch.no_grad()
    def predict_probs(self, clip: torch.Tensor) -> torch.Tensor:
        """Single-clip inference helper (the shape the live app uses).

        Input: clip float tensor (T, C, H, W) — ONE clip, no batch axis.
        Output: probabilities float32 (num_verbs,), softmax over classes,
            summing to 1.

        Side effect (intentional and tested): the model is left in eval()
        mode. This is the BatchNorm batch-size-1 trap fix — calling this
        after model.train() anywhere would otherwise silently use batch
        statistics from a single clip. Callers that want to resume training
        must call .train() again themselves.
        """
        self.eval()  # BN uses running stats; Dropout off (never re-enters train)
        logits = self.forward(clip.unsqueeze(0))  # (1, num_verbs)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_verbs,)
        return probs.to(torch.float32)
