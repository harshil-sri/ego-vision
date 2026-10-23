"""Verb-model building blocks (WP3): backbone factory + clip variants.

Public surface:
    make_backbone(name, pretrained) -> (nn.Module, feature_dim)   [backbones]
    VerbModel(backbone_name, num_verbs, variant, pretrained)      [variants]
"""

from src.models.backbones import make_backbone
from src.models.variants import VerbModel

__all__ = ["make_backbone", "VerbModel"]
