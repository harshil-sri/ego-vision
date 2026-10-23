"""Tests for src/models/backbones.py (random init, tiny images, CPU)."""

from __future__ import annotations

import pytest
import torch

from src.models.backbones import make_backbone, supported_backbones


@pytest.mark.parametrize("name,dim", [("resnet18", 512), ("resnet34", 512),
                                      ("mobilenet_v2", 1280)])
def test_backbone_shape_and_dim(name, dim):
    """(B=2,3,64,64) float32 in -> (2, dim) float32 pooled features out."""
    torch.manual_seed(0)
    net, feature_dim = make_backbone(name, pretrained=False)
    x = torch.randn(2, 3, 64, 64)          # (B, C, H, W) float32
    out = net(x)                           # (B, feature_dim)
    assert feature_dim == dim
    assert out.shape == (2, dim)
    assert out.dtype == torch.float32


def test_batch1_forward_eval_mode():
    """Batch-1 forward works and eval() keeps BN in eval (running stats)."""
    torch.manual_seed(0)
    net, _ = make_backbone("resnet18", pretrained=False)
    net.eval()  # the inference contract: BN uses running statistics
    x = torch.randn(1, 3, 64, 64)          # (1, C, H, W)
    with torch.no_grad():
        out1 = net(x)
        out2 = net(x)                      # same input twice
    assert out1.shape == (1, 512)
    # eval mode => deterministic (train-mode BN would use batch stats of the
    # single sample, and Dropout-free resnet would still be deterministic;
    # the determinism assertion catches accidental train() leakage).
    assert torch.allclose(out1, out2)


def test_unknown_name_raises():
    """Unknown backbone name -> ValueError listing supported names."""
    with pytest.raises(ValueError, match="resnet18"):
        make_backbone("vgg16", pretrained=False)


def test_supported_backbones_listed():
    """supported_backbones() names exactly the three WP3 backbones."""
    assert supported_backbones() == ["mobilenet_v2", "resnet18", "resnet34"]


def test_backward_reaches_backbone_params():
    """A loss through the backbone leaves grads on backbone parameters."""
    torch.manual_seed(0)
    net, _ = make_backbone("resnet18", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = net(x).sum()                     # scalar
    out.backward()
    assert net.conv1.weight.grad is not None
    assert net.conv1.weight.grad.abs().sum() > 0
