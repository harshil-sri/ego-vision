"""Frozen-feature verb model: head shapes, resize rule, app verb source.

The real encoders (DINOv2, VideoMAE) are NOT loaded here: they need large
downloads. A tiny fake encoder with the same interface stands in, so these
tests check the wiring (shapes, RGB order, checkpoint format), not accuracy.
"""

import json

import numpy as np
import pytest
import torch
import yaml

VERBS = {"cut": 0, "mix": 1, "put": 2, "take": 3}


class _Packet:
    """Duck-typed FramePacket: predict() only reads .image (BGR uint8)."""

    def __init__(self, image):
        self.image = image


class _FakeDino:
    """Same interface as DinoFrameEncoder; feature = mean colour, tiled."""

    dim = 768
    seen_rgb_mean = None

    def __init__(self, device="cpu", half=None):
        pass

    def encode_numpy(self, frames_rgb, batch=64):
        _FakeDino.seen_rgb_mean = frames_rgb.reshape(-1, 3).mean(0)
        m = frames_rgb.reshape(len(frames_rgb), -1, 3).mean(1) / 255.0    # (N, 3)
        return np.tile(m, (1, 256)).astype(np.float32)                    # (N, 768)


class _FakeVideoMAE:
    dim, frames, size = 768, 16, 224
    seen_shape = None

    def __init__(self, device="cpu", half=None):
        pass

    def encode_window(self, frames_rgb):
        _FakeVideoMAE.seen_shape = frames_rgb.shape
        return np.full(768, float(frames_rgb.mean()) / 255.0, dtype=np.float32)


def _write_run(tmp_path, kind, head_kind, T):
    from src.models.frozen import FeatureHead
    head = FeatureHead(768, len(VERBS), kind=head_kind)
    state = {"model_state_dict": head.state_dict(), "head": head.config(),
             "epoch": 0, "verb_map": VERBS, "kind": kind, "T": T}
    torch.save(state, tmp_path / "last.pt")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": kind, "T": T}))
    (tmp_path / "verb_map.json").write_text(json.dumps(VERBS))
    return tmp_path


@pytest.mark.parametrize("kind", ["mean", "attn", "tfm"])
@pytest.mark.parametrize("motion", [False, True])
def test_head_output_shape(kind, motion):
    from src.models.frozen import FeatureHead
    head = FeatureHead(768, 8, kind=kind, motion=motion).eval()
    assert head(torch.randn(4, 8, 768)).shape == (4, 8)


def test_mlp_head_takes_clip_vectors():
    from src.models.frozen import FeatureHead
    head = FeatureHead(768, 8, kind="mlp").eval()
    assert head.motion is False            # no time axis, so no motion channel
    assert head(torch.randn(4, 768)).shape == (4, 8)


def test_tfm_head_sees_frame_order_and_mean_head_does_not():
    from src.models.frozen import FeatureHead
    torch.manual_seed(0)
    x = torch.randn(2, 8, 768)
    rev = torch.flip(x, dims=[1])
    mean = FeatureHead(768, 8, kind="mean", motion=False).eval()
    tfm = FeatureHead(768, 8, kind="tfm", motion=False).eval()
    with torch.no_grad():
        assert torch.allclose(mean(x), mean(rev), atol=1e-5)
        assert not torch.allclose(tfm(x), tfm(rev), atol=1e-5)


def test_target_size_is_a_multiple_of_the_patch():
    from src.models.frozen import resize_frames, target_size
    assert target_size(256, 456) == (224, 392)
    assert target_size(480, 640) == (224, 294)
    out = resize_frames(np.zeros((2, 256, 456, 3), np.uint8))
    assert out.shape == (2, 224, 392, 3)


def test_frozen_dino_source(tmp_path, monkeypatch):
    import src.models.frozen as frozen
    from src.pipeline.model_source import FrozenVerbSource, make_verb_source
    monkeypatch.setattr(frozen, "DinoFrameEncoder", _FakeDino)
    run = _write_run(tmp_path, "frozen_dino", "tfm", 8)
    src = make_verb_source("clip", str(run))
    assert isinstance(src, FrozenVerbSource)
    assert src.T == 8 and src.verbs == ["cut", "mix", "put", "take"]
    assert src.head.training is False
    blue_bgr = np.zeros((48, 64, 3), np.uint8)
    blue_bgr[..., 0] = 200                      # channel 0 is BLUE in BGR
    probs = src.predict([_Packet(blue_bgr) for _ in range(8)])
    assert probs.shape == (4,) and probs.dtype == np.float64
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    # the encoder must receive RGB: blue ends up in the LAST channel
    assert _FakeDino.seen_rgb_mean.tolist() == [0.0, 0.0, 200.0]
    assert src.predict([]) is None


def test_frozen_videomae_source_resamples_to_16_frames(tmp_path, monkeypatch):
    import src.models.frozen as frozen
    from src.pipeline.model_source import FrozenVerbSource
    monkeypatch.setattr(frozen, "VideoMAEEncoder", _FakeVideoMAE)
    run = _write_run(tmp_path, "frozen_videomae", "mlp", 16)
    src = FrozenVerbSource(str(run))
    assert src.T == 16
    rng = np.random.default_rng(0)
    window = [_Packet(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)) for _ in range(5)]
    probs = src.predict(window)                 # 5 frames in, encoder needs 16
    assert _FakeVideoMAE.seen_shape == (16, 48, 64, 3)
    assert probs.shape == (4,) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_resolve_data_path(monkeypatch):
    from src.paths import resolve_data_path
    monkeypatch.setenv("EGO_DATA_ROOT", "D:/data")
    assert resolve_data_path("/mnt/ssd/epic/x") == "D:/data/epic/x"
    assert resolve_data_path("runs/baseline") == "runs/baseline"
