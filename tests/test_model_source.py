"""ModelVerbSource: checkpoint loading (train's save format) and the
predict() tensor contract (RGB, (1,T,3,S,S), [0,1]).

Regression history: predict() once permuted a stacked (T,S,S,3) batch as a
single frame and crashed every Tier-2/3 session at the first model tick;
loading once looked for state["model"] while src/train/loop.py saves
"model_state_dict". Both found by the first real Tier-2 dry run.
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


@pytest.fixture()
def run_dir(tmp_path):
    from src.models.variants import VerbModel
    model = VerbModel(backbone_name="resnet18", num_verbs=len(VERBS),
                      variant="single_frame")
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0,
                "backbone": "resnet18", "variant": "single_frame",
                "num_verbs": len(VERBS), "verb_map": VERBS},
               tmp_path / "best.pt")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {"backbone": "resnet18", "variant": "single_frame", "img_size": 64}))
    (tmp_path / "verb_map.json").write_text(json.dumps(VERBS))
    return tmp_path


def test_loads_train_checkpoint_format(run_dir):
    from src.pipeline.model_source import ModelVerbSource
    src = ModelVerbSource(str(run_dir), device="cpu")
    assert sorted(src.verbs) == sorted(VERBS)
    assert src.model.training is False  # eval mode (BatchNorm trap)


def test_predict_over_a_window(run_dir):
    from src.pipeline.model_source import ModelVerbSource
    src = ModelVerbSource(str(run_dir), device="cpu")
    rng = np.random.default_rng(0)
    window = [_Packet(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
              for _ in range(8)]
    probs = src.predict(window)
    assert probs is not None
    assert probs.shape == (len(VERBS),)
    assert probs.dtype == np.float64
    assert (probs >= 0).all() and probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert src.predict([]) is None


def test_missing_run_dir_is_a_clear_error(tmp_path):
    from src.pipeline.model_source import make_verb_source
    with pytest.raises(FileNotFoundError):
        make_verb_source("clip", None)
