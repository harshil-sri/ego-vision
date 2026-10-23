"""Generate the shared equivalence test vectors from the REFERENCE
implementations.

Output: tests/equivalence/vectors/*.npz (+ state_machine.json). The same
vectors are replayed against whichever implementation the registry selects
(pytest tests/equivalence --impl core), so the owner can verify his own
core/ versions against the reference.

Run:
    python3 -m scripts.gen_equivalence_vectors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from reference.metrics_ref import (confusion_matrix, majority_baseline,
                                   mean_class_accuracy, per_class_accuracy,
                                   topk_accuracy)
from reference.frame_sampler_ref import sample_indices
from reference.enrollment_ref import IngredientMatcher
from reference.temporal_attention_ref import TemporalAttention
from reference.state_machine_ref import Recipe, RecipeStateMachine, Step
from tests.equivalence.common import toy_embed

VECTORS = Path(__file__).resolve().parent.parent / "tests" / "equivalence" / "vectors"


def gen_metrics() -> None:
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(20, 8)).astype(np.float32)      # (20, 8)
    scores = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = rng.integers(0, 8, size=20).astype(np.int64)          # (20,)
    pred = scores.argmax(1).astype(np.int64)                  # (20,)
    np.savez(VECTORS / "metrics.npz", scores=scores, y=y,
             top1=topk_accuracy(scores, y, 1),
             top3=topk_accuracy(scores, y, 3),
             per_class=per_class_accuracy(pred, y, 8),
             mean_class=mean_class_accuracy(pred, y, 8),
             confusion=confusion_matrix(pred, y, 8),
             majority=majority_baseline(y))


def gen_frame_sampler() -> None:
    cases = {  # name -> (start, stop, T, fps, stop_inclusive)
        "typical": (100, 180, 8, 59.94, True),
        "first_segment": (8, 202, 8, 59.94, True),
        "T_gt_len": (0, 2, 5, 50.0, True),
        "single_frame_clip": (50, 50, 4, 25.0, True),
        "boundary": (0, 99, 4, 25.0, True),
        "exclusive": (100, 180, 8, 59.94, False),
    }
    out = {}
    for name, (a, b, T, fps, inc) in cases.items():
        out[name] = sample_indices(a, b, T, fps, stop_inclusive=inc)
        out[name + "_args"] = np.array([a, b, T, fps, inc])
    np.savez(VECTORS / "frame_sampler.npz", **out)


def gen_enrollment() -> None:
    rng = np.random.default_rng(7)
    def frame(color, noise=6):
        f = np.ones((8, 8, 3), np.uint8) * np.array(color, np.uint8)
        return np.clip(f + rng.integers(-noise, noise + 1, f.shape), 0, 255
                       ).astype(np.uint8)                     # (8, 8, 3)

    red, green, blue = (220, 20, 20), (20, 220, 20), (20, 20, 220)
    enroll_sets = {"red": [frame(red) for _ in range(4)],
                   "green": [frame(green) for _ in range(4)],
                   "blue": [frame(blue) for _ in range(4)]}
    queries = {"red_q": frame(red), "green_q": frame(green),
               "grey_q": np.full((8, 8, 3), 128, np.uint8)}
    matcher = IngredientMatcher(toy_embed, threshold=0.9)
    # Enroll with exactly the frame subsets saved in the vector file (the
    # test replays these two frames per class), so prototypes match to 1e-5.
    matcher.enroll("red", [enroll_sets["red"][0], enroll_sets["red"][3]])
    matcher.enroll("green", [enroll_sets["green"][0], enroll_sets["green"][1]])
    matcher.enroll("blue", [enroll_sets["blue"][0], enroll_sets["blue"][2]])
    results = {q: matcher.match(img) for q, img in queries.items()}
    np.savez(VECTORS / "enrollment.npz",
             threshold=np.array(0.9),
             names=np.array(matcher.names, dtype=object),
             red0=enroll_sets["red"][0], green1=enroll_sets["green"][1],
             blue2=enroll_sets["blue"][2], red3=enroll_sets["red"][3],
             green0=enroll_sets["green"][0], blue0=enroll_sets["blue"][0],
             q_red=queries["red_q"], q_green=queries["green_q"],
             q_grey=queries["grey_q"],
             match_red_name=np.array(results["red_q"][0]),
             match_red_score=np.array(results["red_q"][1]),
             match_green_name=np.array(results["green_q"][0]),
             match_green_score=np.array(results["green_q"][1]),
             match_grey_name=np.array(results["grey_q"][0]),
             match_grey_score=np.array(results["grey_q"][1]))


def gen_temporal_attention() -> None:
    torch.manual_seed(42)
    att = TemporalAttention(16)
    x = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(43))
    pooled, weights = att(x)
    np.savez(VECTORS / "temporal_attention.npz",
             x=x.numpy(),
             W_weight=att.W.weight.detach().numpy(),          # (16, 16)
             W_bias=att.W.bias.detach().numpy(),              # (16,)
             v_weight=att.v.weight.detach().numpy(),          # (1, 16)
             pooled=pooled.detach().numpy(),                  # (4, 16)
             weights=weights.detach().numpy())                # (4, 8)


def gen_state_machine() -> None:
    recipe = Recipe("vector recipe",
                    ["a", "b"], ["take", "cut", "put"],
                    [Step("take", "a"), Step("cut", "a"),
                     Step("take", "b"), Step("put", "b")])
    events = [("take", "b"),      # future -> wrong_order
              ("take", "a"),      # correct -> advance
              ("take", "a"),      # just done -> repeat
              ("open", "a"),      # unrelated -> idle
              ("cut", "a"),       # correct -> advance
              ("take", "b"),      # correct -> advance
              ("put", "b"),       # correct -> done
              ("put", "b")]       # after done -> idle
    sm = RecipeStateMachine(recipe)
    trace = []
    for i, (verb, ing) in enumerate(events):
        ev = sm.update(verb, ing, t_ns=i)
        trace.append({"verb": verb, "ingredient": ing,
                      "kind": ev.kind, "step_index": ev.step_index})
    payload = {"recipe": {"name": recipe.name,
                          "ingredients": recipe.ingredients,
                          "verbs": recipe.verbs,
                          "steps": [{"verb": s.verb, "ingredient": s.ingredient,
                                     "note": s.note} for s in recipe.steps]},
               "events": trace}
    (VECTORS / "state_machine.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    VECTORS.mkdir(parents=True, exist_ok=True)
    gen_metrics()
    gen_frame_sampler()
    gen_enrollment()
    gen_temporal_attention()
    gen_state_machine()
    print(f"vectors written to {VECTORS}")
