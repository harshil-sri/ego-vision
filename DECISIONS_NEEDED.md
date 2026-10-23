# DECISIONS NEEDED

Design decisions that belong to Harshil. Each entry: question, options,
evidence (with commands), a clearly-labelled suggestion, and what would change
the answer. Until he picks, the listed default is active (config switch, not
baked in).

---

## Forced deviation (data route) — needs his sign-off

### V1: Internal validation split instead of official validation videos
- **Question:** the only fast data route (HF mirror `awsaf49/epic_kitchens_100`)
  carries **train-split videos only**. The official validation videos are
  reachable only via the official server at 182 KB/s (~9 h per video).
- **Options:** (a) carve an internal validation from downloaded train videos
  (hold out videos, never participants already sparse); (b) spend nights
  trickle-downloading official validation videos at 182 KB/s; (c) check
  `lightly-ai/epic-kitchens-100-clips` (24 GB) for validation coverage.
- **Evidence:** reports/data_access.md (speeds, mirror inventory).
- **What I did meanwhile (reversible):** (a) — configs/data.yaml will define
  `heldout_videos`; the Dataset respects it. Training never touches held-out
  videos; evaluation reports say "internal split" on every table.
- **What would change the answer:** any route that delivers official
  validation videos at usable speed (college network, 300 Mbps WiFi, torrent).

---

## D1: Verb-model backbone
- Options: ResNet-18 / ResNet-34 / MobileNetV2 (ImageNet-pretrained).
- Evidence pending: accuracy-vs-latency table (reports/latency.md +
  runs/*/metrics.json).
- Default active: **ResNet-34**.

## D2: Train big then distil, or train small
- Options: big-then-distil / train small directly.
- Evidence: Kaggle GPU hours available (30 h/week verified working); training
  cost table once WP3 runs.
- Default active: **train small**. (Kaggle offload means "small" still trains
  comfortably.)

## D3: Handling "nothing is happening"
- Options: (a) confidence threshold, (b) background class sampled from gaps
  between segments, (c) motion gating.
- Evidence pending: replay of an idle clip per option (fires-per-minute).
- Default active: **(a), threshold 0.5**.

## D4: Where ingredient identity comes from
- Options: (a) whole frame, (b) MediaPipe hand crop, (c) fixed center
  "show zone".
- Evidence pending: matcher top-1 per option on webcam frames
  (reports/matcher_eval.md).
- Default active: **(c) show zone**.

## D5: Clip length T and stride
- Options: T in {4, 8, 16} over ~2 s windows.
- Evidence pending: accuracy vs inference time table.
- Default active: **T=8**.

## D6: Smoothing of live predictions
- Options: majority vote over last k / EMA of probabilities / hysteresis.
- Evidence pending: recorded-clip replay per policy.
- Default active: **EMA**.

## D7: Scan-phase embedding model
- Options: DINOv2 ViT-S/14 (384-d) / CLIP / ResNet features.
- Evidence pending: held-out top-1 on 4+ ingredients.
- Default active: **DINOv2-Small**. License note for README: repo weights
  listed CC-BY-NC in one source, HF `dinov2-small` Apache 2.0 in another;
  fine for a personal demo, check before any commercial use.

## D8: Training resolution and preprocessing
- Options: offline downscale to short side 256 (fps and frame count kept) /
  on-the-fly 1080p decode / pre-extracted frames.
- Evidence pending: loader benchmark (reports/loader_benchmark.md).
- Default active: **offline 256**.

## S1: `stop_frame` convention in the frame sampler
- Options: inclusive (a segment covers start_frame..stop_frame) / exclusive.
- Evidence: EPIC annotation semantics — the CSVs pair start_frame/stop_frame
  with start/stop timestamps whose durations match stop-start (verify per
  video with ffprobe once first video lands; P01_01 row 0: frames 8..202,
  timestamps 0.14s..3.37s at 59.94 fps -> 202-8=194 frames vs 3.23s*59.94=193.6
  -> consistent with **inclusive stop_frame**).
- Reference implements `stop_inclusive: bool = True` as an explicit parameter;
  the owner decides the canonical value for his own core/ version.

## S2: Recipe for the final demo
- Placeholder recipe (recipes/example_salad.yaml) is active. Owner replaces
  with his 3-4 ingredient recipe using verbs the trained model recognizes
  well (see reports/verb_counts.md when choosing).
