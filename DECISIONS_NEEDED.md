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
- Evidence (three independent checks):
  1. Timestamp math, P01_01 row 0: frames 8..202, 0.14s..3.37s at 59.94 fps
     -> 3.23s * 59.94 = 193.6 ≈ 202-8=194 -> inclusive (202-8+1=195, stop-start=194; the half-frame is rounding).
  2. REAL DATA CONFIRMED (2026-10-24, subagent WP1b): 5 segments from
     P01_107 cut with ffmpeg at start_frame/fps for (stop_frame-start_frame+1)/fps
     -> frames_actual == frames_expected EXACTLY (92, 279, 39, 36, 41) in
     5/5 cuts. reports/annotation_check/labels.csv.
  3. Downscaled copy of P01_107 keeps 5972 frames; sampler indices valid end to end.
- Reference implements `stop_inclusive: bool = True` as an explicit parameter;
  the owner decides the canonical value for his own core/ version.

## D8: evidence so far (loader benchmark, 2026-10-24, subagent WP1b)
- Decode-on-the-fly from downscaled 256px mp4: **1.94 clips/s** (T=8, 24 clips,
  single process, download running concurrently — labelled estimate).
- Pre-extracted JPEG frames reload: **198.45 clips/s**; pre-extraction disk
  cost **0.33 MB/clip** (~0.33 GB per 1000 clips).
- Suggestion (labelled): pre-extraction is ~100x faster at modest disk cost;
  but 1.94 clips/s already feeds a T4-sized GPU at batch 8-16. Decide with
  the accuracy run's wall-clock in hand. Full table: reports/loader_benchmark.md.

## S2: Recipe for the final demo
- Placeholder recipe (recipes/example_salad.yaml) is active. Owner replaces
  with his 3-4 ingredient recipe using verbs the trained model recognizes
  well (see reports/verb_counts.md when choosing).

## D1/D5 latency evidence (RTX 4060, measured 2026-10-24 ~11:35 IST, commit 5da5ffc, batch 1, img 224, warmup 5 + timed 30)
- resnet34 (D1 default): T=8 fp16 **5.6 ms (180 windows/s)**; T=16 fp16 9.6 ms (104/s); T=4 fp16 4.0 ms (248/s).
- resnet18: T=8 fp16 ~5.5 ms (~190/s); T=16 fp16 5.7 ms (177/s).
- mobilenet_v2: T=8 fp32 5.6 ms (177/s); T=16 fp16 5.3 ms (190/s); fp16 no gain at T=4/T=8 (bandwidth-bound).
- Requirement was >= 3 predictions/s: every config clears it by >20x. Suggestion (labelled): keep D1 default ResNet-34 and D5 default T=8; pick by ACCURACY when the Kaggle baseline lands, latency no longer constrains. Full table: reports/latency.md, raw: runs/latency/latency.json.
