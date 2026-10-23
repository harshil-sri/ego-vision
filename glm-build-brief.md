# EGO-KITCHEN: Build Brief for GLM and its sub-agents

*Written Sat Oct 24, 2026 for Harshil to hand to his OpenCode orchestrator (GLM 5.3). Self-contained: read this file fully before touching anything. The owner's longer plan lives in `egocentric-december-roadmap.md` (if present in the repo, read it second; this brief wins where they differ for the next 48 hours).*

**Your window:** GLM access ends about Mon Oct 26. Treat Sat Oct 24 and Sun Oct 25 as the working days and Mon Oct 26 as buffer and handoff. Plan so that something useful works at the end of each day.

---

## 0. Mission and hard rules

**Mission.** Build a polished, working demo system plus a *study library* around it, so that the owner can (a) show a stunning demo, and (b) learn every piece of it afterwards by reading working reference code and rewriting the core pieces himself.

**The demo in one paragraph.** A head-mounted webcam watches the owner cook a fixed recipe. Before cooking, a 10-second **scan phase** per ingredient (he holds each one up) stores what it looks like. During cooking, a **trained verb model** recognizes the action from the last ~2 seconds of video, a **scan-phase matcher** identifies which ingredient is in hand, and a **recipe state machine** compares each (verb, ingredient) event to a hard-coded recipe and shows the next step or flags a wrong order, all in a good-looking live UI.

### Hard rules (violating any of these makes the work unusable to him)

1. **Never create, edit or delete anything under `core/`.** That folder holds the owner's own hand-written code. You may read it. If a change there seems needed, describe it as a diff in `DECISIONS_NEEDED.md`.
2. **Do not make design decisions D1 to D8 (section 9).** Implement every option as a config switch, run the comparisons, write the evidence into `DECISIONS_NEEDED.md`, and let him choose. Until he chooses, use the listed default so the system runs.
3. **Do not look for, propose or write up a research question, research gap or paper contribution.** He deliberately wants to find that himself. No literature surveys aimed at gaps, no "future work" suggestions about novelty, no paper text. Engineering notes about your own code are fine.
4. **Never fabricate or estimate results.** Every number (accuracy, FPS, latency, file sizes, counts) must come from a command you actually ran. Log it in `STATUS.md` with the exact command, date, commit hash and seed. If something was not run, write `NOT RUN`. If a test fails, say so.
5. **Write studyable code.** Every function outside tests has a docstring stating input and output **shapes and dtypes**; every non-obvious line in reference code has a shape comment (`# (N, D) @ (D, C) -> (N, C)`). Prefer plain loops-then-vectorized progressions in `reference/` over clever one-liners. No unexplained magic.
6. **Small commits on branch `glm-window`**, clear messages, no force-pushes, no history rewriting. Commit at least every work package.
7. **Do not modify the existing camera code's behavior** (`src/capture.py`, `src/frame_buffer.py`, `src/camera_worker.py`, `scripts/camera_*`). Add to it if needed; the existing tests must keep passing.
8. **Respect the disk.** Never use more than 60% of free disk. Never download the full dataset. See section 8.
9. **When blocked for more than 15 minutes**, write the blocker to `BLOCKERS.md` (what you tried, error text, what you need) and switch to another track.
10. **Stay inside scope.** The out-of-scope list in section 11 is binding.

### How to run yourself
- Work in parallel tracks (section 6): data/model, reference library/docs, app/UI.
- Use a separate **verifier sub-agent** that re-runs the commands you claim in `STATUS.md` and the test suite before any package is marked done. Work does not grade itself.
- Keep `STATUS.md` current: what is done, what is running, what failed, next steps.

---

## 1. The owner and how to write for him

- **Harshil**, 20, B.Tech AI & ML (Sem 5), Delhi. Dictates through Wispr Flow, so transcription errors happen; interpret sensibly.
- **Strong on concepts and intuition** (finished CS231n lectures; understands kNN, softmax, CNNs, attention conceptually, with good visual intuition). **Weak on blank-file implementation** of things he has never seen working, especially matrix row/column/shape conventions (he could explain kNN but stalled writing its distance matrix because he had never seen the working version).
- **His learning method:** study a working reference with a tiny hand-traceable example, take notes, close it, write his own version the next day, test it against the reference. Your reference code and worked examples are that study material. Make them excellent.
- He wants to be able to **explain and defend every design choice and rewrite any part**. Agents are welcome for boilerplate, plumbing, UI and tests; the five core pieces are his (section 4).
- Tone in docs: direct and factual, no hype, no reassurance, no emojis.

---

## 2. The product

### 2.1 User flow
1. **Setup:** pick a recipe (a YAML file; section 12). The app lists its ingredients.
2. **Scan phase:** for each ingredient in turn, the UI says "Hold up *tomato*", shows a 10-second countdown ring, captures frames (about 3 per second), embeds them, averages them into one prototype vector per ingredient, and shows the captured thumbnails. After the last ingredient: "Ready to cook".
3. **Cooking phase:** live video with overlay. A sliding window over the last ~2 s feeds the verb model; the matcher identifies the ingredient in hand; the state machine consumes (verb, ingredient) events and advances, flags wrong order, or waits.
4. **Outputs:** recipe checklist with the current step highlighted, next-step hint, wrong-order warning, and a debug drawer for the owner.

### 2.2 Demo ladder (all three tiers must be selectable by a flag; same code path)
| Tier | Mode flag | What it is |
|---|---|---|
| 1 | `--mode keyboard` | Scan phase + state machine on the live webcam; verbs come from key presses or UI buttons. No trained model needed |
| 2 | `--mode clip --source <video>` | Real trained verb model on a recorded clip, replayed through the same pipeline |
| 3 | `--mode live` | Everything live on the head-mounted webcam |

---

## 3. Environment and current state

### Hardware and runtime
- **Runtime:** WSL2 (Linux) on a Windows host. The ML code lives in WSL2.
- **Home PC:** Ryzen 7 7700X, **RTX 4060 8 GB**. Training and the live demo must both fit it. The model that ships in the demo must run on it at a few windows per second (working target: at least 3 verb predictions per second including preprocessing; measure, do not assume).
- **College supercomputer** (8x H100; the "100 GB" is VRAM, not disk) may be available for about 2 weeks from Oct 14, unconfirmed beyond. **Do not depend on it.** Make every training script device-agnostic and config-driven so it can move.
- **Webcam:** attached to WSL2 via usbipd-win USB passthrough. Validated baseline on the old integrated camera: **640x480 MJPEG, requested 30 FPS, measured about 15 to 18 FPS**; 720p negotiates but times out under WSL2; YUYV streaming hangs (the app must explicitly request MJPEG). A **new external webcam** has been bought and **has not been tested**; re-run `python -m scripts.camera_preview` against it first; expect a config change only. `/dev/video0` is the video stream, `/dev/video1` is metadata only.
- **Existing code** (`ego-vision/`, find it with `find ~ -maxdepth 3 -name camera_preview.py` if not in the working directory): `src/` has `capture.py` (`CameraCapture(source, width, height, fps, pixel_format)`), `frame_buffer.py` (thread-safe `LatestFrameBuffer`, keeps newest frame, counts drops), `camera_worker.py` (`CameraWorker`, acquisition separated from processing); `scripts/` has `camera_test.py`, `camera_preview.py`, `test_frame_buffer.py`. `FramePacket(frame_id, timestamp_ns, image)`. Capture and display work; reuse them.
- **Python/ML stack:** use what is installed; pin everything in `requirements.txt`. Expected: PyTorch + torchvision, OpenCV (currently 5.0.0), numpy, pandas, ffmpeg, pytest, FastAPI + uvicorn for the UI backend. Verify `nvidia-smi` and `torch.cuda.is_available()` in WP0.

### What is NOT done (the honest state)
- **EPIC-KITCHENS-100 data access has never been attempted.** It is the largest unknown and it gates all model work. WP0 handles it first.
- No model code, no recipe state machine, no verb classes finalized against real counts, no fundamentals exercises done.
- **Do not block Tier 1, the reference library or the UI on the data.** If the data fails, keep building those and report it.

---

## 4. Repository layout and ownership

```
ego-vision/
  src/
    capture.py frame_buffer.py camera_worker.py     # existing, do not change behavior
    data/            # annotation loading, subset selection, Dataset class, video decoding
    models/          # verb model variants, backbone factory, embedding extractor
    train/           # training loop, eval, configs loader
    pipeline/        # sliding window, smoothing, adapters, event bus, session logging
    app/             # FastAPI backend + web UI + OpenCV fallback UI
    registry.py      # chooses impl per module: "reference" or "core" (section 5.3)
  core/              # OWNER-ONLY. Agents never write here. Starts empty
  reference/         # YOU write: working reference implementations (study material)
    exercises/       # knn.py softmax.py batchnorm.py (warm-ups)
    metrics_ref.py frame_sampler_ref.py enrollment_ref.py
    temporal_attention_ref.py state_machine_ref.py
  exercises/stubs/   # signature + docstring only, no bodies; owner copies to core/ and fills in
  docs/
    ARCHITECTURE.md  worked/<topic>.md  GLOSSARY.md  STUDY_QUEUE.md
  recipes/           # YAML recipes (section 12)
  configs/           # YAML experiment configs
  scripts/           # CLIs: download, select videos, preprocess, train, eval, record, run app
  tests/             # unit tests, equivalence tests, smoke tests
  runs/              # outputs of every logged run (metrics.json, config, stdout)
  reports/           # generated tables (verb counts, comparisons)
  STATUS.md  BLOCKERS.md  DECISIONS_NEEDED.md  HANDOFF.md
```

### The five core pieces (owner writes these himself later, in `core/`)
1. `metrics.py`: top-k, per-class and mean-class accuracy, confusion matrix, majority baseline.
2. `frame_sampler.py`: (start_frame, stop_frame, T, fps) to frame indices.
3. `enrollment.py`: scan-phase prototype matcher.
4. `temporal_attention.py`: attention over per-frame features.
5. `state_machine.py`: recipe-order logic.

You write a **reference** version of each in `reference/`, plus a **stub** in `exercises/stubs/` (signature + docstring, no body), plus a **worked example** in `docs/worked/` (section 6, WP2).

---

## 5. Interface contracts

Plain types, no frameworks. Keep these signatures stable; the owner's `core/` files must be drop-in replacements.

### 5.1 Core-piece interfaces
```python
# metrics
topk_accuracy(scores: np.ndarray[N, C], y: np.ndarray[N], k: int) -> float
per_class_accuracy(pred: np.ndarray[N], y: np.ndarray[N], num_classes: int) -> np.ndarray[C]
mean_class_accuracy(pred, y, num_classes) -> float          # ignores classes absent from y; document that
confusion_matrix(pred, y, num_classes) -> np.ndarray[C, C]  # rows = true class, cols = predicted
majority_baseline(y: np.ndarray[N]) -> float

# frame sampler (the stop_frame convention must be a documented parameter; list it in DECISIONS_NEEDED)
sample_indices(start_frame: int, stop_frame: int, T: int, fps: float) -> np.ndarray[T] (int64)
#   - all indices within the clip; length exactly T; sorted; works when T > clip length (document how)

# scan-phase matcher
class IngredientMatcher:
    def __init__(self, embed_fn, threshold: float): ...      # embed_fn: list[np.ndarray HxWx3] -> np.ndarray[N, D], L2-normalised
    def enroll(self, name: str, frames: list[np.ndarray]) -> None   # prototype = mean embedding, re-normalised
    def match(self, image: np.ndarray) -> tuple[str, float]         # cosine similarity; ("unknown", score) below threshold
    def save(self, path) / load(path)

# temporal attention
class TemporalAttention(nn.Module):
    forward(x: Tensor[B, T, D]) -> tuple[Tensor[B, D], Tensor[B, T]]   # pooled features, attention weights summing to 1 over T

# recipe state machine
class RecipeStateMachine:
    def __init__(self, recipe: Recipe): ...
    def update(self, verb: str, ingredient: str, t_ns: int) -> Event
# Event(kind: "advance"|"wrong_order"|"repeat"|"idle"|"done", step_index: int, next_step: Step|None, message: str)
```

### 5.2 Model and pipeline interfaces
```python
VerbModel.predict(window: list[FramePacket]) -> np.ndarray[num_verbs]   # softmax probabilities
Embedder.__call__(images: list[np.ndarray]) -> np.ndarray[N, D]         # L2-normalised
Smoother.update(probs: np.ndarray[num_verbs], t_ns: int) -> tuple[str | None, float]   # (verb or None, confidence)
```
Training-time tensors: clip batch is `(B, T, C, H, W)`; backbones see `(B*T, C, H, W)`; per-frame features `(B, T, D)`.

### 5.3 Swapping implementations
`src/registry.py` reads `configs/impl.yaml`:
```yaml
metrics: reference         # or core
frame_sampler: reference
enrollment: reference
temporal_attention: reference
state_machine: reference
```
`pytest tests/equivalence --impl core` must run the **same shared test vectors** against whichever implementation is selected, so the owner can check his own version against yours. The app uses the registry and runs end to end with every module on `reference` until he switches them.

---

## 6. Work packages, tracks and priorities

**Tracks run in parallel.** Priority order if time runs short: **WP0, WP5 (Tier 1), WP2, WP1, WP3, WP4, WP6, WP7**.

### Track A: data and model
**WP0 Gate: environment and data access (do this first, about 2 hours).**
- Verify GPU, Python env, ffmpeg, `df -h ~` (write free GB to `STATUS.md`).
- Clone the annotation and download-script repos (section 8) and download **one** video (`P01_01`). Watch speed for 10 minutes. If the connection resets or crawls, switch to a fallback route (section 8) and record what happened.
- Load `EPIC_100_train.csv` and `EPIC_100_validation.csv`; confirm columns; check `P01_01` fps with `ffprobe`.
- **Acceptance:** one video on disk; an `reports/data_access.md` with commands run, speeds, sizes, fps, free disk. If data cannot be obtained within the window, say so loudly in `BLOCKERS.md` and continue other tracks.

**WP1 Data pipeline.**
- `reports/verb_counts.md`: instances per verb in train and validation, the owner's 8 candidate verbs highlighted (take, put, wash, cut, open, close, mix, pour), and a flag for any candidate with fewer than 100 training instances. Do **not** change the verb list yourself; list proposals in `DECISIONS_NEEDED.md`.
- `scripts/select_videos.py --budget-gb N`: greedily pick videos rich in the configured verbs, estimating about 0.66 GB per video (derived from one tutorial's 464 GB figure over 700 videos; an estimate, replace with measured sizes after the first downloads).
- Downscale script: ffmpeg to short side about 256 **keeping fps and frame count unchanged** so annotation frame indices remain valid. Verify frame counts before and after on one video.
- `Dataset` class: reads the official `EPIC_100_train` / `EPIC_100_validation` CSVs restricted to downloaded videos, uses the sampler interface, returns `(T, C, H, W)` plus verb label; DataLoader batches `(B, T, C, H, W)`. Handle missing videos gracefully.
- **Benchmark and report (evidence for D8, D5):** decode-on-the-fly from downscaled mp4 vs pre-extracting fixed frames per segment; clips per second; disk cost. Record numbers in `reports/loader_benchmark.md`.
- **Acceptance:** tests for shapes and index correctness on real clips; the 5-clip annotation check (cut 5 annotated segments with ffmpeg and save them to `reports/annotation_check/` with their labels in a CSV so the owner can watch and compare).

**WP3 Models, training, evaluation.** Everything config-driven (`configs/*.yaml`); no decisions baked in.
- Backbone factory: ResNet-18, ResNet-34 (ImageNet-pretrained), MobileNetV2.
- Variants: `single_frame` (average of per-frame logits or middle frame; document which), `tsn_avg` (average per-frame features then head), `temporal_attn` (uses the `temporal_attention` module from the registry). Optional later: TSM-style shift variant, clearly marked optional.
- Training: loop with AdamW/SGD options, cosine/step LR, AMP, class-weight option, seeds, gradient clipping option, checkpointing (best by validation mean-class accuracy), CSV logging. Include `--overfit-batch 32` mode that must drive training accuracy to about 100%.
- D3 option: optional background class sampled from gaps between annotated segments, with its own tests.
- Evaluation script: uses `metrics` from the registry; writes `runs/<id>/metrics.json` containing top-1, top-5 where applicable, per-class table, mean-class accuracy, **majority-class baseline next to every number**, confusion matrix PNG.
- Measure and report **latency on the RTX 4060** (batch size 1, fp16 and fp32) for each backbone and each T in {4, 8, 16}; `reports/latency.md`.
- **Do not train bigger runs than the data justifies, and do not claim improvement without the table.**
- **Acceptance:** overfit-32 test passes; a baseline run on the downloaded subset completes with logged metrics; latency table exists. If no data exists, deliver the code plus tests on synthetic data and mark runs `NOT RUN`.

**WP4 Scan phase and matcher evaluation.**
- Embedder wrapper with switchable models: DINOv2 ViT-S/14 via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` (384-dim, default), CLIP, and a ResNet-feature baseline.
- Matcher using the registry's `enrollment` (reference by default).
- Ingredient-in-hand options (D4), as config: (a) whole-frame embedding, (b) crop around the hand using MediaPipe Hand Landmarker, (c) fixed "show zone" crop in the center of the frame.
- `scripts/record_ingredient_frames.py`: records labelled frames from the webcam into `data/ingredients/<name>/*.jpg` for evaluation only.
- `scripts/eval_matcher.py`: held-out top-1 per option and embedder, plus a threshold sweep (false-accept vs false-reject) written to `reports/matcher_eval.md`.
- **Acceptance:** works end to end on synthetic or webcam frames; evaluation script runs; results only reported if actually run.

### Track B: reference library and docs
**WP2 Reference implementations and worked examples (high value; do this early).**
For each of: `knn` (L2 distances with two loops, one loop and no loops, plus `predict_labels`, with an `np.allclose` equivalence test between versions), `softmax` (loss and gradient, naive then vectorized, plus a numerical gradient check; subtract-max explained), `batchnorm` (forward in train and eval modes with running statistics, compared against `nn.BatchNorm2d`), `metrics`, `frame_sampler`, `enrollment`, `temporal_attention`, `state_machine`:
1. `reference/<name>.py` with shape comments on every step.
2. `docs/worked/<name>.md`: a **tiny hand-traceable example** (numbers small enough to compute on paper: e.g. 2 test points, 3 train points, 2 features for kNN; 2 samples by 3 classes for softmax; frames 100 to 180 with T=8 for the sampler) with every intermediate value and shape tabulated, a "what each row and column means" paragraph, three common mistakes, and the edge cases.
3. `exercises/stubs/<name>.py`: signature + docstring only.
4. Shared test vectors in `tests/equivalence/` so his own version can be checked against yours.
- Notable edge cases to cover: sampler with T larger than clip length and boundary frames; the `stop_frame` inclusive vs exclusive convention (make it an explicit documented parameter; list as open in `DECISIONS_NEEDED.md`); BatchNorm at batch size 1 in train vs eval mode.
- **Acceptance:** all reference tests pass; equivalence harness runs with `--impl reference`; each worked example's hand numbers match the code's output (the verifier sub-agent checks this).

**WP6 Explanations.**
- `docs/ARCHITECTURE.md`: data flow diagram (text/ASCII is fine), with shapes at every arrow, and a table mapping each module to the concept it uses.
- `docs/GLOSSARY.md`: every term used in the code, one or two plain sentences, no jargon chains.
- `docs/STUDY_QUEUE.md`: the owner's study order, one line each, linking the reference file, the worked example, and the resource from section 10. Order: kNN, softmax, metrics, BatchNorm, frame sampling, training loop (read your train code), transfer learning, embeddings/prototypes, temporal aggregation, attention.

### Track C: app and UI
**WP5 Application and UI (Tier 1 first).**
- **Pipeline** (`src/pipeline/`): `CameraWorker` frames feed an inference loop at a fixed rate; a sliding window of the last ~2 s is sampled to T frames by the sampler registry; smoothing options (D6): majority vote over last k windows, EMA of probabilities, hysteresis; confidence-threshold option (D3a); the state machine consumes (verb, ingredient) events.
- **Event bus:** every state change emits a typed JSON event (`scan_progress`, `ingredient_matched`, `verb_predicted`, `step_advanced`, `wrong_order`, `idle`, `done`, `metrics_tick` with FPS, dropped frames, latency). The UI only consumes events; no logic in the front end.
- **Session logging:** write each session to `runs/sessions/<timestamp>.jsonl`; optional recording of the overlaid video to mp4 (for the owner's reels).
- **Modes:** `--mode keyboard|clip|live` (section 2.2). `--ui web|cv` (OpenCV-window fallback with a plain overlay, for latency and for when the browser path breaks).
- **Backend:** FastAPI + uvicorn: an MJPEG stream endpoint for video, a WebSocket for events, small REST endpoints for start/stop/recipe/scan. Serve the front end as static files. **No Node build step.**
- **Acceptance (Tier 1):** `python -m scripts.run_app --mode keyboard` opens the UI, completes a scan phase on the webcam, steps through the example recipe with keyboard verbs, shows a wrong-order case, and logs a session. A smoke test script automates it with synthetic frames.
- **Acceptance (Tier 2/3):** same flags run with a trained checkpoint (if one exists) on a recorded clip and live; if no checkpoint exists, the flags fail with a clear message, not a crash.

**WP7 Tests and packaging.**
- `pytest` for everything; `make test`, `make smoke`, `make app`; `requirements.txt` pinned; `README.md` with install and run in under 10 lines; CI-style script `scripts/check_all.sh`.

### Suggested timeline (adjust, but keep the gates)
| When | Goal |
|---|---|
| Day 1, hours 0 to 2 | WP0 gate. Skeleton of repo, registry, event bus, interfaces |
| Day 1, hours 2 to 10 | WP2 references and worked examples (parallel); WP5 Tier 1 app and UI (parallel); WP1 as data allows |
| End of day 1 | **Vertical slice:** Tier 1 runs on the webcam with all-reference modules; STATUS.md current; commit and tag `day1` |
| Day 2 | WP3 models/training/eval, WP4 matcher evaluation, Tier 2 wiring, WP6 docs, UI polish |
| Day 3 (buffer) | Verification pass, fix failures, `HANDOFF.md`, tag `handoff` |

---

## 7. UI specification

**Goal:** stunning and legible when screen-recorded for Instagram, and genuinely useful for debugging. Use your frontend-design skills if you have them. Direction below; stay inside it unless there is a clear reason.

- **Look:** dark by default, warm kitchen-appropriate accent (one accent color, used sparingly), generous spacing, one font family (system stack, no external CDNs so it works offline), large readable type (the 1080p screen recording must be legible on a phone), subtle motion only (no flashing; 150 to 250 ms transitions).
- **Main view:** large live video with a clean overlay: verb chip with confidence, ingredient chip, and a small state indicator. Right side: **recipe timeline** with step states (done, current, upcoming), the current step enlarged, a one-line hint ("Next: cut cucumber"). A wrong-order event shows a calm, clear warning for about 3 seconds, with the expected step.
- **Scan phase view:** a step wizard. One ingredient at a time, a countdown ring for 10 seconds, thumbnails of captured frames appearing as they are taken, a checkmark when the prototype is stored, a progress bar for the whole scan. A "retake" button per ingredient.
- **Debug drawer** (toggle with `d`): verb probability bars, ingredient similarity scores, FPS, dropped frames, inference latency, the sampled-frames strip showing exactly which T frames the model saw, current smoothing and threshold settings. Config switches for D3, D4, D6 as live controls where cheap.
- **Presentation mode** (toggle with `p`): hides debug, enlarges type, optimized for recording.
- **Keyboard shortcuts:** documented in an on-screen help overlay (`?`). In keyboard mode, number keys emit verbs.
- **Responsive and robust:** works from 1280x720 up; survives camera loss, model-missing and WebSocket reconnects with visible, friendly states (never a blank page).
- **Accessibility basics:** sufficient contrast, focus states, no information conveyed by color alone.

---

## 8. Technical facts (verified earlier by the owner's assistant; re-check anything that fails)

### EPIC-KITCHENS-100
- About 100 h of first-person, head-mounted kitchen video; 45 kitchens; about 90K action segments in 700 videos; 97 verb classes and about 300 noun classes; long-tailed class distribution. Native video is 1920x1080, usually 50 fps; **check each video's fps** (annotation v1.1.0 includes per-video fps/duration/resolution metadata).
- Annotations repo: `https://github.com/epic-kitchens/epic-kitchens-100-annotations`. Files: `EPIC_100_train.csv` (15 columns including start/stop info and verb/noun and their classes), `EPIC_100_validation.csv`, `EPIC_100_test_timestamps.csv` (no labels), `EPIC_100_verb_classes.csv`, `EPIC_100_noun_classes.csv`; pickles also exist.
- Download scripts: `https://github.com/epic-kitchens/epic-kitchens-download-scripts` (Python 3.5+, no external libraries). Flags: `--videos`, `--rgb-frames`, `--specific-videos P01_01,P02_122`, `--participants`, `--output-path`. Data lands in an `EPIC-KITCHENS` subfolder, one folder per participant; already-downloaded files are skipped, so batches can run across sessions. `epic_video_downloader_verify.py` adds resume and MD5 verification with a CSV report.
- First test command:
```
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
cd epic-kitchens-download-scripts
python epic_downloader.py --videos --specific-videos P01_01 --output-path ~/data
```
- **Known problem:** the README warns the script can fail with "connection reset by peer" depending on region and can be very slow; the owner is in **India**, so treat this as likely. Fallbacks it recommends: **Academic Torrents** (links in the README), plus a Hugging Face mirror (`awsaf49/epic_kitchens_100`; one tutorial quotes 464 GB for all videos, so download only a subset) and a pre-cut clips dataset (`lightly-ai/epic-kitchens-100-clips`, about 24 GB; check its coverage before relying on it).
- Pre-extracted RGB frames come as one `.tar` per video at full frame rate; downscaled videos are likely smaller for his purposes.
- Split policy: use the official `EPIC_100_train` / `EPIC_100_validation`, restricted to downloaded videos. Test labels are hidden, so validation acts as the test set.
- Many-shot check: the owner's earlier rule is a class counts as many-shot when it appears more than 100 times in training. Only verbs are trained now, so only verb counts need to clear it; compute them yourself.
- A reference for the "no clip boundaries in live video" problem: `https://github.com/iranroman/ego_actrecog_analysis` runs fixed-size sliding windows over full EPIC videos.

### Gotchas to design around
- **BatchNorm at batch size 1:** the live app runs at batch size 1. `model.eval()` must be set at inference, or batch statistics from a batch of 1 break predictions. Add a test that fails if inference code leaves the model in train mode.
- **Frame indices vs timestamps:** annotations give start/stop *frame* numbers; do all arithmetic in frames at the video's true fps.
- **No background class and pre-trimmed clips in training, but a continuous live stream:** the model will fire on idle footage. This is why D3 and D6 exist; measure it rather than assume.
- **Domain gap:** EPIC is other people's kitchens, the owner's webcam and counter differ. Provide the evaluation-only webcam check tooling (below).
- **DINOv2 license note:** one source lists the repo weights as non-commercial (CC-BY-NC), another lists the Hugging Face `dinov2-small` as Apache 2.0. Fine for a personal demo; note the license in the README.

### Evaluation-only webcam check (provide the tooling)
- `scripts/record_verb_clips.py`: record short labelled clips (about 5 per verb) from the webcam into `data/webcam_eval/<verb>/`; **never used for training**.
- `scripts/eval_on_webcam_clips.py`: run a chosen checkpoint over them with the owner's metrics module and report the domain gap.

---

## 9. Design decisions that belong to the owner (implement as config; do not decide)

| ID | Decision | Options to implement | Evidence you produce | Default until he decides |
|---|---|---|---|---|
| D1 | Verb-model backbone | ResNet-18, ResNet-34, MobileNetV2 | Accuracy vs latency table on the subset (`reports/`) | ResNet-34 |
| D2 | Train big then deploy small, or train small | Config sizes; no distillation unless he asks | Whether H100 time is available; training cost | Train small |
| D3 | Handling "nothing is happening" | (a) confidence threshold, (b) background class from gaps between segments, (c) motion gating | Replay a recorded idle clip and report how often the model fires, per option | (a) threshold 0.5 |
| D4 | Where ingredient identity comes from | (a) whole frame, (b) MediaPipe hand crop, (c) fixed show zone | Quick matcher test per option on real webcam frames | (c) show zone |
| D5 | Clip length T and stride | T in {4, 8, 16}, window about 2 s | Accuracy vs inference time | T=8 |
| D6 | Smoothing of live predictions | Majority vote over last k, EMA, hysteresis | Replay recorded clips with each policy | EMA |
| D7 | Scan-phase embedding model | DINOv2-Small, CLIP, ResNet features | Held-out top-1 on 4+ ingredients | DINOv2-Small |
| D8 | Training resolution and preprocessing | Short side about 256 offline vs on-the-fly 1080p decode; pre-extracted frames | Loader throughput and disk cost | Offline 256 |

Write each decision's evidence into `DECISIONS_NEEDED.md` as: *question, options, numbers (with the commands that produced them), what you would pick and why (clearly labelled as a suggestion), what would change the answer*. He decides.

---

## 10. Resources to link in the docs (opened and confirmed earlier)
- CS231n Assignment 1 (kNN, softmax, two-layer net): https://cs231n.github.io/assignments2025/assignment1/
- d2l.ai batch normalization (ResNet is the next section): https://d2l.ai/chapter_convolutional-modern/batch-norm.html
- ESIEE lab "Build a Residual CNN" (manual BatchNorm vs PyTorch): https://perso.esiee.fr/~chierchg/deep-learning/concepts/residual-cnn-student.html
- PyTorch Transfer Learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- PyTorch "Learning PyTorch with Examples": https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html
- TSM paper: https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html and code https://github.com/mit-han-lab/temporal-shift-module
- DINOv2: https://github.com/facebookresearch/dinov2 ; embeddings + simple classifier: https://blog.roboflow.com/how-to-classify-images-with-dinov2/
- MediaPipe Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- EPIC-KITCHENS-100 paper: https://arxiv.org/abs/2006.13256

---

## 11. Out of scope (binding)
- Action **anticipation**, temporal **detection/localization**, any research direction or novelty work (the owner is handling "when should the system intervene" himself, later).
- Any **learned recipe-step network** (step order is the state machine with a hard-coded recipe).
- **Own-kitchen dataset collection for training or fine-tuning** (webcam clips are evaluation only).
- **Self-pretraining on Ego4D** (optional frozen public features are not in this window).
- Any **LLM in the perception loop** (verbs and ingredients come from the models above; text phrasing in the UI is templated).
- Network calls from the app at runtime other than loading model weights.
- Anything under `core/`.

---

## 12. Recipe config (placeholder; the owner will replace it)

`recipes/example_salad.yaml`: use this as the default and make the recipe format general. **This recipe is a placeholder; do not hard-code it anywhere.** The owner's final recipe will use 3 to 4 ingredients and only verbs from the trained subset.
```yaml
name: Example salad
ingredients: [tomato, cucumber, dressing]
verbs: [take, wash, cut, put, pour, mix]
steps:
  - {verb: take,  ingredient: tomato}
  - {verb: wash,  ingredient: tomato}
  - {verb: cut,   ingredient: tomato}
  - {verb: put,   ingredient: tomato,   note: "into bowl"}
  - {verb: take,  ingredient: cucumber}
  - {verb: cut,   ingredient: cucumber}
  - {verb: put,   ingredient: cucumber, note: "into bowl"}
  - {verb: pour,  ingredient: dressing}
  - {verb: mix,   ingredient: dressing}
```
Validation: loading a recipe must fail clearly if a step uses a verb not in `verbs` or an ingredient not in `ingredients`.

---

## 13. Handoff (`HANDOFF.md`, due at the end of the window)
Must contain:
1. **What runs today** (per tier, with the exact command) and **what does not**.
2. **Quick start in at most 10 lines.**
3. **File map**: one line per folder and key file.
4. **Numbers**: every result with its command, date, commit and seed, or `NOT RUN`.
5. **Open decisions** (link to `DECISIONS_NEEDED.md`) and **blockers** (link to `BLOCKERS.md`).
6. **Known issues and risks**, including anything flaky.
7. **Study queue** (link to `docs/STUDY_QUEUE.md`).
8. **Verification log**: what the verifier sub-agent re-ran, and the result.

**Definition of done for the window:** Tier 1 runs on the webcam with the full UI; the reference library, worked examples, stubs and equivalence tests exist and pass; the data pipeline, training and evaluation code exist with honest status on what was actually run; `HANDOFF.md` is complete. Anything not achieved is reported as such, not papered over.
