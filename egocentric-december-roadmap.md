# Egocentric Vision: December Demo Roadmap

*Written Oct 14, 2026 (Wed). Self-contained: a new chat, model, or agent harness should be able to pick up from this file alone. It supersedes the Dec 8 technical-report plan in `egocentric-vision-master.md` (keep that file for deep background: Ego4D/EPIC-KITCHENS notes, webcam setup history, the original plan text).*

---

## 0. Read this first (for any model or agent picking this up)

**Who:** Harshil, 20, B.Tech AI & ML (Sem 5), Delhi. Dictates through Wispr Flow, so transcription errors happen (e.g. "Wiser" meant VISOR); interpret sensibly, confirm if unsure. Long-term goal: MEXT scholarship for a CV/AI master's in Japan; this project feeds that (Sato Lab-aligned egocentric vision).

**How to work with him:**
- He is in the driving seat. He makes the design and architecture decisions; you present options, evidence and trade-offs, then he picks. Do not decide for him.
- He wants to understand every function. Be direct and factual. He has said he does not need reassurance.
- His learning method is **worked example first, then write your own**. He understands concepts well and recalls code he has seen, but stalls writing from a blank file when he has never seen the working row/column/shape version (e.g. kNN distance matrix). So: show him a small hand-traceable example and the reference code, let him study it, then he closes it and writes his own version. Never ask him to invent an unseen implementation cold.
- Agents (his OpenCode setup on WSL: a GLM 5.3 Flash orchestrator plus about 34 sub-agents on mostly free models) may do boilerplate, scaffolding and tests that need no attention. They may not touch the hand-written core files (see section 9).

**Status as of Oct 14, 2026:**
- Done: webcam capture pipeline (section 11), reading of the Ego4D, EPIC-KITCHENS-100 and VISOR papers, many design decisions (section 2).
- Not done: EPIC-KITCHENS-100 data access has never been tested; no model code; recipe state machine not written; no verb classes finalized against the data; no fundamentals rebuilds started. Checkpoint A (Oct 5) of the old plan was missed.
- He has bought the new webcam and plans to head-mount it (cap or helmet).
- The lock-in began today: 2 to 3 hours/day for this project, which is his number one priority track, ahead of Japanese and the RSNA Kaggle competition.

---

## 1. Goal and finish lines

**Why this has felt aimless:** CS231n had a countable finish line (18 lectures). This project had weeks, not "done" states. Every stage below now has an exit test: something you can run plus something you can rebuild.

**The project:** a head-mounted webcam watches Harshil cook a fixed recipe. Before cooking, a 10-second **scan phase** (he holds each ingredient up) stores what each ingredient looks like. During cooking, a **trained verb model** recognizes the action (take, put, wash, cut, open, close, mix, pour) from the last ~2 seconds of video, the **scan-phase matcher** identifies which ingredient is in hand, and a **hand-written recipe state machine** compares each (verb, ingredient) event with the hard-coded recipe and shows the next expected step or flags a wrong order.

**Finish lines (demo ladder, so something is always postable):**

| Tier | What it is | Target date |
|---|---|---|
| 1 | Scan phase + state machine on the live webcam, verbs simulated from the keyboard. No trained model needed | Sun Nov 8 |
| 2 | Same, with the trained verb model running on a recorded clip | Sun Nov 22 |
| 3 | Everything live on the head-mounted webcam, one clean end-to-end recipe run recorded | Sun Nov 22 (fallback: stays Tier 2) |

**Done means (December exit test):**
1. A script runs on his webcam, completes the scan phase, and prints/overlays verb + ingredient + next step in real time using weights he trained.
2. He can cold-rebuild the four hand-written core pieces (metrics, frame sampler, scan-phase matcher, temporal attention) in about 25 minutes each without help.
3. He can explain and defend every design decision in section 4 (the D1 to D8 list) with the evidence he gathered.

**Out of scope for December:** anticipation (Stage 2), detection (Stage 3), any learned recipe-step network, own-kitchen data collection, self-pretraining on Ego4D. Anticipation belongs to the Feb 2027 preprint track.

---

## 2. Decisions already made (do not reopen without a reason)

| # | Decision | Date |
|---|---|---|
| 1 | Runtime is WSL2 (the ML stack lives there). Webcam via usbipd-win USB passthrough | Sep 24 |
| 2 | EPIC-KITCHENS-100 is the training dataset. Ego4D is background reading only; VISOR not used for now | Sep 28 |
| 3 | No self-pretraining on Ego4D (compute-class mismatch). Frozen public Ego4D-pretrained features are an optional stretch only if there is slack | Sep 28 |
| 4 | December project = recognition + hard-coded recipe logic. **Anticipation is not part of it** (too shaky to post) | Oct 14 |
| 5 | **No own-kitchen dataset / fine-tuning.** Ingredient identity comes from the scan phase instead of a trained noun classifier | Oct 14 |
| 6 | **No second neural network for recipe steps.** Step order is the state machine with a hard-coded recipe | Oct 14 |
| 7 | Verb model trained on EPIC-KITCHENS-100 only, on a small many-shot verb subset (candidates: take, put, wash, cut, open, close, mix, pour; verify against the data) | Oct 14 |
| 8 | Camera is head-mounted (closest to the EPIC-KITCHENS viewpoint) | Oct 14 |
| 9 | Fundamentals method: study a worked example, then write your own. Four core pieces hand-written (section 9) | Oct 14 |
| 10 | Demo is built as a ladder (section 1) | Oct 14 |
| 11 | The demo is the finale of his 100-day reel series (section 10) | Oct 14 |
| 12 | The H100 access is a bonus, not a plan dependency | Oct 14 |

---

## 3. Calendar and hours

All dates 2026. Honest hours: 2 to 3 h/day when free; the minimum viable day is 60 minutes on that day's single task.

| Week | Dates | Fixed events | Realistic project hours |
|---|---|---|---|
| W1 | Thu Oct 15 – Sun Oct 18 | Lock-in; college off through Tue Oct 20 | 10–12 |
| W2 | Mon Oct 19 – Sun Oct 25 | **RSNA Kaggle heavy week (Oct 19–25)** | 8–10 (minimum-viable days) |
| W3 | Mon Oct 26 – Sun Nov 1 | College H100 access ends about **Oct 28** (maybe longer, uncertain); **Hack CBS Oct 31 – Nov 1** | 9–12 (Fri–Sun off) |
| W4 | Mon Nov 2 – Sun Nov 8 | | 14–18 |
| W5 | Mon Nov 9 – Sun Nov 15 | | 14–18 |
| W6 | Mon Nov 16 – Sun Nov 22 | | 14–18 |
| W7 | Mon Nov 23 – Sun Nov 29 | Internals + midterm at end of Nov (exact dates TBC) | 5–8, mostly buffer. **Code freeze Sun Nov 29** |
| Dec 1–5 | Mon Nov 30 – Sat Dec 5 | | Reel editing only, ≤60 min/day. No new engineering |
| Dec 6–7 | Sun Dec 6 / Mon Dec 7 | **JLPT N5 Dec 6**, recovery Dec 7 | Zero |
| Dec 8–14 | Tue Dec 8 – Mon Dec 14 | **End-sem prep week** (CGPA target this semester: at least 9.1) | Zero |
| Dec 15+ | from Tue Dec 15 | End-sem exams | Zero |

**Important consequence:** "mid-December" in practice means **code frozen Nov 29, reel edited by Dec 5, posted later**. With the series at about Day 36 today and one reel/day from here, **Day 100 lands about Dec 17 (Dec 18 if he is on Day 35)**, which is inside end-sems. So the finale reel must be pre-edited and queued, or the series pace must change. Decision for Harshil: see open question Q1.

---

## 4. System design

### 4.1 Pipeline

```
head-mounted webcam (640x480 MJPEG, ~15-18 FPS under WSL2)   [built: CameraCapture / LatestFrameBuffer / CameraWorker]
        |
        +--> SCAN PHASE (before cooking): per ingredient, ~10 s of frames
        |       -> embedding model -> average embeddings -> one prototype vector per ingredient
        |
        +--> COOKING PHASE
              +--> VerbModel: last ~2 s window, T sampled frames -> verb probabilities (8 classes + maybe background)
              +--> IngredientMatcher: frame (or hand crop) -> embedding -> cosine similarity to prototypes -> ingredient
              +--> RecipeStateMachine: (verb, ingredient, time) -> expected next step / wrong-order flag / suggestion
              +--> overlay on the live display
```

Interface sketch (a proposal, his to change). Plain types, no framework:

```
Enroll:   matcher.enroll(name: str, frames: list[image])
Match:    matcher.match(image) -> (name: str, score: float)
Verb:     verb_model.predict(window: list[FramePacket]) -> probs: array[num_verbs]
Logic:    state_machine.update(verb: str, ingredient: str, t_ns: int) -> Event(kind, next_step, message)
```

### 4.2 Open design decisions (his to make; gather the evidence listed)

| ID | Decision | Options | Evidence to gather | Decide by |
|---|---|---|---|---|
| D1 | Verb-model backbone | ResNet-34 (ImageNet-pretrained) vs ResNet-18 vs MobileNetV2. It must run at a few Hz on the RTX 4060 | Accuracy vs latency table on the subset (latency in W2, accuracy in W3) | Oct 29 |
| D2 | Train big, deploy small, or just train small | With the H100 window: train bigger, distill/shrink. Otherwise: train small | Whether the H100 is still available when code is ready | Oct 28 |
| D3 | "Nothing is happening" handling. EPIC-KITCHENS has **no background class** and trains on pre-trimmed clips, but the live stream has no boundaries | (a) confidence threshold, (b) add a background class sampled from the gaps between annotated segments, (c) motion gating | Run the model over a recorded clip of idle time and see how often it fires | Nov 15 |
| D4 | Where ingredient identity comes from during cooking | (a) whole-frame embedding, (b) crop around the hand (MediaPipe Hand Landmarker), (c) fixed "show zone" in front of the camera | Quick test on 5 enrollment images per option, on the real head-mounted view | Nov 8 |
| D5 | Clip length T and sampling stride for the verb model | e.g. 8 frames over about 2 s uniformly; compare T=4/8/16 | Accuracy vs inference time | Nov 1 |
| D6 | Smoothing policy for live predictions | Majority vote over last k windows, EMA of probabilities, hysteresis | Replay recorded clips with each policy | Nov 22 |
| D7 | Embedding model for the scan phase | DINOv2-Small (ViT-S/14, 384-dim; check weights license), CLIP, ResNet features | Top-1 on held-out frames of 4+ ingredients | Nov 8 |
| D8 | Training resolution and preprocessing of EPIC videos | Downscale to short side ~256 with ffmpeg once, offline, vs decode 1080p on the fly | Data-loader throughput measurement | Oct 25 |

### 4.3 Known technical gotchas
- **BatchNorm in live inference:** the demo runs at batch size 1. Call `model.eval()`, or BN uses batch statistics from a batch of 1 and predictions break. This is covered in the topic 1 exit check.
- **Domain gap:** a model trained on other people's kitchens may behave differently on his webcam and counter. Mitigation: an evaluation-only webcam check early (Oct 29, section 7).
- **Frame indices vs timestamps:** EPIC annotations give start/stop frames; confirm each video's fps (annotation v1.1.0 added fps/duration/resolution metadata) before trusting frame arithmetic in the sampler.
- **Official split:** keep the official train/validation split (`EPIC_100_train` / `EPIC_100_validation`), restricted to the videos he downloads. The test labels are hidden, so validation acts as the test set.
- **Many-shot definition check:** the master file defines many-shot as verb and noun each appearing more than 100 times in training. Since only verbs are trained now, only the verb counts need to clear the bar, but verify the counts from `EPIC_100_train` yourself.

---

## 5. Learning method (applies to every topic)

**The loop:**
1. **Study** a working reference with a tiny hand-traceable example. Write shapes as comments first (input shape, output shape, one line per step). Take notes in his own words.
2. **Close** the reference. Next day, **write your own version** from the notes (25 minutes). If stuck after 25 minutes, open official docs only, and log every function you looked up.
3. **Test** on a tiny hand-computable array and assert shapes.
4. **Redo cold** +1 day and +3 to 4 days later (core pieces only).

**Typical day (about 2 h):** 20 min cold redo of an earlier piece, 60 min study or write, 40 min build (agents may help with plumbing).

**Verification drill (for the "can't check AI code" worry):** after he writes a function, have an agent write the same one with an undisclosed subtle bug (off-by-one in the frame window, wrong softmax axis) and find it with tests.

**Numpy/PyTorch verbs to log and drill:** argsort, bincount, argmax, fancy indexing (`scores[np.arange(N), y]`), linspace, where, broadcasting, reshape between (B, T, C, H, W) and (B·T, C, H, W).

**Unblocking rule for the L2 distance function:** the output is `(num_test, num_train)`, so `dists[i, j]` pairs test row `i` with train row `j`. Fix the output shape first.

---

## 6. Fundamentals curriculum (each topic tied to a project piece)

Legend for resources: ✅ = opened or confirmed this session (Oct 14); ◻ = from memory, not opened, check before relying on it. URLs are in section 13.

| # | Topic | Why it is in this project | Study | Write / do | Exit check | Week |
|---|---|---|---|---|---|---|
| 0 | kNN + vectorized numpy (warm-up) | Same idea as the scan-phase matcher (nearest prototype); trains row/column/shape reading | CS231n Assignment 1 `knn.ipynb` ✅ | Write L2 distance with loops, then vectorized; `predict_labels` | Vectorized output equals loop output (`allclose`); can explain `dists[i,j]` aloud | W1 |
| 2 | Softmax, cross-entropy, class imbalance | Verb head loss; EPIC is long-tailed | CS231n A1 `softmax.ipynb` ✅; CS231n linear-classifier notes ◻ | Softmax loss + gradient; then a class-weighted version | Gradient check passes; can explain why you subtract the max | W1 |
| 8 | Evaluation: top-1/top-5, per-class accuracy, validation protocol | How every result in this project is judged; guards against fooling yourself with majority-class accuracy | EPIC-KITCHENS-100 paper (metrics section) ✅ | **Core piece 1: metrics module** (top-k, per-class/mean-class accuracy, confusion matrix) | Hand-computed 5-sample example matches; reports majority-class baseline next to every number | W1 |
| 1 | Convolution, ResNets, batch norm (train vs eval) | The verb model backbone; the BN-at-batch-size-1 trap in the live demo | d2l.ai batch-norm and ResNet sections ✅; ESIEE "Build a Residual CNN" lab ✅ | BatchNorm forward (train and eval modes, running stats) compared with `nn.BatchNorm2d`; a residual block | Matches PyTorch within tolerance; can explain what breaks with `model.train()` at batch size 1 | W2 |
| 6a | Frame sampling | Data pipeline for clips; where "causal" bugs hide | TSN / TSM paper sections on sampling ✅ | **Core piece 2: frame sampler** (start_frame, stop_frame, T, fps → indices) | Tests on 3 edge cases (short clip, T > length, boundary); can explain segment-based vs dense sampling | W2 |
| 3 | Optimizers, LR schedules, mixed precision, the overfit-one-batch ritual | Training loop that fits on the 4060 (and the H100 if available) | "Learning PyTorch with Examples" ✅; Karpathy "A Recipe for Training Neural Networks" ◻ | Training loop with AdamW/SGD, cosine or step LR, AMP | Overfits 32 clips to ~100% train accuracy; can read a loss curve | W3 |
| 4 | Transfer learning (frozen vs fine-tuned) | Start from an ImageNet-pretrained ResNet instead of from scratch | PyTorch Transfer Learning tutorial ✅ | Frozen-features run vs fine-tune run on the subset | Table of both; can say when BN layers should stay frozen | W3 |
| 5 | Embeddings, cosine similarity, prototypes | The scan-phase matcher | DINOv2 repo ✅; Roboflow "How to Classify Images with DINOv2" ✅; Prototypical Networks paper ◻ | **Core piece 3: scan-phase matcher** (average frame embeddings per ingredient, cosine match, reject below threshold) | Held-out top-1 on 4+ ingredients; threshold chosen from evidence, not guessed | W4 |
| 6b | Temporal aggregation (TSN average, TSM shift) | Turns per-frame features into a clip decision | TSM paper and repo (`ops/temporal_shift.py`, the 3-line shift) ✅ | TSN-style average of per-frame features (few lines) | Single-frame vs TSN-average rows in the results table | W5 |
| 7 | Layer norm and attention | Temporal attention over the T frame features (the CBAM-style instinct from MVSS-Net Lite, applied over time) | Karpathy "Let's build GPT" video ◻; d2l.ai attention chapters ◻ | **Core piece 4: temporal attention module** | Shapes asserted; attention weights sum to 1; beats or honestly ties TSN-average, with the numbers shown either way | W5 |

*(Topic numbers match the eight-item list agreed in the Oct 14 conversation; 0 is the added warm-up and 6 is split into sampling and aggregation.)*

**Kaggle note:** the Kaggle Learn courses use pre-built Keras layers, so they don't test hand-implementation (skipped). Searches this session did not surface a Kaggle notebook I could verify as a good fit. One option that runs on Kaggle GPUs: Aakash N S's "PyTorch: Zero to GANs" Part 6 (CIFAR-10 ResNet with data augmentation) ◻, found by name only. When RSNA public notebooks are being read anyway, use the same lens: pick ones that write their own training loop and metrics.

---

## 7. Week-by-week plan

Each week: **Learn** (study), **Write** (his hands), **Build** (agents may help; he reads every line), **Exit** (done means these pass), **Reel** (suggested topic for the 100-day series).

### W1: Thu Oct 15 – Sun Oct 18 (college off). Unblock the data, rebuild the first fundamentals
*Revised Oct 15: the Oct 14 data test did not happen, so the week starts Thursday. About 10 project hours over four days; Sat after gym and Sun after the morning block are deliberately off.*

- **Thu Oct 15 (~2 h): data unblock.**
  - Run `df -h ~` in WSL and write down the free GB (answers Q2).
  - Clone both repos and start `epic_downloader.py --videos --specific-videos P01_01` in the background (commands in section 11). Watch the speed for 10 minutes. Connection reset or crawling means switch to the Academic Torrents / HF mirror route.
  - While it downloads: load `EPIC_100_train.csv` and the validation CSV in pandas, print the columns, `value_counts()` on verb, and save a count table for the 8 candidate verbs.
  - Paste the `core/` rule into the agent config (section 9).
  - Done when: one video on disk, verb-count table saved, free disk known.
  - Reel: first look at an egocentric clip next to its label.
- **Fri Oct 16 (~3 h): kNN.**
  - Study (60 min): on paper, 2 test points, 3 train points, 2 features, so a 2×3 distance matrix; label every cell `dists[i, j]`; write shape comments for each step; read the CS231n A1 `knn.ipynb` flow.
  - Close it. Write (75 min): L2 with two loops, then one loop, then no loops; then `predict_labels`; `np.allclose` between versions.
  - Annotation check (45 min): take 5 rows of P01_01, convert start/stop frames to seconds with the video's real fps (`ffprobe`), cut each clip with ffmpeg (an agent can write the command), watch them and compare with the labels. Note any that don't match.
  - Done when: vectorized equals loop; you can explain `dists[i, j]` aloud; the clips visibly line up with their labels.
  - Reel: "I couldn't write kNN from a blank file, so I changed the method".
- **Sat Oct 17 (~3 h, then off): softmax.**
  - Study (60 min): CS231n A1 `softmax.ipynb` plus the linear-classifier notes; hand example of 2 samples × 3 classes: scores, subtract max, exp, normalize, minus log of the correct class; trace the gradient with respect to scores (probabilities minus one-hot, divided by N).
  - Close it. Write (75 min): loss with a loop, then vectorized, then the gradient; numerical gradient check; class-weighted version only if time.
  - Done when: gradient check passes; you can say why the max is subtracted.
  - Reel: softmax in 30 seconds.
- **Sun Oct 18 (~2 h, morning or early afternoon, then off): metrics and spec.**
  - Hand-compute a 5-sample example on paper first, then write `core/metrics.py` (core piece 1): top-1, top-5, per-class and mean-class accuracy, confusion matrix, majority-class baseline; test against the hand answer.
  - One-page spec: pipeline sketch with D1 to D8 marked decided or open (30 min).
  - Cold redo of kNN (20 min).
  - **Exit (Checkpoint 1):** a video and its annotation rows load and line up; verb-count table exists; kNN, softmax and metrics written and tested; spec written.

### W2: Mon Oct 19 – Sun Oct 25 (RSNA heavy; Mon and Tue are bonus days because college is off)
- **Mon Oct 19 (~3 h): frame sampler (core piece 2).**
  - Study (45 min): TSN segment-based vs dense sampling; hand example: frames 100 to 180, T=8, split into 8 equal segments and take the centre of each; then the T greater than clip length case. Decide whether `stop_frame` is inclusive and write it down.
  - Close it. Write (60 min): `core/frame_sampler.py` (start_frame, stop_frame, T, fps → indices). Tests: short clip, T > length, boundary frames.
  - Build (60 min, agent plumbing, you read every line): ffmpeg script downscaling to short side ~256 (D8); time it on P01_01 and note the size change.
  - Done when: sampler tests pass and a downscaled file exists.
- **Tue Oct 20 (~2.5 h; 90 min if college is on): recipe state machine (core piece 5).**
  - Decide the recipe first (Q4): 3 to 4 ingredients, steps built only from the 8 verbs.
  - Write `core/state_machine.py` as plain control flow against a mocked (verb, ingredient) stream: correct order, wrong order, repeated event, long idle.
  - Done when: a full mocked recipe runs, including a wrong-order case.
- **Wed Oct 21 (60–90 min):** BatchNorm study (d2l batch-norm section and the ESIEE lab). Study only.
- **Thu Oct 22 (60–90 min):** write BatchNorm forward from your notes (train and eval modes, running stats); compare with `nn.BatchNorm2d`; explain what breaks with `model.train()` at batch size 1.
- **Fri Oct 23 (60–90 min):** Dataset class around the annotation CSV plus your sampler (agent scaffolds, you read every line, docstrings state shapes). A clip comes out as `(T, C, H, W)`, a DataLoader batch as `(B, T, C, H, W)`.
- **Sat Oct 24 (60–90 min):** ResNet-34 forward on a fake `(B, T, C, H, W)` batch, then a real one; reshape to `(B·T, C, H, W)` and back; time ResNet-18, ResNet-34 and MobileNetV2 forward on the RTX 4060 (the latency half of D1; accuracy comes in W3). Decide D8. If the loader is solid, an hour testing throughput on the H100 box is a bonus, not a goal.
- **Sun Oct 25 (~45 min):** cold redo of metrics and sampler (20 min each); update the progress log; Sunday goal-setting.
- **Exit (Checkpoint 2):** sampler passes its tests; state machine runs a full mocked recipe including a wrong-order case; the loader yields correctly shaped batches from real clips.
- **Reels:** Mon "How I turn a video clip into 8 frames", Wed/Thu "Why batch norm breaks at batch size 1", Sat "My first ResNet forward pass on real kitchen clips".

### W3: Mon Oct 26 – Sun Nov 1 (H100 window closing; Hack CBS Oct 31 – Nov 1, so work Mon–Thu)
- **Learn:** optimizers/LR/AMP (topic 3); transfer learning (topic 4).
- **Write:** training loop (his own, even if short); the overfit-32 test.
- **Build:** single-frame verb model (pretrained ResNet-34 + a verb head). Train on the subset, evaluate with his metrics module, report against the majority-class baseline. **Use the H100 only if the code is ready by about Oct 28; do not rush for it.** Decide D1, D2, D5.
- **Oct 29 (evaluation-only webcam check):** record about 5 clips per verb on the real head-mounted webcam (no training on them) and run the baseline. This tells you early how big the domain gap is.
- **Exit (Checkpoint 3, Oct 29):** overfit-32 reaches ~100%; baseline numbers (top-1, per-class, majority baseline) in a table; webcam check numbers noted.
- **Reel:** "First model numbers. Here's how far from chance I actually am".

### W4: Mon Nov 2 – Sun Nov 8. The scan phase + Tier 1 demo
- **Learn:** embeddings, cosine similarity, prototypes (topic 5).
- **Write:** **scan-phase matcher (core piece 3)**.
- **Build:** the enrollment UI flow (hold each ingredient up for 10 s), overlay, wiring of matcher + state machine on the live webcam with keyboard-simulated verbs. Decide D4 and D7 with the evidence in section 4.2.
- **Exit (Checkpoint 4, Nov 8):** **Tier 1 runs live on the head-mounted webcam**; a recording exists.
- **Reel:** "My camera learns my ingredients in 10 seconds" (first real demo clip).

### W5: Mon Nov 9 – Sun Nov 15. Temporal model
- **Learn:** TSN average, TSM shift (topic 6b); layer norm and attention (topic 7).
- **Write:** **temporal attention module (core piece 4)**.
- **Build:** train the three variants on the same subset and split: single-frame, TSN-average, temporal attention. TSM shift is an optional fourth row if time allows. Decide D3.
- **Exit (Checkpoint 5, Nov 15):** a three-row results table with per-class numbers; final verb model chosen with reasons.
- **Reel:** "Does attention across frames help? My results, including if it doesn't".

### W6: Mon Nov 16 – Sun Nov 22. Integration (Tier 2 and Tier 3)
- **Build:** sliding-window inference over the live frame buffer, smoothing (D6), the glue between verb model, matcher and state machine. Run on a recorded clip first (Tier 2), then live (Tier 3). Latency check on the RTX 4060.
- **Exit (Checkpoint 6, Nov 22):** Tier 3 runs live for one full recipe run, or the fallback is declared and Tier 2 becomes the finale.
- **Reel:** "The full loop running live".

### W7: Mon Nov 23 – Sun Nov 29. Buffer, film, freeze
- ≤45 min/day (internals and midterm at end of Nov). Film the final recipe run. **Cold-rebuild exam: metrics, sampler, scan-phase matcher, temporal attention, 25 min each, no help.** Code freeze Sun Nov 29.

### Nov 30 – Dec 5, then Dec 6–7 and beyond
- Reel editing only (≤60 min/day, no new engineering). Dec 6 JLPT, Dec 7 recovery, Dec 8–14 end-sem prep, Dec 15+ end-sems. No project work in those windows.

---

## 8. Checkpoints and descope triggers

The rule: every slip resolves toward shipping something honest, never toward compressing JLPT or end-sem prep.

| Checkpoint | Date | If missed |
|---|---|---|
| 1 | Oct 18 | Cut the spec to half a page; the data test is the non-negotiable part. If data access is still failing, switch to the mirror/torrent route and tell me; everything depends on it |
| 2 | Oct 25 | Drop the BatchNorm rebuild (do it later); keep sampler + state machine |
| 3 | Oct 29 | If the baseline cannot beat the majority-class baseline meaningfully, stop and diagnose (sampler, labels, learning rate) before adding any temporal model |
| 4 | Nov 8 | Tier 1 is the priority; drop the hand-crop experiment (D4) and use the fixed "show zone" |
| 5 | Nov 15 | Drop temporal attention; ship TSN-average (or single-frame). Report what was dropped |
| 6 | Nov 22 | Declare Tier 2 the finale; the live version becomes a future-work line |
| Freeze | Nov 29 | No exceptions. Cut scope, do not move the date |

---

## 9. Hand-written vs agent split

**Hand-written by Harshil, in `core/` (agents may read, never edit):**
1. `metrics.py`: top-k, per-class/mean-class accuracy, confusion matrix
2. `frame_sampler.py`: (start_frame, stop_frame, T, fps) → frame indices
3. `enrollment.py`: scan-phase prototype matcher
4. `temporal_attention.py`: attention over per-frame features
5. `state_machine.py`: recipe-order logic (plain control flow, mocked first)

**Agents may do (he reads every line before it counts as done):** CLI/config parsing, download and ffmpeg preprocessing scripts, the Dataset class around the annotation CSVs, logging, checkpointing, plotting, the enrollment UI/overlay, camera plumbing, test scaffolding, and the "subtle bug" verification drill.

**Rule text to paste into his OpenCode agent config:**
```
Never create, edit or delete files under core/. You may read them.
If a change to a core file seems needed, describe it in chat as a diff and wait.
When you write code outside core/, keep functions small and add a one-line
docstring stating input shape(s) and output shape(s).
```

---

## 10. Demo, reels and the 100-day series

- The 100-day series (Day 1 = Aug 28, 2026) has slipped; "days" are reels uploaded, not calendar days. About Day 36 on Oct 14.
- The weekly reel ideas in section 7 let the series follow the build. The finale is the Tier 3 (or Tier 2) clip.
- Recipe for the demo (from the verbs in decision 7): take tomato → wash → cut → put in bowl → take cucumber → cut → put in bowl → pour dressing → mix. Verify the recipe only uses verbs the model actually recognizes well.
- He dropped DaVinci Resolve (no time to learn editing), so budget for a simple editor. Editing happens Nov 30 – Dec 5 at most 60 min/day.

---

## 11. Technical state and facts

### Webcam pipeline (done Sep 24, WSL2)
- Camera shows as "Integrated Camera" (Microdia, VID:PID 0c45:64ab, BUSID 4-18). `/dev/video0` = video stream, `/dev/video1` = metadata only. Permissions via the `video` group (no sudo). OpenCV 5.0.0, `v4l-utils` installed.
- **Validated baseline: 640×480 MJPEG, requested 30 FPS, measured about 15–18 FPS.** 720p negotiates but times out under WSL2 USB/IP; YUYV streaming hangs. The app must explicitly request MJPEG because the camera can come back as YUYV after a reset.
- **The new external webcam has not been tested yet.** The pipeline was validated on the integrated camera. Re-run the preview script on the new camera first; expect it to be a config change.
- Recovery when the USB attachment goes stale: `usbipd list` → `usbipd detach --busid 4-18` → `wsl --shutdown` → (admin PowerShell) `usbipd attach --wsl --busid 4-18` → verify with `lsusb` and `ls -l /dev/video*`. `--auto-attach` is an option later.
- Code (`ego-vision/`): `src/` (capture.py, frame_buffer.py, camera_worker.py), `scripts/` (camera_test.py, camera_preview.py, test_frame_buffer.py). Run as `python -m scripts.camera_preview`. Abstractions: `CameraCapture(source, width, height, fps, pixel_format)`, `FramePacket(frame_id, timestamp_ns, image)`, thread-safe `LatestFrameBuffer` (keeps newest frame, counts drops), `CameraWorker` (acquisition separate from processing).

### Hardware
- Home PC: RTX 4060 8GB, Ryzen 7 7700X, reached via WSL2/Tailscale SSH. Training and the live demo both have to fit this.
- College supercomputer (8× H100; the "100 GB" is VRAM, not disk) confirmed for about 2 weeks from Oct 14, possibly longer, unconfirmed. Ask whether it has persistent storage before depending on it for data.

### EPIC-KITCHENS-100 (verified Oct 14)
- About 100 h, 45 kitchens, head-mounted, about 90K action segments in 700 videos, 97 verb classes and about 300 noun classes, long-tailed.
- Annotations repo: `epic-kitchens/epic-kitchens-100-annotations`. Files: `EPIC_100_train.csv` (15 columns, including start/stop info and verb/noun and their classes), `EPIC_100_validation.csv`, `EPIC_100_test_timestamps.csv`, `EPIC_100_verb_classes.csv`, `EPIC_100_noun_classes.csv`; pickle versions exist (v1.1.0 added per-video fps, duration, resolution).
- Download repo: `epic-kitchens/epic-kitchens-download-scripts` (Python 3.5+, no external libraries). Key flags: `--videos`, `--rgb-frames`, `--specific-videos P01_01,P02_122`, `--participants`, `--output-path`. Data lands in an `EPIC-KITCHENS` subfolder of the output path, one folder per participant. Previously downloaded files are skipped, so batches can run over several sessions. `epic_video_downloader_verify.py` adds resume (HTTP Range) and MD5 verification with a CSV report.
- Test command for day one:
```
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
cd epic-kitchens-download-scripts
python epic_downloader.py --videos --specific-videos P01_01 --output-path ~/data
```
- **Known problem:** the README says the script can fail with a "connection reset by peer" error depending on region, and that speed can be very slow. Fallbacks it recommends: **Academic Torrents** (links in the README), plus a Hugging Face mirror (`awsaf49/epic_kitchens_100`; one tutorial quotes 464 GB for all videos, so download only a subset) and a pre-cut clips dataset (`lightly-ai/epic-kitchens-100-clips`, about 24 GB; check its coverage before relying on it). He is in India, so the region issue is a real risk.
- Pre-extracted RGB frames come as one `.tar` per video at full frame rate; for his own sampler, downscaled videos (D8) are likely smaller to store.
- A reference for the "no clip boundaries in live video" problem: `iranroman/ego_actrecog_analysis` runs fixed-size sliding windows over full EPIC videos (including TSM models).

### Agent harness
OpenCode on WSL, GLM 5.3 Flash orchestrator, about 34 sub-agents on mostly free providers (Gemini, Mistral, Big Pickle, OpenCode Zen and others) with many skills.

---

## 12. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Data download fails or is very slow from India | Medium–high | Test day one; torrent/HF fallback; download a small subset only |
| Verb model does badly on his webcam (domain gap) | Medium | Evaluation-only webcam check Oct 29; ladder so Tier 1/2 still ship |
| Live predictions fire noisily with no clip boundaries | High | D3 (background class or thresholds) and D6 (smoothing) planned explicitly |
| RSNA week eats W2 | High | W2 targets are minimum-viable (sampler + state machine) |
| Late-Nov internals squeeze W7 | Medium | W7 is buffer only; freeze Nov 29 |
| Head mount (cap/helmet) wobble ruins footage | Medium | Test the mount in W1–W3 with the new webcam; keep a stand fallback |
| Pretrained-weight license | Low | One source lists DINOv2 repo weights as non-commercial (CC-BY-NC); another lists the Hugging Face `dinov2-small` as Apache 2.0. Fine for a personal demo; check before any commercial use |
| Drifting back into "agent writes it all" | Medium | `core/` rule in agent config; cold-rebuild exam in W7 |

---

## 13. Resources

**Dataset**
- ✅ EPIC-KITCHENS-100 annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- ✅ EPIC-KITCHENS download scripts: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- ✅ EPIC-KITCHENS-100 paper ("Rescaling Egocentric Vision"): https://arxiv.org/abs/2006.13256 (journal version: https://link.springer.com/article/10.1007/s11263-021-01531-2)
- ✅ Tutorial on loading the annotations and a video subset: https://lightly.ai/product-updates/epickitchens-100-in-lightlystudio
- ✅ Sliding-window evaluation reference: https://github.com/iranroman/ego_actrecog_analysis
- ✅ Furnari's recognition/anticipation code (for the later anticipation track): https://github.com/antoninofurnari/rulstm

**Fundamentals**
- ✅ CS231n Assignment 1 (kNN, softmax, two-layer net, features, FC nets; starter code linked from the page): https://cs231n.github.io/assignments2025/assignment1/
- ◻ CS231n linear classifier notes: https://cs231n.github.io/linear-classify/
- ✅ d2l.ai batch normalization: https://d2l.ai/chapter_convolutional-modern/batch-norm.html (ResNet is the next section of the same chapter)
- ✅ ESIEE lab "Build a Residual CNN" (manual BatchNorm vs PyTorch, residual blocks; TODO-style): https://perso.esiee.fr/~chierchg/deep-learning/concepts/residual-cnn-student.html
- ✅ PyTorch Transfer Learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- ✅ PyTorch "Learning PyTorch with Examples": https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html
- ◻ Karpathy, "A Recipe for Training Neural Networks": https://karpathy.github.io/2019/04/25/recipe/
- ◻ Karpathy, "Let's build GPT" (YouTube) for attention from scratch
- ◻ d2l.ai attention chapters: https://d2l.ai/
- ◻ Aakash N S, "PyTorch: Zero to GANs" Part 6 (CIFAR-10 ResNet, runs on Kaggle), found by name on Jovian

**Temporal modeling**
- ✅ TSM paper (ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
- ✅ TSM code (the shift is about 3 lines in `ops/temporal_shift.py`; paper also describes an online mode for live use): https://github.com/mit-han-lab/temporal-shift-module

**Scan phase**
- ✅ DINOv2: https://github.com/facebookresearch/dinov2 (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, 384-dim embeddings for ViT-S)
- ✅ Embeddings + a simple classifier on top of them: https://blog.roboflow.com/how-to-classify-images-with-dinov2/
- ✅ MediaPipe Hand Landmarker (for the D4 hand-crop option): https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- ◻ Prototypical Networks (Snell et al., 2017), arXiv 1703.05175

---

## 14. After December (not part of this roadmap)

- Feb 2027: first preprint target; next project's preprint by May/June 2027. Anticipation (Stage 2) belongs here: causal split, top-5 evaluation, `rulstm` as a reference. Whether there is a genuinely novel angle (for example, when to intervene) is a Dec/Jan decision once real results exist.
- Optional stretch (only with slack): frozen Ego4D-pretrained features (LaViLa/EgoVLP) vs the ResNet baseline.

---

## 15. Open questions and progress log

**Open questions for Harshil:**
- Q1: With Day 100 landing about Dec 17 at one reel/day, will the finale reel be queued in advance, or will the series pace change?
- Q2: Free disk space on the PC and whether the H100 box has persistent storage (decides the data subset size, D8).
- Q3: Exact dates of the late-Nov internals/midterm and the end-sem exams (W7 and Dec plan).
- Q4: Ingredient list and recipe (the master file mentions about 10 ingredients; the demo recipe needs about 3 to 4).
- Q5: Head-mount choice (cap vs helmet) and when it gets tested.

**Progress log (append dated one-liners; newest at the bottom):**
- 2026-10-14: Roadmap created. Nothing from sections 6 to 7 started yet.
- 2026-10-15: W1 and W2 rewritten day by day (college off through Oct 20; Oct 14 data test had not happened, so W1 starts Thu Oct 15).
