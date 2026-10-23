# Egocentric Vision Project — Master Context File

*Snapshot as of Sept 28, 2026 (end of Week 2). Consolidates the original plan, the Ego4D
addendum, and everything decided or built in the Week 2 chat. Where this file's Part 1–3
conflict with the original plan text in Parts 4–5, **Parts 1–3 win** (they are newer).*

## How to use this file in a new chat
- Treat Part 1 as current state, Part 2 as the decisions log, Part 3 as working guidance.
- Parts 4–5 are the original documents, unedited, kept for full context.
- The user (Harshil) dictates through Wispr Flow, so transcription errors happen
  (e.g. "Wiser" was meant to be **VISOR**). Interpret sensibly and confirm if unsure.
- Harshil wants to be in the driving seat: he makes design and architecture decisions, and
  wants to understand every function. Be direct and factual; he has said he does not need
  reassurance.

---

# Part 1 — Current State (Sept 28, 2026)

## Where he is on the roadmap
- **Phase 0, end of Week 2** (Sep 22–28). Next: Week 3 (Sep 29 – Oct 5), ending at Checkpoint A.
- **Done:**
  - Webcam capture pipeline (see Part 3E) — deeper than the plan required.
  - Runtime decision: WSL2, tested and working.
  - Ego4D paper: Pass 1 done plus a YouTube talk by the original authors; Pass 2 partial
    and selective (deliberately not reading all ~91 pages; see Part 3A).
- **Not yet done from Week 2:**
  - EPIC-KITCHENS-100 paper and VISOR overview (not started at the time of writing).
  - Final 5–10 many-shot verb classes (candidate list in Part 3B, unverified against data).
  - 1–2 fixed demo recipes sketched from that vocabulary.
  - Confirm the ordered external webcam has arrived. The pipeline was validated on the
    laptop's *Integrated Camera*, not the new unit.
- **Correction to the plan's framing:** the plan says CS231n is "fully complete." In practice
  he finished the *lectures* but had not done the assignments (see Part 3D).

## Week 3 (Sep 29 – Oct 5) — unchanged from plan
- Read Ego-Exo4D, Sato Lab's ego-exo survey, Vinci2.
- Finalize the Stage 1 architecture spec on paper.
- Test EPIC-KITCHENS-100 data access (annotations + selected clips).
- Build the recipe-order state machine against mocked inputs (now to be **hand-written**,
  see Part 3D).
- Checkpoint A: Oct 5.

---

# Part 2 — Decisions Log (from the Week 2 chat)

1. **Runtime = WSL2**, not native Windows. Reason: the whole ML stack already lives there.
2. **Webcam Step 3 (alternate-source swap test) skipped** as optional. It will effectively
   be exercised when the external webcam arrives.
3. **Ego4D is background reading only.** EPIC-KITCHENS-100 is the training dataset. VISOR
   is a segmentation add-on, not used in Stage 1/2 right now. No plan to extract a kitchen
   subset from Ego4D.
4. **No self-pretraining on Ego4D** (1,536 GPU-hours class of compute vs an RTX 4060 8GB).
   Optional Week 8 stretch only: frozen features from a public Ego4D-pretrained checkpoint
   (LaViLa / EgoVLP) vs the ResNet-34 baseline. Dropped entirely if the buffer is needed.
   Full text in Part 5.
5. **Skip the Kaggle "Intro to Computer Vision" course** as a fundamentals check. It uses
   pre-built Keras layers, so it would not test hand-implementation.
6. **Fundamentals plan adopted** (Part 3D): shape-first blank-file rebuilds of project
   components, about 45–60 min/day. Cut order if Checkpoint A is at risk: drop the two-layer
   net, keep kNN → softmax → metrics → sampler.
7. **Hand-written checkpoint:** Week 7's temporal-attention module is written by hand
   before any agent touches it (from the LLM Council verdict, Part 3F).
8. **Week 2 → Week 3 spillover accepted:** reading and verb-class selection are the
   remaining Week 2 items and should not push into Week 4.

---

# Part 3 — Working Guidance

## 3A. Reading guidance

**Three-pass method (Keshav):**
- Pass 1 (~5–10 min): title, abstract, headings, conclusion, figures, references. Decide
  relevance.
- Pass 2 (~45–60 min): read in order, skip proofs and derivations, study figures and
  tables, list unfamiliar terms and citations.
- Pass 3 (hours): re-derive or re-implement mentally, question every assumption. Reserve for
  papers the project depends on (Ego4D, EPIC-KITCHENS-100).

**Ego4D (he printed the ~91-page version).** Facts he has verified: 3,670 hours, 931 camera
wearers, **74 locations, 9 countries**.
- Read carefully: abstract and intro; related work / dataset comparison table; data
  collection and diversity; benchmark overview, especially **Forecasting** (closest to
  Stage 2) and **Hand-Object Interaction** (background for the Sato Lab direction).
- Skim: Episodic Memory, Audio-Visual Diarization, Social Interaction; ethics and privacy.
- Skip: per-site collection appendices, consent forms, annotation UI, extended tables.
- Sample clips without a license: the interactive world map on ego4d-data.org and the
  interactive Figure 1 page (ego4d-data.org/fig1.html). Raw dataset access needs a signed
  license (roughly 48 hours to approve). Worth starting early, alongside the EPIC-KITCHENS
  access test.

**EPIC-KITCHENS-100.** About 100 hours of unscripted head-mounted kitchen video, 45 kitchens,
Full HD at 50 fps, ~90,000 action segments (~700 videos), labeled as (verb, noun) pairs from
97 verb classes and 300 noun classes. Long-tailed. Benchmarks: recognition, anticipation,
detection, cross-modal retrieval, domain adaptation. Challenges: class imbalance and
heavy occlusion.

**VISOR.** Pixel-level segmentation masks for hands and active objects on a subset of
EK-100 frames. Read the ~10-page main body (annotation pipeline, statistics, benchmarks);
skip the appendices. Background only for now.

## 3B. Candidate many-shot verb classes (UNVERIFIED — confirm against the data)
"Many-shot" in EPIC-KITCHENS = verb and noun each appear more than 100 times in training.
Candidate set: **take, put, wash, cut, open, close, mix, pour** (add turn-on/off or move
if a recipe needs it).

Example recipe using only these verbs: take tomato → wash → cut → put in bowl → take
cucumber → cut → put in bowl → pour dressing → mix.

**Verification step (do not skip):** load the official annotation files (for example
`many_shot_verbs()` in the `epic_kitchens` package or the annotations repo) and check that
the specific verb–noun pairs the recipe needs (tomato, cucumber, bowl, etc.) are actually
many-shot, not just the verbs.

## 3C. Week 2 daily plan as originally laid out (Sep 22–28)
Tue: order webcam, start Ego4D. Wed: finish Ego4D notes. Thu: capture skeleton and WSL2
vs Windows test. Fri: EK-100 paper. Sat: finish EK-100 and VISOR. Sun: verb classes and
recipe sketch. Mon: buffer. (Only the engineering items and the Ego4D reading actually
happened on schedule.)

## 3D. Coding recall-gap plan (built around this project)

**Diagnosis.** About 6 months into deep learning. Done: CS231n lectures, ML basics,
micrograd and makemore, hand-derived backprop. Recognition is strong (reads and explains
AI-written code line by line). Recall is weak: blanks from an empty file, struggles to map
formulas to indexing and shapes, does not recall numpy verbs (argsort, bincount). Started
CS231n Assignment 1 (kNN) without AI and got stuck on which rows go into the L2 distance
function and on predict_labels. Reading is recognition; writing from a blank file is
generation, and recognition does not train generation.

**Method for every rebuild:**
1. Write shapes first, as comments: input shape, output shape, one line per step.
2. Blank file, 25-minute attempt, no AI. After 25 minutes, official docs only; log every
   function looked up.
3. Test on a tiny hand-computable array; assert shapes.
4. Redo cold the next day and again 3–4 days later.

Unblocking rule for the L2 function: the output is `(num_test, num_train)`, so
`dists[i, j]` pairs test row `i` with train row `j`. Fix the output shape first.

**Rebuild order (each maps to a project piece):**

| When | Rebuild from blank | Project use |
|---|---|---|
| Week 3 | kNN (L2 with loops, then vectorized, `predict_labels`) | broadcasting, `argsort`, `bincount` |
| Week 3 | Softmax loss + gradient | verb/noun head loss; `scores[np.arange(N), y]` |
| Week 3 | Metrics module: top-1, top-5, per-class accuracy | Stage 1 and 2 evaluation |
| Week 3 | Recipe state machine (already a plan task) | control flow; mocked action streams |
| Week 4 | Uniform frame sampler (start_frame, stop_frame, T → indices) and anticipation window (frames before start − τ) | data pipeline; Stage 2 causal split |
| Week 5 | Two-head model + training loop, overfit 32 samples | Stage 1 baseline |
| Weeks 6–7 | Temporal aggregation reshape (B,T,C,H,W) ↔ (B·T,…), then temporal attention | Week 7 hand-written checkpoint |
| Week 8 | Cold-rebuild exam: metrics, sampler, attention, 25 min each | exit test |

Starter numpy verb list to log and drill: argsort, bincount, argmax, fancy indexing,
linspace, where.

**Verification drill (for the "can't verify AI code" worry).** After writing a function,
have an agent write the same one with an undisclosed subtle bug (off-by-one in the frame
window, wrong softmax axis) and find it with tests. The window bug matters most because it
silently leaks future frames into anticipation.

**Rule going forward.** The components in the table are hand-written first. AI may take
config and CLI, download scripts, logging, plotting, and capture plumbing.

**Time budget.** About 45–60 min/day, one block; he confirmed this is acceptable. If
Checkpoint A is at risk, cut the two-layer net and keep kNN → softmax → metrics → sampler.

## 3E. Webcam capture pipeline — final state (Sept 24, 2026)

- Runtime: WSL2 with `usbipd-win` USB passthrough; camera shows as "Integrated Camera"
  (Microdia, VID:PID 0c45:64ab, BUSID 4-18). `/dev/video0` = video stream,
  `/dev/video1` = metadata only. Permissions fixed via `video` group (no sudo). OpenCV 5.0.0,
  `v4l-utils` installed.
- Modes: MJPEG 640×480 @ 30 and 1280×720 @ 30; YUYV 640×480 @ 30 and 1280×720 @ 10.
- **Validated baseline: 640×480 MJPEG, requested 30 FPS, measured ~15–18 FPS.**
  720p negotiates but times out under WSL2 USB/IP; YUYV streaming hangs. The app must
  explicitly request MJPEG because the camera can come back as YUYV after a reset.
- Recovery when the USB attachment goes stale: `usbipd list` → `usbipd detach --busid 4-18`
  → `wsl --shutdown` → (admin PowerShell) `usbipd attach --wsl --busid 4-18` →
  verify with `lsusb` and `ls -l /dev/video*`. `--auto-attach` is an option later.
- Code structure (project root `ego-vision/`): `src/` (capture.py, frame_buffer.py,
  camera_worker.py) and `scripts/` (camera_test.py, camera_preview.py,
  test_frame_buffer.py). Run as `python -m scripts.camera_preview`.
- Abstractions: `CameraCapture` (source, width, height, fps, pixel_format), `FramePacket`
  (frame_id, monotonic timestamp_ns, image), thread-safe `LatestFrameBuffer` (keeps newest
  frame, counts drops), `CameraWorker` (acquisition separated from processing).
- Clean shutdown via q, ESC, window close, Ctrl+C. Harmless Qt font warning left alone.
- Step 3 (alternate source) skipped; a future 1080p60 webcam should be a config change.
- Still out of scope here: ML model, inference, recipe state machine, training.

## 3F. LLM Council verdict on agent-heavy workflow and fundamentals

- Agreed: automation of no-learning-value tasks (streak bots, DM automation, setup scripts)
  is harmless; keeping DSA AI-free is the right line.
- Tension: agent-written model code can erode implementation fluency vs. orchestration and
  architectural judgment being a real skill. Resolution: judgment is built on hands-on reps,
  not instead of them.
- Recommendation: leave the wider automation stack alone; for this project, hand-implement
  the component that *is* the research idea at each stage. First checkpoint: the Week 7
  temporal-attention module, written without an agent.
- After the council, he identified the concrete gap: CNN machinery had only ever existed as
  lecture content because the CS231n assignments were skipped. That led to Part 3D.

## 3G. Other artifacts and content notes
- A Notion page, "Paper Reading Notes — Pass 1 & 2" (draft, pre-filled for Ego4D), exists for
  note-taking while reading; duplicate it per paper.
- A webcam setup handoff file (`webcam-capture-setup.md`) was produced for a cheaper agent
  and has since been completed.
- 100-day Instagram series: Day 1 = Aug 28, 2026, ending Dec 6. He records short (10–15 s)
  voiceover reels; the finalized Ego4D reel script and a webcam-bridge reel script were
  written. Screen recordings pair the paper on one side with Ego4D sample clips or the
  live camera pane on the other.

---

# Part 4 — Original Project Plan (verbatim: `egocentric-vision-plan.md`)

*Sections below are unedited. Heading levels are shifted down one step so this file stays
navigable. Where they conflict with Parts 1–3, Parts 1–3 win.*

## Egocentric Vision Project — Implementation Plan (Sept 15 → Dec 8, 2026)

### 0. Framing

**Dec 8 is a checkpoint, not a publication deadline.** It marks the completion of a rigorous
Stage 1 + Stage 2 technical report on EPIC-KITCHENS-100 (recognition + anticipation), **plus
a live, working demonstration** of that system on your own webcam. Whether this becomes a
genuinely novel paper — with a real gap identified and tested — is a decision made in
December/January with actual results in hand, feeding into MEXT professor outreach
(Aug 2027), not forced onto this 12-week window.

**The final vision, and where Dec 8 fits in it.** The actual target — a head-worn camera that
watches you work (e.g. through a lecture) and learns when to proactively interject — is not
this project. It's multi-year, and it's genuinely an open research question (Sato Lab's Vinci2
work is doing something adjacent at far larger scale). Dec 8 builds the first rung toward it:
a constrained, working version of "watch → recognize → anticipate → decide to say something,"
in a kitchen instead of a lecture hall, with a hardcoded recipe instead of open-ended context.
If the intervention-timing piece proves interesting once this is running, that may become the
actual novel angle this project has been missing — worth deciding once Stage 1+2 results and
a working demo both exist, not before.

**Control philosophy.** Every function in this build gets written and understood by you, not
handed off wholesale to an agent. AI tools can generate code, but you review and understand it
down to the function level before it's considered done — the same standard you wished you'd
held yourself to on MVSS-Net Lite.

**Compute:** Home PC, RTX 4060 8GB VRAM + Ryzen 7 7700X, via WSL2/Tailscale SSH. This caps
*training* architecture choices: ResNet-34 backbone (not large ViT), aggressive frame
subsampling, small batch size with gradient accumulation, mixed precision throughout. The live
demo is inference-only at low frame rate, so it is not a compute risk the way training is —
budget your concern accordingly.

**Known concurrent load:** SIH 2026 (leading team, in research-plan phase), college mid-sem
exams (dates still TBD), gym 5x/week (non-negotiable), Japanese study, DSA, and the 100-day
Instagram series (continues in parallel — it's the accountability log for this exact plan, and
the live demo footage doubles as strong closing content for it). CS231n is now fully complete
(Sept 20, 2026) — ahead of the original schedule, which frees up real time in Week 1.
Mastercard Innovation Challenge is already submitted (Aug 31) — no longer a conflict.

**Focused time available:** ~2–3 hrs/day deliberate focus + passive time throughout the day
(~14–21 focused hrs/week). All estimates below are built against that budget, not a full-time
pace.

---

### 1. Week-by-Week Plan

#### Phase 0 — Reading List + Demo Groundwork (Weeks 1–3, Sept 15 – Oct 5)

**Week 1 (Sep 15–21):**
- Read Ego4D dataset paper (+ notes). CS231n being done ahead of schedule frees this week up.
- Build the webcam capture skeleton: USB-extended webcam → frame grab loop → basic live
  display window. Zero ML dependency — this can be built and tested now, in parallel with
  reading, and gets your hardware rig (mount, extender, positioning) sorted early rather than
  as a last-minute scramble in November.

**Week 2 (Sep 22–28):**
- **Order the webcam today.** You have the USB extenders already, but the webcam itself isn't
  ordered yet — this is now more time-critical than the reading, because shipping has a lead
  time you don't control, while reading is compressible. Don't let this slip further.
- Don't block on the shipment, though: build and test the capture pipeline script now using
  whatever camera you already have (laptop webcam, or a phone via a virtual-cam app) — the
  logic (frame grab loop, display window) doesn't care which physical device it's reading
  from, so swapping in the real webcam once it arrives is a one-line change, not a rewrite.
- **Test the WSL2 capture path early, not on demo day.** Your home PC setup runs through WSL2 —
  accessing a physical USB webcam from inside WSL2 needs USB passthrough (usbipd-win) and can
  be finicky. Decide now whether the capture/demo code runs natively on Windows (simpler, and
  you'll be sitting at the desk anyway when actually demoing) or through WSL2, and test that
  decision today with whatever camera is on hand — the same "test it before you need it"
  principle already applied to EPIC-KITCHENS data access.
- Read EPIC-KITCHENS-100 dataset paper(s), including the VISOR extension overview
- Identify your 5–10 many-shot verb classes for the Stage 1 subset — **choose these with the
  eventual demo in mind**: pick classes that plausibly cover common food-prep actions (cut,
  put, wash, open, mix, take, etc.), since this same vocabulary becomes your demo's action set.
  Don't pick two separate class lists for training vs. demo — one list, used for both.
- Sketch 1–2 fixed demo recipes built entirely from that class vocabulary (e.g. "add tomato →
  add cucumber → mix"). Keep it to a small, hardcoded set — this is a demo, not a general
  cooking assistant.

**Week 3 (Sep 29 – Oct 5):**
- Read Ego-Exo4D, Sato Lab's ego-exo cross-view survey, Vinci2 paper
- Finalize Stage 1 architecture spec on paper (backbone, temporal aggregation approach,
  head design) before writing any code
- **Test data access**: confirm you can actually download/access the EPIC-KITCHENS-100
  subset you need (annotations + selected clips) — this is a logistics task, not a reading
  task, and it belongs here, not Week 4
- Build the recipe-order state machine against mocked/fake inputs (no trained model yet):
  track completed steps, compare the next detected action against the expected next step,
  flag correct/incorrect order, decide when to surface a suggestion. This is plain software
  logic, not ML — there is no reason to wait for Stage 1/2 to be trained before building it.
- By the end of this week, the capture pipeline and the state machine should both be
  demo-ready and simply *waiting* for a trained model to plug in.

**Checkpoint A (Oct 5):** Reading done, architecture spec written, data access *tested and
working* for your chosen subset, capture pipeline + state machine built and tested against
mocked inputs.
→ **If missed:** Drop Ego-Exo4D or the survey (keep Ego4D + EPIC-KITCHENS + Vinci2 as the
non-negotiable minimum). Do not let reading or demo-groundwork slip push into Week 4 — Stage 1
starts Oct 6 regardless.

---

#### Phase 1 — Stage 1: Recognition Build (Weeks 4–8, Oct 6 – Nov 9)

**Week 4 (Oct 6–12):**
- Data pipeline: download annotations + selected clips, preprocessing, frame subsampling,
  train/val split matching the official protocol
- Treat this as infra work only — no model training yet. If download/storage becomes a
  bottleneck, this is the week to fight that battle, not later.

**Week 5 (Oct 13–19):**
- Single-frame baseline: ResNet-34 backbone → two heads (verb, noun)
- Get something training end-to-end on your subset; establish baseline accuracy numbers,
  even if mediocre

**Week 6 (Oct 20–26):**
- Extend to temporal aggregation (multi-frame/clip input) over the single-frame baseline
- Compare against Week 5 numbers

**Week 7 (Oct 27 – Nov 2):**
- Add temporal attention (CBAM-style instinct carried over from MVSS-Net Lite, applied
  temporally instead of spatially-only)
- Compare all three variants (single-frame / temporal-agg / temporal-attention)

**Week 8 (Nov 3–9):**
- Buffer week — absorbs exam/SIH slippage from Weeks 4–7
- Consolidate Stage 1 results into a clean table; write the Stage 1 section draft (method +
  results) while it's fresh

**Checkpoint B (Nov 9) — hard gate for Stage 2:** Stage 1 fully trained, evaluated, documented.
→ **If missed:** Stop. Do not start Stage 2. Skip straight to full write-up using Weeks 9–11
(three weeks instead of one) for a Stage 1-only report. Attempt the live demo with the Stage 1
(recognition-only) model if time allows — ingredient identification alone is still a
legitimate, honest demo even without anticipation. This protects the Nov 30 draft wall below —
it does not get renegotiated.

---

#### Phase 2 — Stage 2: Anticipation Build + Live Demo Integration (Weeks 9–10, Nov 10–23)

*(Only proceeds if Checkpoint B was met.)*

**Week 9 (Nov 10–16):**
- Causal data re-splitting: only pre-action footage as input, action start timestamps as the
  cut point
- Adapt Stage 1's best-performing architecture (not building anticipation from scratch);
  set up top-5 accuracy evaluation

**Week 10 (Nov 17–23):**
- Train, tune, and finalize anticipation results
- Consolidate a combined Stage 1 + Stage 2 results table
- **Wire the trained Stage 1 + Stage 2 models into the capture pipeline and state machine
  built back in Weeks 1–3.** Because that plumbing already exists and is already tested, this
  is a swap-in-the-model step, not new engineering — budget ~2–3 focused days, not a full week.
- Record a short demo clip in your kitchen running your 1–2 target recipes end-to-end (webcam
  → recognition → anticipation → suggestion). This clip becomes both a figure/section in the
  report and closing material for the Instagram series.

**Checkpoint C (Nov 23):** Stage 2 trained, evaluated, documented.
→ **If missed:** Write up Stage 1 as the core result, mention Stage 2 as an in-progress /
future-work section rather than a completed result. Do not let an unfinished Stage 2 delay
the draft.

**Checkpoint D (Nov 23):** Live demo running end-to-end on your chosen recipe(s), with a
recorded clip in hand.
→ **If missed:** Cut the live demo from the report scope. Describe it as a built-and-tested
prototype (capture + state machine working, model integration pending) rather than a
completed demonstration. Do not let this threaten the Nov 30 wall — the benchmark results are
the load-bearing content either way.

---

#### Phase 3 — Writing (Week 11, Nov 24–30)

- Full draft: intro, related work (from your Phase 0 reading), method, experiments/results
  (Stage 1 + Stage 2 or Stage 1 alone per the checkpoints above), a system demonstration
  section covering the live kitchen demo (if Checkpoint D was met), honest limitations,
  conclusion
- Get at least one outside read — flag this now rather than assuming availability in
  late November; a mentor or peer review takes lead time to arrange

**Hard wall — Nov 30:** A complete, submittable draft must exist by this date, independent of
which checkpoint outcomes above applied. This is non-negotiable because of what comes next.

---

#### Phase 4 — JLPT Priority Week + Submission (Week 12, Dec 1–7, and Dec 8)

- **Dec 1–5:** Light revision only (typos, formatting, figures) — no new writing, no new
  results, no further demo work. Japanese prep takes priority for focused time this week.
- **Dec 6:** JLPT N5 exam. Zero project work.
- **Dec 7:** Recovery day — final proofread only, nothing substantive.
- **Dec 8:** Submit / finalize the technical report checkpoint.

---

### 2. Descope Triggers — Summary

| Trigger | Condition | Action |
|---|---|---|
| A | Reading/architecture/demo-groundwork not done by Oct 5 | Cut one paper, hold Stage 1 start date firm |
| B | Stage 1 not done by Nov 9 | Cancel Stage 2 entirely, write Stage 1-only report; attempt demo with Stage 1 model only if time allows |
| C | Stage 2 not done by Nov 23 | Report Stage 1 as core result, Stage 2 as future work |
| D | Live demo not working by Nov 23 | Cut demo from report scope; describe as built-but-not-integrated prototype |
| Wall | Draft not complete by Nov 30 | No exceptions — cut scope further, not the deadline |

The plan is designed so **every slip point resolves toward "ship something honest," never
toward "compress the JLPT week."** The demo is real scope now, but it is explicitly the first
thing to fall away under pressure — the benchmark report is the part that must exist regardless.

---

### 3. Council Verification (original Stage 1+2 plan)

This plan's core Stage 1+2 structure was run through a 5-advisor council stress-test before the
live demo was added.

**Contrarian:** Flagged the EPIC-KITCHENS download/storage logistics as the single highest-risk
unknown — 100 hours of source video, even filtered to a subset, can silently eat a week if
storage or bandwidth becomes a bottleneck, and the original draft buried this inside Week 4
alongside actual model work. Also flagged that "get outside feedback" in Week 11 was listed
with no owner or lead time.

**First Principles Thinker:** Asked why Stage 2 gets only 2 weeks when the roadmap itself
called anticipation "harder, more ambiguous" than recognition — the compression was justified
by infra reuse, but reuse doesn't shrink the *conceptual* difficulty, only the *engineering*
setup time.

**Expansionist:** Noted the plan doesn't explicitly say what happens to the extra time *if*
things go well (e.g., Stage 1 finishes early) — worth deciding in advance whether early
progress buys Stage 2 more time or gets banked as draft-quality buffer, rather than deciding
ad hoc in the moment.

**Outsider:** Pointed out the plan never states the 8GB VRAM constraint's actual impact on
timeline — it's mentioned as a design constraint but not costed into any week's time estimate,
which is inconsistent with how carefully everything else is scheduled.

**Executor:** Confirmed the descope triggers were the strongest part of the plan — concrete
dates, concrete actions, no ambiguity — but the original draft's Week 4 combined "download
data" and "preprocess data" into one line, understating it as a single task when it's really
two with different failure modes (access/bandwidth vs. pipeline bugs).

**Chairman's Verdict:** The plan's core structure — checkpoints that always resolve toward
descoping rather than compressing the JLPT week — is sound and doesn't need to change.

**Note on the live demo addition:** the demo was folded in after this council pass by
deliberately decoupling it from the ML-risk timeline — the capture pipeline and state machine
are pure software with no model dependency, so they were pulled into Weeks 1–3 (freed up by
CS231n finishing ahead of schedule) rather than added as new load on the already-tight Stage 2
weeks. Only the final model-swap step lands in Week 10, and it has its own descope trigger
(D) so it can fail without threatening the Nov 30 wall. This directly addresses what the
Executor's review would flag in a fresh pass: don't bury a new task inside an already-loaded
week, give it its own checkpoint and its own fallback.

**One thing to do first:** Before Week 1 ends, do a 30-minute dry run of the EPIC-KITCHENS-100
download/annotation access process — not to start real work, just to confirm the logistics
Checkpoint A depends on aren't a surprise in Week 3.


---

# Part 5 — Ego4D Pretraining Addendum (verbatim: `ego4d-pretraining-addendum.md`)

*Unedited; heading levels shifted down one step.*

## Addendum: Ego4D-Pretrained Backbone (Optional Stretch Goal)

*Companion note to `egocentric-vision-plan.md` — not a replacement, meant to be folded in
under Phase 1 (Stage 1: Recognition Build) or referenced from Week 8's buffer slot.*

### Where this came from
Tested the intuition: "what if I properly pretrain on Ego4D, then fine-tune on
EPIC-KITCHENS-100 (+ VISOR)?" — since that's a real, published pattern (EgoVLP, LaViLa both
do exactly this).

### Verdict: don't self-pretrain on Ego4D
The pattern is sound, but doing the pretraining step yourself is not feasible on this
project's hardware/timeline:
- EgoVLPv2's pretraining run reports **1,536 GPU-hours**
- LaViLa's public checkpoints were trained via multi-node Slurm jobs (4–8 nodes)
- Even papers explicitly optimizing for cheap pretraining ("Training a Large Video Model on
  a Single Machine in a Day") assume a well-resourced single machine, not an 8GB consumer card

This is a compute-class mismatch, not something a longer timeline fixes. It would consume
Stage 1's entire 4-week budget on its own, before EPIC-KITCHENS-100 work even starts.

### The feasible version of the same idea
Use an **already-published Ego4D-pretrained checkpoint** (LaViLa or EgoVLP release public
TimeSformer-based checkpoints trained this way) as a **frozen feature extractor** — extract
features once, train only lightweight classification heads on top for the EPIC-KITCHENS-100
verb/noun task. No backprop through the large pretrained backbone, so this fits comfortably
in 8GB VRAM.

#### Caveats
1. **Architecture mismatch** — these public checkpoints are TimeSformer/ViT-based, not
   ResNet-34. This is a second, parallel setup, not a drop-in swap into the existing Stage 1
   architecture.
2. **Scope discipline** — this must stay optional. It does not replace the ResNet-34 baseline,
   which remains the primary, compute-safe, guaranteed-to-work path.

### How it fits into the existing plan
- **Primary path — unchanged.** Weeks 5–7: ResNet-34 single-frame baseline → temporal
  aggregation → temporal attention, exactly as already planned.
- **Optional bonus (Week 8 buffer only).** If Stage 1 finishes early and there's no
  exam/SIH slippage to absorb, optionally test frozen LaViLa/EgoVLP Ego4D-pretrained features
  vs. the from-scratch ResNet-34 baseline as an ablation, and report the comparison if time
  allows. This could become genuinely interesting content ("does Ego4D-scale pretraining
  transfer to our kitchen task better than ImageNet init?") — but it is not required.
- **Hard rule:** if Week 8's buffer is needed to absorb slippage from Weeks 4–7 (as the
  original plan already anticipates it might be), this stretch goal is dropped entirely, no
  exception. It must never threaten Checkpoint B (Nov 9).

### One-line takeaway
Good intuition, wrong execution path — skip self-pretraining on Ego4D, keep the ResNet-34
plan as primary, and treat a public-checkpoint frozen-feature comparison as a stretch
experiment gated strictly by Week 8 buffer slack.
