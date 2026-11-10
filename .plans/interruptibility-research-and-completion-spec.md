# Spec: finish the working model, then pilot "interruptibility from hands"

Written 2026-11-10 from an interview with Harshil. Supersedes the research
framing in `egocentric-vision-master.md` ("when to intervene" as an open gap).
Frontend is out of scope.

## 1. Where the project stands (checked 2026-11-10)

- Git: 57 commits on `glm-window`, all authored by Harshil, no co-author
  trailers, in sync with `origin` except one local commit (`4e3934b`,
  .gitignore). `main` is 51 commits behind `glm-window`.
- Tests on native Windows (global venv): 94 passed, 10 skipped. Reason for
  the 10 skips not yet checked.
- Plumbing works end to end (scan phase, matcher, state machine, smoother,
  recipe prior, web app, clip and live modes).
- The verb model does not work: mean-class accuracy 0.2205 against a 0.189
  majority baseline, best at epoch 1, then overfit (2,402 train clips).
- Configs still point at WSL paths (`/mnt/ssd/...`). The data is now at
  `C:\ssd-data\epic-kitchens` (55 downscaled videos present) and
  `C:\ssd-data\epic-tools`.
- `core/` (the hand-written pieces) does not exist yet. Not a blocker.
- Storage: C: 209 GB free, N: 45 GB free. GPU (RTX 4060 8 GB) is free.

## 2. Decisions from the interview

| Topic | Decision |
|---|---|
| Research gap | Interruptibility from hands (section 4) |
| Verb model | Frozen DINOv2 per-frame features + small temporal head |
| 5-day shape | Demo first (days 1-2), then research pilot (days 3-5) |
| Pilot type | Measure human experts first (analysis, not a new model) |
| Code handover | Working code, shapes in docstrings; no stub/worked-example overhead |
| Compute | GPU free for long runs; large downloads allowed |
| Storage | New data under `C:\ssd-data\`; check free space before each download |

## 3. Part A: working verb model (days 1-2)

1. Make data paths work on Windows (config values, no code forks).
2. Extract DINOv2 ViT-S/14 features once for every annotated segment of the
   55 videos (T frames per clip, 384-d each); cache to disk.
3. Train temporal heads on cached features: mean-pool, then temporal
   attention. Same internal split (P22 + P30 held out), same metrics module,
   majority baseline beside every number.
4. Known risk: per-frame features carry little motion, so open/close and
   take/put may stay confused. If mean-class accuracy stays under about 0.40,
   add frame-difference features or try a video backbone. This fallback
   needs Harshil's sign-off because it changes the backbone decision.
5. Re-run the smoother and loop evaluations with the new model; refresh the
   demo preset.
6. Run clip mode (Tier 2) on a held-out video on Windows.

Done means: a results table with real numbers, a checkpoint the app loads,
and a Tier 2 run on Windows. The live head-mounted run (Tier 3) needs
Harshil and the camera and is not part of this spec.

## 4. Part B: research direction

### What the literature search showed (abstracts only, 2026-11-10)

"When should an egocentric assistant speak" is no longer open. It has
benchmarks and a challenge:

- Vinci2 / EgoServe / EgoMemo (Sato lab, arXiv 2607.11523): proactive
  assistance as a context-dependent decision; 3,000+ service instances.
- EgoProactive track, ECCV 2026 Wearable AI Challenge: intervene or stay
  silent after each 8-second segment.
- ProAssist (2506.05904), Pro2Assist (2605.04227), EgoPro-Bench
  (2605.07299), P3IBench (2610.21568), Streaming Interventions
  (2606.09547), YETI (2501.09355), EgoVoice (2610.12248).
- Speaking before the error: TRAFA (2605.24526, assembly, with a user
  study), MistExit (2603.14252), PREGO / TI-PREGO, PROVIA (2609.20638,
  uses a false-alarm budget).
- Cost-sensitive gating: PRISM (2602.01532, text agents), ProMemAssist
  (2507.21378, working-memory model on smart glasses).

Nearly all of these decide timing from what the task needs. They score a
prediction against an annotated time. They do not model whether the person
can absorb an interruption at that moment.

### The proposed gap

Human-factors research shows that interruptions cost less at task
breakpoints than in the middle of an action (Powers and Scerbo 2023;
Hirsch et al. 2025). In egocentric video the hands show those breakpoints:
contact, release, tool in hand, both hands busy. Hand-object interaction is
the Sato lab's core topic.

Hypothesis: hand-object state predicts the cost of an interruption, and
good timing is a trade between that cost and the urgency of the message.
An assistant should wait for the next hand-free moment unless the coming
mistake cannot be undone.

Status: one search found no paper that does this. That is weak evidence.
The claim is unverified until the full-text check in step B1.

### Pilot (days 3-5)

- B1. Read the full text of Vinci2, Pro2Assist, TRAFA, ProMemAssist,
  PROVIA, EgoPro-Bench, Ego-EXTRA and HoloAssist. Write a gap map: what each
  one measures about timing, and whether any uses hand or motor state.
  Stop and report if the gap is already taken.
- B2. Pick the dataset with real instructor speech and time stamps
  (Ego-EXTRA or HoloAssist). Check licence, size and access time first;
  access approval may take longer than 5 days, which would force the other
  dataset or annotation-only analysis.
- B3. Measure: where do expert interventions fall relative to action
  boundaries and hand-contact state, compared with a shuffled-time
  baseline? Do corrections of mistakes break the pattern more often than
  routine instructions?
- B4. Write up the result, including a null result, with the method and
  every number traceable to a command.

Not in the pilot: a new timing model, a user study, any VideoLLM work.

## 5. Open items

- Why 10 tests skip on Windows.
- Whether `main` should be fast-forwarded to `glm-window`.
- Push of commit `4e3934b` (not pushed; needs Harshil's go-ahead).
- Foundations list to add beside kNN/softmax/BN/LN/CNN/RNN: attention and
  transformers (ViT), self-supervised features (DINO), calibration and
  thresholds (precision/recall, ROC), basic hypothesis testing (permutation
  tests, confidence intervals) for step B3.
