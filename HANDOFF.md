# HANDOFF

GLM window deliverable, branch `glm-window` (pushed to
github.com/harshil-sri/ego-vision). Written 2026-10-24 ~01:20 IST;
numbers updated as runs completed (see the log at the bottom).
**Timezone correction 2026-10-25:** all "IST" timestamps below were read
off the machine clock, which is UTC — add 5:30 for actual IST.

## 1. What runs today (exact commands, from the repo root)

| Thing | Command | Status |
|---|---|---|
| Full test suite | `python3 -m pytest tests/ -q` | **102 passed** (re-run 2026-10-25; incl. smoother-alignment, prior, policy, restart-cycle, lifespan tests) |
| Tier 1, no camera | `python3 -m scripts.run_app --mode keyboard --synthetic` then open http://127.0.0.1:8000 | **WORKS** (verified headless: state, MJPEG, WS, full flow via smoke test) |
| Tier 1, webcam | `python3 -m scripts.run_app --mode keyboard` | **WORKS** — live 2026-10-24 ~06:15 IST, real-browser verified (wizard renders, MJPEG live, WS open, 19 fps). First real look found+fixed: `.overlay{display:flex}` overrode `[hidden]` so the stop-overlay covered the page; stop→restart cycle repaired too (tests/test_app_smoke.py) |
| Tier 1, OpenCV UI | same + `--ui cv` | Code complete; needs a display — NOT RUN headless |
| Tier 2 (clip) | `python3 -m scripts.run_app --mode clip --clip <mp4> --run-dir runs/<id>` | **WORKS end-to-end** 2026-10-24 10:35 IST vs runs/local_gpu_sanity on EPIC-256 P22_104: scans → enroll → cooking, model tick verb_predicted take@0.513, 155.7 ms CPU inference, smoother + state machine + window_sample strip all firing. (First dry run caught two real bugs — checkpoint key + tensor permute — now fixed + unit-tested.) **Correction 2026-10-25:** that `take@0.513` label was wrong — the smoother was built over the recipe's 6 verbs instead of the model's 8, so the model's `close` class was displayed as `take`. Fixed via `_smoother_verbs()` in engine.py; regression covered by tests/test_prior.py (its 8-verb fake source vs 6-verb recipe fails under the old mapping). The mechanics (tick → smoother → machine) were correct; only the label mapping was off. Needs a TRAINED checkpoint for meaningful verbs; the 1-epoch sanity model predicts, barely |
| Tier 3 (live) | `python3 -m scripts.run_app --mode live --run-dir runs/<id>` | **Wiring VERIFIED** 2026-10-24 11:26 IST (`--mode live --synthetic --run-dir runs/local_gpu_sanity`, no camera): scans → cooking, model ticks at 331.9 ms CPU (smoother correctly withheld low-conf preds on synthetic frames; events incl. window_sample + ingredient_matched logged). Live-camera run = same path + webcam, pending owner + trained checkpoint |
| Training (Kaggle) | `kaggle/` kernel on mahajantanish1/ego-vision-baseline | **RUNNING at handoff time** — see §4 |
| RTX 4060 latency | `python -m scripts.latency_bench --device cuda --img-size 224` | **DONE 2026-10-24 ~05:45** — resnet34 T=8 fp16 5.6 ms = 180 windows/s (≥3/s needed); table in DECISIONS_NEEDED.md D1/D5 |
| Matcher eval | `python -m scripts.eval_matcher --root data/ingredients` | Tool verified on synthetic; real-webcam frames NOT RUN |
| Smoother bake-off | `python3 -m scripts.eval_smoothers` | **DONE 2026-10-25** — 593 val segs via runs/eval_cache/val_preds.npz (built with fast_seek, ~90 s); reports/smoother_eval.md |
| Reasoning-chain eval | `python3 -m scripts.eval_loop --policy optimistic --leak 0.05` | **DONE 2026-10-25** — 10 val videos as ≤15-step pseudo-recipes; reports/loop_eval.md (+ strict variant) |
| Trained verb model (local) | `runs/baseline/best.pt` | **ep 1 of 12, val_mca 0.2205** vs majority-class 0.189 (run stopped at ep 7 at owner request, GPU freed; val mca declined after ep 1 — overfit) |

## 2. Quick start

```bash
cd ~/ego-vision && python3 -m pytest tests/ -q      # everything green?
python3 -m scripts.run_app --mode keyboard           # Tier 1 on the webcam
# -> http://127.0.0.1:8000 ; scan 3 ingredients, then number keys 1-6 emit verbs
# d = debug drawer, p = presentation, ? = help
python3 -m scripts.run_app --mode keyboard --synthetic   # no-camera variant
```

## 3. File map (one line per folder / key file)

- `src/pipeline/` — engine.py (phases, event flow, recipe-prior wiring, `_smoother_verbs()`), window.py (sliding window + camera/clip/synthetic sources), smoothers.py (D6), prior.py (recipe-aware verb prior), embedder.py (D7), model_source.py, overlay.py, events.py (bus), session_log.py, recipe.py
- `src/app/` — server.py (FastAPI: MJPEG/WS/REST), static/ (web UI)
- `src/models/` — backbones.py, variants.py (single_frame/tsn_avg/temporal_attn)
- `src/train/` — config.py (strict loader), loop.py (train ritual), eval_model.py
- `src/data/` — annotations.py, splits.py (V1 internal split), video_io.py, dataset.py, synthetic.py
- `src/registry.py` + `configs/impl.yaml` — reference/core switch, EGO_IMPL env
- `reference/` — the 5 core-piece references + exercises/ (knn, softmax, batchnorm)
- `exercises/stubs/` — copy to core/, fill in, test with `pytest tests/equivalence --impl core`
- `docs/` — ARCHITECTURE.md (shapes at every arrow), GLOSSARY.md, STUDY_QUEUE.md, worked/ (8 hand-traced examples)
- `recipes/example_salad.yaml` — placeholder recipe (yours to replace)
- `configs/` — app.yaml (all D-knobs), data.yaml, train_*.yaml, impl.yaml
- `scripts/` — run_app, train, eval_verb_model, latency_bench, verb_counts, select_videos, download_subset, downscale, bench_loader, annotation_check, record_ingredient_frames, eval_matcher, record_verb_clips, eval_on_webcam_clips, eval_smoothers, eval_loop, gen_equivalence_vectors, check_all
- `tests/` — 102 tests (unit, equivalence with shared vectors, app smoke)
- `runs/`, `reports/` — logged outputs; every number's provenance is in the file
- `kaggle/` — kernel.py + metadata (the training offload)
- `STATUS.md` (running log), `DECISIONS_NEEDED.md` (D1-D8 + S1/S2 + V1), `BLOCKERS.md`, `HANDOFF.md` (this file)

## 4. Numbers (command · date · commit · seed, or NOT RUN)

All produced on this machine unless marked Kaggle. Commit = git hash at
the time of the run; seed 42 where a run is seeded.

- Verb counts (train): take 3412, put 2106, wash 2078, cut 830, open 4851,
  close 3463, mix 212, pour 655 — all 8 many-shot (>100).
  `python -m scripts.verb_counts` · 2026-10-23 · b8006ab · reports/verb_counts.md
- Data speeds: official data.bris 182 KB/s sustained (rejected); HF mirror
  3.09 MB/s (selected). reports/data_access.md · 2026-10-23 · b8006ab.
- New webcam: 19.4 FPS at 640x480 AND 1280x720 MJPG (156 frames / 8 s,
  both). reports/camera_check.md · 2026-10-23 · b8006ab.
- Annotation convention: 5/5 ffmpeg cuts from P01_107 match inclusive
  stop_frame EXACTLY (92, 279, 39, 36, 41 frames). reports/annotation_check/
  · 2026-10-23 · 9384baf (subagent WP1b).
- Downscale: P01_107 362 MB -> 10 MB, 5972 frames preserved, fps 50 kept.
  reports/downscale_check.md · 2026-10-23.
- Loader (D8 evidence): decode-on-the-fly 1.94 clips/s vs pre-extracted
  JPEG 198.45 clips/s; 0.33 MB/clip extraction cost. Single process,
  download concurrent (labelled estimate). reports/loader_benchmark.md.
- CPU latency (NOT the 4060): resnet34 tsn_avg T=8 fp32 29.75 ms mean;
  mobilenet_v2 14.48 ms; resnet18 16.60 ms. `python -m
  scripts.latency_bench --device cpu --img-size 96` · 2026-10-23 · e45e7a1.
  **RTX 4060 table: NOT RUN** (run `--device cuda --img-size 224`).
- Training on real EPIC subset: **local run, stopped at owner request**
  2026-10-24 20:12 UTC after epoch 7 of 12 (~26 min/epoch, fast_seek +
  6 workers on the 4060). best.pt = epoch 1, val_mean_class_acc
  **0.2205** (majority-class 0.189); val mca declined every epoch after
  (0.221 → 0.213 → 0.209 → 0.171 …) — classic overfit on 2402 segments.
  Full log: runs/baseline/log.csv. The parallel Kaggle kernel was
  cancelled externally at ~7.75 h execution; its epoch-0 checkpoint +
  overfit-32 gate (mca 1.0) were harvested to runs/kaggle-baseline-partial/.
  Seed 42 throughout.
- fast_seek decode audit: `CAP_PROP_POS_FRAMES` seek vs sequential grab
  — **pixel-exact 40/40 sampled frames** on our EPIC-256 re-encodes,
  **21.5× faster** end-to-end; unblocked 26-min epochs (was 92).
  reports/downscale_check.md + STATUS.md 2026-10-24.
- Smoother bake-off (D6, offline on 593 val segments, min_conf 0.5):
  raw coverage 0.2024 / fire-precision 0.2535; **majority coverage
  0.5683 / fire-precision 0.2125**; ema & hysteresis collapse
  (coverage 0.0236, 5 fires each). Majority wins on coverage — the
  demo needs ticks, and precision is within noise at this model quality.
  `python3 -m scripts.eval_smoothers` · 2026-10-25 · reports/smoother_eval.md.
- Reasoning-chain eval (10 val videos as ≤15-step pseudo-recipes,
  oracle = true verbs): strict policy advances only 0.336 even for
  ORACLE — long recipes + strict ordering are a protocol artifact.
  **Optimistic policy (later-step match advances, "assumed done
  off-screen"): oracle advance-rate 0.748, 6/10 recipes completed.**
  With the real model (majority smoother + recipe prior, leak 0.05):
  advance-rate 0.133 vs 0.0 strict without prior (~9×). The prior only
  helps under optimistic tracking (under strict it starves the machine
  of legal fires). `python3 -m scripts.eval_loop` · 2026-10-25 ·
  reports/loop_eval.md.
- **Recommended demo config (measured, needs owner sign-off D6/D9/D10):**
  smoother=majority, policy=optimistic, recipe_prior=true (leak 0.05).
- Matcher eval, webcam verb clips, Tier 1 live recording: NOT RUN
  (need you awake / daylight / webcam pointed at ingredients).
- GPU status at handoff update: **free** (8 % util, no trainer) — owner
  needs it; a proposed ~2 h / 4-epoch regularized retrain (dropout +
  stronger augmentation, aiming to beat 0.2205) is queued as item 5 and
  will NOT start without explicit go-ahead. Camera still detached since
  the WSL crash; re-attach needs owner: admin PowerShell
  `usbipd attach --wsl --busid 4-10`.

## 5. Open decisions + blockers

- DECISIONS_NEEDED.md: D1-D10 (evidence attached where gathered: S1
  verified, D8 benchmarked, D6 smoother bake-off measured, D9/D10
  policy + prior measured via eval_loop oracle rows), plus V1 (internal
  validation split — needs your sign-off) and S2 (final recipe).
- BLOCKERS.md: no open blockers. WSL crashed once at 23:05 (nothing lost);
  if it recurs overnight, the download/downscale resume on their own.

## 6. Known issues and risks

- The UI was verified headlessly (events/state/HTTP), never in a real
  browser — pixel layout/countdown ring are unverified. 10-minute check
  with coffee.
- `--mode clip/live` need a checkpoint; until the Kaggle run lands they
  exit with a clear message (by design).
- Validation split (P22/P30) videos download last; any local run before
  they arrive warns "validation split is empty" (honest, not an error).
- EPIC-256 corpus lives on /mnt/ssd; C: has only 78 GB free — never point
  videos_root at the WSL home.
- DINOv2 weights (~84 MB) download on first scan-phase use; the engine
  pre-warms off-thread now, but the very first session start on a cold
  cache still takes ~30 s before matching begins.
- The toy embedder (default in tests) accepts nearly everything as known
  — matcher_threshold sweeps in reports/matcher_eval.md will look bad on
  purpose until you run DINOv2 on real frames.

## 7. Study queue

docs/STUDY_QUEUE.md — 11 items in order, each with the reference file,
the worked example, and the external link; the five core pieces to
rewrite in core/ are marked (metrics -> sampler -> enrollment ->
attention -> state machine).

## 8. Verification log

- **Verifier sub-agent pass, 2026-10-24 ~01:35 IST — ALL 10 CHECKS PASS:**
  1. Full suite 85 passed in 20.4s (79 unit + 6 equivalence).
  2. Equivalence harness replays the shared vectors for all five pieces.
  3. EGO_IMPL=core fails with the intended actionable message; core/ absent.
  4. Regenerated equivalence vectors byte-identical (deterministic).
  5. App boot: /api/state ("Example salad"), MJPEG stream 200 + boundary,
     UI served with app.js; smoke 3/3.
  6. All worked-example numbers reproduce EXACTLY: kNN distances,
     softmax loss 1.0679, sampler [105..176], metrics 0.6667/0.7778/0.5,
     BatchNorm running_var [1.5667, 1.0333], state-machine trace
     [wrong_order, advance, repeat, idle, advance, advance, done, idle],
     matcher cosine 0.8937.
  7. Reports honest: verb counts recompute exactly from official CSVs
     (open 4851, mix 212, take 3412...); RTX 4060 latency and webcam eval
     explicitly NOT RUN; downscale frame counts re-verified via ffprobe.
  8. `git ls-files core/` empty; all 5 stubs raise NotImplementedError.
  9. Bad recipe rejected loudly naming the offending step.
  10. check_all.sh exit 0, "ALL CHECKS PASSED" end to end.
- Also independently reproduced during the build: real-data dry-run
  training (6 segments, 1 epoch, CPU, 2026-10-24, e45e7a1); Kaggle kernel
  accepted and RUNNING (2026-10-24 01:10 IST).
- Not verified (out of scope / deferred): Kaggle run contents until it
  finishes; loader-benchmark throughput re-measurement (caveats labelled
  in the report); anything needing the real webcam or the 4060.
- **2026-10-25 final gate (reasoning-layer session):** full suite
  **102 passed** (fresh re-run, 17.9 s); `bash scripts/check_all.sh`
  exit 0 "ALL CHECKS PASSED"; eval caches + reports committed
  (reports/smoother_eval.md, reports/loop_eval.md); GPU left free.

---
*Append below as things complete overnight.*
