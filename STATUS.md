# STATUS

Working log for the GLM window (branch `glm-window`). Newest entries at the
bottom. Every number links to a command that was actually run on this machine;
anything not run is marked NOT RUN.

## Setup facts (2026-10-23)

- Runtime: WSL2, Python 3.12.3 at `~/.venvs/global/bin/python3`, torch
  2.6.0+cu124, CUDA works on RTX 4060 8GB (`torch.randn(8,8,device='cuda')`
  matmul OK).
- ffmpeg: apt 6.1.1 at /usr/bin/ffmpeg (installed by owner) + static build at
  ~/.local/bin/ffmpeg; ffprobe available both ways; aria2c 1.37 at /usr/bin.
- Disk: data on /mnt/ssd (149 GB free, ext4). C: has only 78 GB free — never
  used for data. See reports/data_access.md.
- New webcam: verified, ~19.4 FPS at 640x480 and 1280x720 MJPG
  (reports/camera_check.md). Camera released after test; nothing holds
  /dev/video0 during the night.
- Kaggle CLI works (account harshilsrivastav,
  KAGGLE_CONFIG_DIR=/home/harshil/.kaggle-accounts/mrigankrana). GPU policy:
  local GPU shared (<=4 GB, short runs); big training -> Kaggle.

## Log

- 2026-10-23 22:12 — branch `glm-window` created; existing camera layer +
  docs committed (b8006ab). Repo skeleton dirs created.
- 2026-10-23 22:20 — official EPIC downloader started for P01_01; sustained
  182 KB/s over 45 s -> route rejected. (reports/data_access.md)
- 2026-10-23 22:24 — HF mirror probe: 3.09 MB/s over 10.9 s (32 MB).
  Mirror holds 201 train-split videos, 340.8 GB; no validation videos.
- 2026-10-23 22:33 — verb counts computed from official CSVs
  (`python -m scripts.verb_counts --annotations /mnt/ssd/epic-tools/epic-kitchens-100-annotations`)
  -> reports/verb_counts.md. All 8 candidate verbs > 100 train instances
  (min: mix 212, pour 655; max: open 4851). No verb-list change needed.
- 2026-10-23 22:38 — subset selected: 55 train videos, 63.7 GB,
  2,995 candidate-verb train instances
  (`python -m scripts.select_videos --budget-gb 75 ...` ->
  reports/selected_videos.csv). Bulk download running in background
  (pid 156290, log /mnt/ssd/epic-kitchens/logs/hf_download.log).
- 2026-10-23 22:40 — new webcam measured: 19.4 FPS @ 640x480 and 720p MJPG.

## Current state (updated 2026-10-24 00:00 IST)

- RUNNING: bulk HF download (55 videos / 63.7 GB; ~14 GB done, ~2.8 MB/s);
  downscale watcher transcodes completions to /mnt/ssd/epic-kitchens/EPIC-256
  every 10 min (289 MB so far). Kaggle GPU policy: train on Tanish's account
  (mahajantanish1) via KAGGLE_API_TOKEN; NO local GPU until the owner wakes.
- Subagents in flight: WP3 models/training/eval (files landing, tests running);
  WP5 static web UI (src/app/static/ being written).

## Log (continued)

- 2026-10-23 23:00 — WP2 COMPLETE: 5 reference pieces + 3 exercises
  (knn/softmax/batchnorm, loops->vectorized, gradient check, unbiased-var
  BN subtlety) + 5 owner stubs + equivalence vectors/harness (6 tests) +
  8 worked examples in docs/worked/ with every number verified against the
  code. All tests green at commit time.
- 2026-10-23 23:05 — WSL crash (user report): download process survived,
  nothing lost; hourly wake-up automation re-verified.
- 2026-10-23 23:20 — WP1b subagent DONE: src/data (annotations, splits w/
  V1 internal validation, video_io, EpicVerbDataset, synthetic) + scripts
  (downscale idempotent, bench_loader, annotation_check) + 31 tests pass.
  REAL-DATA: 5/5 annotation cuts match inclusive stop_frame EXACTLY;
  downscale P01_107 362MB->10MB, frames 5972->5972 preserved; loader bench
  decode 1.94 clips/s vs pre-extracted 198.45 clips/s (reports/loader_benchmark.md).
- 2026-10-23 23:40 — WP5 backend DONE (mine): engine (scan/cooking phases,
  event bus, D6 smoothers, idle nudge, wrong-order banner), overlay,
  embedders (toy/dinov2/resnet/clip), verb sources, FastAPI (MJPEG + WS +
  REST), run_app CLI (keyboard/clip/live x web/cv, --synthetic).
  Tier 1 smoke test 3/3 on synthetic frames (scan->wrong-order->full
  recipe->done, session jsonl verified). Server boot verified: /api/state,
  /video/stream OK.
- 2026-10-24 00:00 — WP4 (matcher eval scripts verified on synthetic
  frames), WP6 docs (ARCHITECTURE/GLOSSARY/STUDY_QUEUE), WP7 packaging
  (requirements pinned, Makefile, check_all.sh, README). check_all full
  run deferred until WP3 subagent's files settle (two pytest processes
  collided; unit+equivalence suites green separately at their commits).

- 2026-10-24 00:55 — WP3 subagent DONE: src/models (backbones resnet18/34/
  mobilenet_v2, variants single_frame/tsn_avg/temporal_attn w/ registry
  attention, eval-mode predict_probs), src/train (strict config loader,
  loop w/ AdamW|SGD+cosine|step+AMP+class-weights+grad-clip+seeds,
  overfit-32 gate >=0.99, log.csv, best-by-val-mean-class-acc, verb_map.json),
  eval w/ majority baseline beside every number, latency bench (CPU table
  measured; 4060 NOT RUN until morning), webcam eval tooling. Real bug
  found+fixed: infinite iteration on map-style Dataset. Full suite 85/85.
- 2026-10-24 01:00 — real-data CPU dry-run OK: train --train-limit 6
  --epochs 1 on EPIC-256 videos (6 segments decoded, 1 epoch, checkpoint
  saved). Val empty until P22/P30 videos finish downloading (they are in
  the selection: 5+5 videos, 593 candidate instances).
- 2026-10-24 01:05 — UI subagent DONE: static web app (scan wizard w/
  countdown+thumbs+retake, recipe timeline, debug drawer, keyboard verbs,
  presentation mode, help overlay, WS reconnect states) verified headless
  (events, state transitions, MJPEG). Engine embedder pre-warm added.
- 2026-10-24 01:10 — KAGGLE KERNEL PUSHED + RUNNING:
  mahajantanish1/ego-vision-baseline (Tanish account, GPU, internet).
  Flow: clone glm-window -> annotations -> stream 55 videos from HF mirror
  -> downscale in-batch -> overfit-32 gate on real clips -> baseline train
  (12 epochs) -> outputs in /kaggle/working/runs. Expected ~2h.
- 2026-10-24 02:35 — wake-up check: download 40/63.7 GB alive; watcher
  alive; Kaggle baseline kernel still RUNNING (~2.5h). CV UI probed under
  WSLg: this venv has opencv-python-headless shadowing opencv-python, so
  imshow is unimplemented -> run_app --ui cv now exits with a clear fix
  message instead of crashing. Fix (morning, owner decision — shared
  global venv): pip uninstall opencv-python-headless. Web UI unaffected.
- 2026-10-24 03:35 — wake-up: download ACTIVE at ~2.1 MB/s measured over
  30 s (du rounding made it look stalled; 38/55 files done, tqdm ETA
  ~4h20m — the late-selection files are larger). Kaggle baseline kernel
  still RUNNING (~3.5h). Watcher alive.
- 2026-10-24 04:35 — wake-up: download healthy — 3.4 MB/s over a 90 s
  window, 6 files in parallel; P22 (heldout) videos now arriving, which
  will make the local validation split non-empty. Kaggle baseline still
  RUNNING (~4.5h).
- 2026-10-24 05:35 — wake-up: download 49.5/63.7 GB (42/55, ~2h left);
  Kaggle baseline still RUNNING (~5.5h; T4 + on-the-fly decode is the
  long pole, within the 12h limit). scripts/harvest_kaggle.py added:
  one-command kernel harvest with anti-fabrication guards.
- 2026-10-24 07:05 — DOWNLOAD COMPLETE: 55/55 videos, 64.8 GB, 5h07m wall
  (avg ~3.5 MB/s). All 55 downscaled copies VALID (ffprobe-checked) at
  /mnt/ssd/epic-kitchens/EPIC-256 (2.0 GB). FINAL SPLIT on the full corpus
  (make_splits, heldout P22+P30): train 2402 segments / 45 videos; val 593
  segments / 10 videos. Subset verb counts (train): take 722, put 604,
  open 411, close 330, wash 182, cut 84, pour 39, mix 30 — NOTE cut/pour/
  mix are under 100 ON THIS SUBSET (many-shot held on the full dataset
  only); class_weights default stays on; per-class numbers for mix (30
  train, 0 val) will be noisy/absent. Kaggle baseline still RUNNING (~8h).

## 2026-10-24 ~06:50 IST — Kaggle timeline correction (evidence)
`kaggle kernels list -m` shows lastRunTime `2026-10-24 00:00:22` (UTC = 05:30:22
IST) for mahajantanish1/ego-vision-baseline. Push was ~18:15 IST Oct 23 → the
kernel spent ~11 h in the worker queue before execution began; it is ~1.25 h
into its 12 h execution budget (NOT over any limit). Output probe mid-run
returns empty as expected. Local-4060 validation of the same train path
(1 epoch, 64 segments) queued as the pre-push gate for any future kernel.

## 2026-10-24 06:40 IST — local-4060 pre-push gate PASSED (evidence)
`python3 -m scripts.train --config configs/train_baseline.yaml --device cuda
--epochs 1 --train-limit 64 --out-dir runs/local_gpu_sanity` — exit 0 in ~10 min.
train 64 segs/45 videos, val 593 segs/10 videos. epoch 0: train_loss 2.3720,
val_acc 0.1720, val_mca 0.1895 (majority baseline 0.3676 — 1 epoch on 64
segs is a pipeline check, not a quality number). Artifacts written:
best.pt/last.pt, config.yaml, log.csv, metrics.json, verb_map.json.
The exact train path any future Kaggle push uses is verified working.

## 2026-10-24 ~09:05 IST — variant pre-push gates PASSED (local 4060)
Same gate as baseline (1 epoch, 64 train segs, full 593-seg val, cuda):
- tsn_avg        exit 0: train_loss 2.3463, val_acc 0.2445, val_mca 0.2199
- temporal_attn  exit 0: train_loss 2.3423, val_acc 0.2479, val_mca 0.2221
(1ep/64segs = pipeline checks vs 0.3676 majority baseline, not quality numbers.)
Artifacts complete in runs/local_gate_{tsn,attn}/. Variant kernels are
cleared to push once the Kaggle baseline finishes (no parallel quota burn);
the kernel reads its variant from kaggle/variant.txt.

## 2026-10-24 ~10:10 IST — final verifier pass over post-fix state
`bash scripts/check_all.sh` → exit 0, ALL CHECKS PASSED (tests, registry
switch, recipe validation, headless app boot) after the morning's engine/
server/UI changes (session lifecycle, graceful shutdown, sampled-frame strip).

## 2026-10-24 ~10:40 IST — Tier 2 (clip mode) VERIFIED end-to-end + 2 bug fixes
First real Tier-2 run (`run_app --mode clip --clip EPIC-256/P22/videos/P22_104.MP4
--run-dir runs/local_gpu_sanity`) crashed twice — both bugs now fixed + tested:
1. model_source loaded state["model"] but train saves "model_state_dict"
   → loader now reads the train format (and prefers ckpt backbone/variant).
2. predict() permuted the stacked (T,S,S,3) window as a single frame →
   RuntimeError at the first model tick → tensor path rewritten to the
   dataset contract (RGB (1,T,3,S,S) [0,1]); engine loop now catches crashes
   and ends the session with an error event instead of freezing silently.
Verified run: scans → enroll → cooking; verb_predicted take@0.513 via ema
smoother; inference 155.7 ms/tick CPU (fps 51.7); events: 96 scan_progress,
16 ingredient_matched, 5 window_sample, 1 verb_predicted, idle_nudge.
New tests/test_model_source.py (3 tests). Suite: 92 passed (89 + 3; an earlier draft of this entry said 93 — corrected to match the actual pytest output).

## 2026-10-24 ~11:30 IST — Tier 3 (live mode) wiring VERIFIED (no camera)
`run_app --mode live --synthetic --run-dir runs/local_gpu_sanity` (port 8003):
scans → cooking; model ticks 331.9 ms/inference CPU (app.yaml device=cpu);
smoother withheld sub-threshold predictions on synthetic frames (correct —
the 1-epoch model has never seen procedural patterns); 96 scan_progress,
5 ingredient_matched, 7 window_sample logged. Live-camera Tier-3 run is the
same code path + webcam; awaiting trained checkpoint + owner.

## 2026-10-24 ~14:45 IST — Kaggle harvest (cancelled run) + local 4060 baseline started
Harvested the cancelled kernel's output (real numbers from the files):
- overfit-32 gate on Kaggle: PASSED — best_epoch 4, val_mean_class_acc 1.0000
  (majority_baseline_val 0.53125); gate ended 18704 s (~5.2 h) after run
  start — video download+downscale consumed most of that.
- baseline: exactly ONE epoch completed before the external cancel —
  epoch 0: train_loss 1.6115, train_acc 0.3497, val_loss 1.8067,
  val_acc 0.3575, val_mean_class_acc 0.2389. best.pt/last.pt recovered to
  runs/kaggle-baseline-partial/; small artifacts in reports/kaggle/.
- Full 12-epoch baseline RESTARTED locally on the RTX 4060 (~14:45 IST,
  runs/baseline, log /tmp/train_baseline_local.log): same config the
  kernel used, all 55 videos + annotations from /mnt/ssd. Zero further
  Kaggle quota consumed.

## 2026-10-24 12:55 (machine clock) — timezone labeling correction
This machine's clock is UTC (Etc/UTC), not IST. All "~..IST" labels written
in this file and HANDOFF.md during 2026-10-23/24 are actually machine-local
UTC; owner-local IST = label + 5:30. Kaggle CLI timestamps quoted as UTC
were already correct. Metric numbers are unaffected. Future entries use
"UTC" labels explicitly.

## 2026-10-24 13:55 UTC — P23_102 re-encoded; local 12-epoch baseline relaunched
Re-downscale: 698 s, 321 MB, ffprobe 178,476 frames = exact parity with the
original (annotations need <= 178,282). Row appended to downscale_check.md.
Training relaunched (~13:55 UTC, runs/baseline): 2402 train / 593 val segs,
process 313865 on cuda. Prior run died on the truncated file ~10 min in;
audit says this was the only short file of 55.

## 2026-10-24 15:30 UTC — baseline restarted with 6 dataloader workers
First relaunch ran 92 min without finishing epoch 0: process averaged exactly
1 core (92:59 CPU-min) with idle gaps -> serial decode/IO bottleneck at the
default --workers 2. Machine has 16 cores; restarted with --workers 6
(training math unchanged: same seed/shuffle/batch; Kaggle kernel used 4).
Expect ~25-40 min/epoch vs 90+; will verify against epoch-0 row timing.

## 2026-10-24 ~17:10 UTC — decode bottleneck found+fixed (fast_seek), WSL crash recovery
Root cause of 90-min epochs: sequential decode grab()s from frame 0 to the
target — a late segment pays ~100k+ grabs (documented cost model in
video_io.py). Audit on OUR EPIC-256 re-encodes: fast_seek (CAP_PROP_POS_FRAMES)
is PIXEL-EXACT on 40/40 sampled late segments and 21.5x faster (368.4s ->
17.2s for the same decodes). Knob plumbed: dataset fast_seek param + train
config fast_seek (default False; true in the 3 train configs + kernel cfg).
Second WSL crash of the session ~17:05 UTC killed trainer/app/pollers (files
intact, nothing lost — epoch 0 had not checkpointed). Camera detached by the
restart: needs usbipd re-attach from Windows. Training relaunched with
fast_seek after the crash.
