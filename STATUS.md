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

## Current state

- RUNNING: bulk HF download (55 videos / 63.7 GB).
- NEXT: WP2 reference library; WP5 pipeline + app; reports/commit cadence
  every work package.
