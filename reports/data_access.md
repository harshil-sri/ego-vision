# EPIC-KITCHENS-100 data access report (WP0)

Date: 2026-10-23. Commit: b8006ab (branch glm-window). All numbers below were
produced by the commands shown, run on this machine.

## Disk reality check

`df -h /` inside WSL reports 696 GB free, but that is the virtual disk ceiling.
The real space is on Windows drives:

| Mount | Device | Size | Used | Free | Notes |
|---|---|---:|---:|---:|---|
| /mnt/c | C: (9p/drvfs) | 931G | 853G | **78G** | WSL VHDX lives here; 92% full. No data here. |
| /mnt/ssd | /dev/sde ext4 | 234G | 74G | **149G** | Kingston SSD, native ext4, writable. All EPIC data goes here. |

Data root: `/mnt/ssd/epic-kitchens/EPIC-KITCHENS` (same `<Pxx>/videos/<id>.MP4`
layout the official downloader produces). Budget cap: 80 GB (inside the brief's
60%-of-free rule: 0.60 * 149 = 89 GB).

## Route 1: official downloader (data.bris.ac.uk) — REJECTED for speed

```
cd /mnt/ssd/epic-tools/epic-kitchens-download-scripts
python3 epic_downloader.py --videos --specific-videos P01_01 --output-path /mnt/ssd/epic-kitchens
```

- Connected and downloaded without errors (no connection reset from this network).
- P01_01 total size reported by the server: **5929.66 MB**.
- Sustained speed measured over 45 s via file size delta: **182 KB/s**
  (15 MB after ~2 min). At that speed one video = ~9 hours; the whole subset
  would take weeks. Stopped; partial file deleted.

## Route 2: Hugging Face mirror `awsaf49/epic_kitchens_100` — SELECTED

- 30 s ranged download of `P01/videos/P01_101.MP4`:
  **3,091,797 B/s** (32 MB in 10.9 s) — ~17x the official route, saturating
  the 30 Mbps WiFi line.
- Layout: `<participant>/videos/<video>.MP4`; **268 MP4 files, 340.8 GB total**,
  13 participants (P01..P37).
- **Coverage limitation (important):** the mirror contains **only train-split
  videos** (201 of them appear in `EPIC_100_train.csv`; none in
  `EPIC_100_validation.csv`). Official validation videos are only reachable via
  the rejected-slow official route. Consequence and handling: see
  DECISIONS_NEEDED.md ("internal validation split").
- `lightly-ai/epic-kitchens-100-clips` (24 GB pre-cut clips) NOT yet checked;
  listed as a possible validation-data source later.

## What is downloading now

```
nohup python3 -m scripts.download_subset \
    --selected reports/selected_videos.csv \
    --repo awsaf49/epic_kitchens_100 \
    --out /mnt/ssd/epic-kitchens/EPIC-KITCHENS --max-workers 4 &
```

55 train videos, 63.7 GB, selected by `scripts/select_videos.py` (greedy by
candidate-verb instances per GB). Log: `/mnt/ssd/epic-kitchens/logs/hf_download.log`.
Resumable: re-running the same command skips completed files.

## Annotations

Cloned to `/mnt/ssd/epic-tools/epic-kitchens-100-annotations` (depth 1).
`EPIC_100_train.csv` columns confirmed (15): narration_id, participant_id,
video_id, narration_timestamp, start_timestamp, stop_timestamp, start_frame,
stop_frame, narration, verb, verb_class, noun, noun_class, all_nouns,
all_noun_classes. `EPIC_100_video_info.csv` gives per-video fps/duration/
resolution (e.g. P01_01: 59.94 fps; P01_101: 50.0 fps — fps varies per video,
must be read per video, confirming the brief's gotcha).
