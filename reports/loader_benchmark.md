# Loader benchmark (D8 evidence: decode-on-the-fly vs pre-extracted JPEGs)

- date: 2026-10-23 23:02
- command: python -m scripts.bench_loader --clips 24 --videos (all complete) --T 8
- root: /mnt/ssd/epic-kitchens/EPIC-256 (downscaled (EPIC-256))
- videos used: P01_102, P01_103, P01_104, P01_107, P01_108, P02_101, P02_102, P02_103, P02_107
- segments measured: 24 (candidate verbs ['take', 'put', 'wash', 'cut', 'open', 'close', 'mix', 'pour']), T=8, single process (dataset[i] loop, num_workers=0), 1 warmup call excluded
- caveat: numbers reflect this machine's load at run time (a concurrent bulk download was active when this benchmark was first run; CPU+disk contention inflates both modes).

| mode | clips ok | clips/sec | disk MB/s | disk cost/clip |
|---|---|---|---|---|
| (a) decode-on-the-fly (downscaled) | 24/24 | 1.94 | 22.5 (ESTIMATE, see note) | 0 extra |
| (b) pre-extracted JPEGs (reload only) | 24/24 | 198.45 | 64.5 (measured jpg bytes read) | 0.33 MB |

- pre-extraction one-time cost: 12.9 s for 24 clips, 7.8 MB total (0.33 MB/clip, i.e. 0.33 GB per 1000 clips)
- decode-mode MB/s is an ESTIMATE: sequential decode reads the file from byte 0 up to the last needed frame, so bytes ~= file_size * max_index / total_frames (279 MB touched over 24 clips); it is not instrumented with iostat.
