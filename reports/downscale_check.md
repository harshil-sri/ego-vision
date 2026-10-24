# Downscale verification (first video of each run)

Frame count via ffprobe nb_frames header (fallback
-count_frames). Requirement: fps and frame count identical to
the original so annotation indices stay valid.

| date | video | crf | orig fps | new fps | orig frames | new frames | orig MB | new MB | encode s | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 22:58 | P01_107 | 23 | 50/1 | 50/1 | 5972 | 5972 | 362 | 10 | 33 | OK |
| 2026-10-23 23:03 | P02_133 | 23 | 50/1 | 50/1 | 4022 | 4022 | 244 | 6 | 22 | OK |
| 2026-10-23 23:05 | P03_114 | 23 | 50/1 | 50/1 | 1382 | 1382 | 84 | 3 | 8 | OK |
| 2026-10-23 23:15 | P07_106 | 23 | 50/1 | 50/1 | 810 | 810 | 48 | 1 | 5 | OK |
| 2026-10-23 23:27 | P07_112 | 23 | 50/1 | 50/1 | 4367 | 4367 | 264 | 7 | 27 | OK |
| 2026-10-23 23:39 | P09_106 | 23 | 50/1 | 50/1 | 7468 | 7468 | 453 | 13 | 46 | OK |
| 2026-10-23 23:51 | P26_106 | 23 | 50/1 | 50/1 | 4424 | 4424 | 268 | 6 | 14 | OK |
| 2026-10-24 00:01 | P26_113 | 23 | 50/1 | 50/1 | 3444 | 3444 | 209 | 4 | 19 | OK |
