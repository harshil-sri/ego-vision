# Downscale verification (first video of each run)

Frame count via ffprobe nb_frames header (fallback
-count_frames). Requirement: fps and frame count identical to
the original so annotation indices stay valid.

| date | video | crf | orig fps | new fps | orig frames | new frames | orig MB | new MB | encode s | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 22:58 | P01_107 | 23 | 50/1 | 50/1 | 5972 | 5972 | 362 | 10 | 33 | OK |
