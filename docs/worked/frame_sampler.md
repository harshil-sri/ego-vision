# Worked example: frame sampling (`reference/frame_sampler_ref.py`)

A video annotated from frame 100 to frame 180 (inclusive — decision S1 in
DECISIONS_NEEDED.md), T = 8, fps = 59.94.

## Segment-centre sampling (the TSN convention)

Clip length (inclusive stop): 180 − 100 + 1 = **81 frames**. Split into
T = 8 segments of 81/8 = **10.125 frames** each. Take each segment's
CENTRE: offset (i + 0.5) · 10.125, round to nearest frame, clip into
[0, 80], add start:

| segment i | centre offset | rint | frame index |
|---|---|---|---|
| 0 | 5.0625 | 5 | **105** |
| 1 | 15.1875 | 15 | **115** |
| 2 | 25.3125 | 25 | **125** |
| 3 | 35.4375 | 35 | **135** |
| 4 | 45.5625 | 46 | **146** |
| 5 | 55.6875 | 56 | **156** |
| 6 | 65.8125 | 66 | **166** |
| 7 | 75.9375 | 76 | **176** |

`sample_indices(100, 180, 8, 59.94)` →
`[105, 115, 125, 135, 146, 156, 166, 176]` ✓ (np.rint rounds .5 to even —
45.5625 and 75.9375 are safely away from halves; know this if you ever
hit an exact .5).

With the EXCLUSIVE convention (stop not included): length 80, segments
of 10.0, centres at (i+0.5)·10 → offsets 5, 15, ..., 75 →
`[105, 115, 125, 135, 145, 155, 165, 175]`. The one-frame difference
matters only at the boundary — but WHICH convention you use must be the
SAME in training and live inference, or every clip is misaligned by up to
half a segment.

## T larger than the clip: frames 0..2, T = 5

Length 3, segment width 0.6, centres 0.3, 0.9, 1.5, 2.1, 2.7 →
rint [0, 1, 2, 2, 3] → clip to [0, 2] → **[0, 1, 2, 2, 2]**: exactly T
values, sorted, never outside the clip, endpoints reused where needed.
Why not error: the LIVE pipeline samples the last ~2 s window even when
the buffer is short (startup, lag spikes) — the model still sees T frames.

## Single-frame clip: frames 50..50, T = 4

Length 1 → all centres round into [0, 0] → [50, 50, 50, 50]. The one
frame is repeated T times; a T-aggregation model (TSN average, attention)
sees a frozen clip.

## What each row and column means

Output is a (T,) int64 VECTOR of frame indices into the video file —
"which decoded frames make up this clip". Index t is the t-th of T
temporal positions; the VALUES are frame numbers in the video's own
indexing. `fps` is NOT used by the arithmetic (indices live in frame
space) but stays in the signature because timestamps ↔ frames conversion
(fps · seconds) belongs beside it at every call site.

## Three common mistakes

1. **Convention drift**: training on inclusive-stop indices, live code
   using exclusive → every window shifted by one frame; small accuracy
   loss, maddening to find. The parameter exists to make the choice
   explicit and testable.
2. **`np.linspace(start, stop, T)` (evenly spaced INCLUDING endpoints)**
   instead of segment centres: different but ALSO reasonable — the bug is
   mixing the two between train and inference, not either choice itself.
3. **Integer division of the length**: 81//8 = 10 with remainder 1 —
   a loop that advances by 10 frames ends at 70, never reaching frame
   176; the float-width + rint approach distributes the remainder.

## Edge cases

- start == stop (inclusive): length 1, repeated index (above).
- stop < start: raises ValueError immediately (annotations occasionally
  contain such rows — fail loudly, don't silently emit empty clips).
- fps from the wrong source: EPIC videos are 59.94 OR 50.0 (checked in
  reports/data_access.md) — always read per-video fps from
  EPIC_100_video_info.csv / ffprobe, never a global constant.
