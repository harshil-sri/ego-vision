# BLOCKERS

Rule: anything blocked for more than 15 minutes gets an entry here, then work
switches to another track. Empty below = no open blockers.

| Date | Blocker | Tried | Need | Status |
|---|---|---|---|---|
| 2026-10-23 | Official EPIC route too slow (182 KB/s) for any video | 45 s sustained measurement during live download | Nothing — resolved by switching to HF mirror | RESOLVED |

## 2026-10-24 ~13:20 IST — Kaggle baseline kernel CANCELLED mid-run (external)
`kaggle kernels status` flipped RUNNING -> CANCEL_ACKNOWLEDGED at ~7.75 h of
execution (started 00:00:22 UTC; execution budget was 12 h). None of our
commands cancel kernels (only status/list/output probes ran), so this was
external: account owner action, quota enforcement, or Kaggle-side. Asked
owner whether Tanish cancelled it. Output bundle still downloadable —
harvest in progress to recover any partial runs/. Fallback if nothing
usable: finish the 12-epoch baseline LOCALLY on the 4060 (all 55 videos +
annotations are on /mnt/ssd) — no further Kaggle quota needed for baseline.

## 2026-10-24 13:40 UTC — local 12-epoch baseline crashed on truncated EPIC-256 file
First local run died ~10 min in: P23_102.MP4 downscaled copy has 158,482
frames but annotations need 178,282. Root cause: last night's interrupted
final downscale pass left this file truncated; reports/downscale_check.md has
NO row for P23_102 (never verified), and the "55 valid" gate checked
playability, not annotation-parity for it. Audit of all 55 (container
nb_frames vs max stop_frame): P23_102 was the ONLY short file. Original is
complete (178,476 frames) -> re-downscaling locally and relaunching.
Correction for the record: the 12:53 UTC "training healthy" note was wrong —
the process had already crashed and the GPU % shown was Windows-side usage.
