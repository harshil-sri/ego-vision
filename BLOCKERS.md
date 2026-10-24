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
