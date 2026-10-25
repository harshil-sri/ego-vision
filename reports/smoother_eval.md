# Smoother bake-off (D6) — offline on real val segments

Model: `runs/baseline` over 593 val segments (10 videos), per-video replay in annotation order,
min_conf=0.5. Fires = stabilized verb edges (what the engine actually consumes). tick_acc counts 'no confident verb' as a miss; tick_mca_conf is over confident ticks only.

| smoother | coverage | tick acc | tick mca (confident) | fires | fire precision |
|---|---|---|---|---|---|
| raw | 0.2024 | 0.0489 | 0.2497 | 71 | 0.2535 |
| ema | 0.0236 | 0.0017 | 0.25 | 5 | 0.0 |
| majority | 0.5683 | 0.1164 | 0.1733 | 80 | 0.2125 |
| hysteresis | 0.0236 | 0.0017 | 0.25 | 5 | 0.0 |
