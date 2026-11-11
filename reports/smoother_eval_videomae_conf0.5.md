# Smoother bake-off (D6) — offline on real val segments

Model: `runs\videomae_mlp` over 1715 val segments (10 videos), per-video replay in annotation order,
min_conf=0.5. Fires = stabilized verb edges (what the engine actually consumes). tick_acc counts 'no confident verb' as a miss; tick_mca_conf is over confident ticks only.

| smoother | coverage | tick acc | tick mca (confident) | fires | fire precision |
|---|---|---|---|---|---|
| raw | 0.7574 | 0.5761 | 0.7307 | 991 | 0.7881 |
| ema | 0.137 | 0.0985 | 0.7069 | 76 | 0.7763 |
| majority | 0.4344 | 0.1988 | 0.3334 | 178 | 0.6236 |
| hysteresis | 0.137 | 0.0985 | 0.7069 | 76 | 0.7763 |
