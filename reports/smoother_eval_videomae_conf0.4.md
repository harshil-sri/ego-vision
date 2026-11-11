# Smoother bake-off (D6) — offline on real val segments

Model: `runs\videomae_mlp` over 1715 val segments (10 videos), per-video replay in annotation order,
min_conf=0.4. Fires = stabilized verb edges (what the engine actually consumes). tick_acc counts 'no confident verb' as a miss; tick_mca_conf is over confident ticks only.

| smoother | coverage | tick acc | tick mca (confident) | fires | fire precision |
|---|---|---|---|---|---|
| raw | 0.8857 | 0.6379 | 0.6957 | 1167 | 0.7498 |
| ema | 0.4163 | 0.2741 | 0.643 | 277 | 0.8412 |
| majority | 0.972 | 0.3773 | 0.3353 | 550 | 0.5727 |
| hysteresis | 0.3953 | 0.2536 | 0.6344 | 227 | 0.8194 |
