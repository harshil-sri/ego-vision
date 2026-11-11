# Reasoning-chain evaluation (real val videos, perfect-matcher assumption)

Cache: 1715 segments / 10 videos from `runs/videomae_mlp/eval_cache.npz`. Pseudo-recipes = each video's annotated (verb, noun) sequence; leak=0.05, min_conf=0.4, policy=optimistic.

| smoother | recipe prior | advance rate | wrong_order | repeat | completed | median first-wrong step |
|---|---|---|---|---|---|---|
| oracle | off | 0.8 | 0 | 0 | 9/10 | None |
| raw | off | 0.4733 | 0 | 2 | 6/10 | None |
| raw | ON | 0.4733 | 0 | 2 | 7/10 | None |
| majority | off | 0.2333 | 0 | 0 | 3/10 | None |
| majority | ON | 0.2467 | 0 | 0 | 4/10 | None |
| ema | off | 0.2 | 0 | 0 | 1/10 | None |
| ema | ON | 0.24 | 0 | 0 | 3/10 | None |
