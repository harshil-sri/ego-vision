# Reasoning-chain evaluation (real val videos, perfect-matcher assumption)

Cache: 1715 segments / 10 videos from `runs/videomae_mlp/eval_cache.npz`. Pseudo-recipes = each video's annotated (verb, noun) sequence; leak=0.05, min_conf=0.4, policy=strict.

| smoother | recipe prior | advance rate | wrong_order | repeat | completed | median first-wrong step |
|---|---|---|---|---|---|---|
| oracle | off | 0.36 | 66 | 0 | 0/10 | 5 |
| raw | off | 0.12 | 59 | 0 | 0/10 | 1 |
| raw | ON | 0.12 | 61 | 0 | 0/10 | 1 |
| majority | off | 0.0733 | 26 | 0 | 0/10 | 1 |
| majority | ON | 0.0733 | 26 | 0 | 0/10 | 1 |
| ema | off | 0.04 | 24 | 0 | 0/10 | 1 |
| ema | ON | 0.04 | 32 | 0 | 0/10 | 1 |
