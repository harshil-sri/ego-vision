# Reasoning-chain evaluation (real val videos, perfect-matcher assumption)

Cache: 593 segments / 10 videos from `runs/baseline`. Pseudo-recipes = each video's annotated (verb, noun) sequence; leak=0.05, min_conf=0.5.

| smoother | recipe prior | advance rate | wrong_order | repeat | completed | median first-wrong step |
|---|---|---|---|---|---|---|
| oracle | off | 0.3357 | 58 | 1 | 0/10 | 3 |
| raw | off | 0.0 | 7 | 0 | 0/10 | 0 |
| raw | ON | 0.021 | 14 | 0 | 0/10 | 0 |
| majority | off | 0.014 | 10 | 0 | 0/10 | 0 |
| majority | ON | 0.021 | 15 | 0 | 0/10 | 0 |
| ema | off | 0.0 | 1 | 0 | 0/10 | 0 |
| ema | ON | 0.007 | 5 | 0 | 0/10 | 0 |
