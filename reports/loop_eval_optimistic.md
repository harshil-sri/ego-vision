# Reasoning-chain evaluation (real val videos, perfect-matcher assumption)

Cache: 593 segments / 10 videos from `runs/baseline`. Pseudo-recipes = each video's annotated (verb, noun) sequence; leak=0.05, min_conf=0.5, policy=optimistic.

| smoother | recipe prior | advance rate | wrong_order | repeat | completed | median first-wrong step |
|---|---|---|---|---|---|---|
| oracle | off | 0.7483 | 0 | 0 | 6/10 | None |
| raw | off | 0.049 | 0 | 0 | 1/10 | None |
| raw | ON | 0.1189 | 0 | 0 | 1/10 | None |
| majority | off | 0.0769 | 0 | 1 | 1/10 | None |
| majority | ON | 0.1329 | 0 | 0 | 1/10 | None |
| ema | off | 0.007 | 0 | 0 | 0/10 | None |
| ema | ON | 0.042 | 0 | 0 | 0/10 | None |
