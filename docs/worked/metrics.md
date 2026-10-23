# Worked example: metrics (`reference/metrics_ref.py`)

6 samples, 4 classes. This example is deliberately NOT perfect — perfect
predictions teach nothing about confusion matrices.

## The setup

```
scores (6, 4), one row per sample, columns = classes 0..3 (higher = more
confident). argmax gives the prediction:

pred = [0, 1, 2, 1, 1, 2]
y    = [0, 1, 2, 1, 0, 0]      note: class 3 NEVER appears in y
```

Sample-by-sample: samples 0, 1, 2, 3 correct; sample 4 (y=0) predicted 1;
sample 5 (y=0) predicted 2.

## Accuracy and top-k

- **top-1** = 4/6 = **0.6667**
- **top-2** = 0.8333 (sample 5's true class 0 has the 2nd-highest score
  0.20 vs class 2's 0.55 — count it; sample 4's class 0 at 0.45 vs 0.50 —
  count it too: 5/6)

## Per-class accuracy: (C,) vector

Class 0: seen in samples 0, 4, 5 → 1/3 = **0.3333** (the model's weakness,
invisible in plain accuracy!)
Class 1: samples 1, 3 → 2/2 = **1.0**
Class 2: sample 2 → 1/1 = **1.0**
Class 3: never seen → **0.0 by convention** (not NaN — documented rule so
tables stay well-defined).

## Mean-class accuracy: the honest headline on long-tailed data

Mean over classes PRESENT in y = (0.3333 + 1.0 + 1.0)/3 = **0.7778**.
Class 3 is EXCLUDED (absent from y), not counted as 0 — the reference
documents this; if you include absent classes you can drag the score
anywhere you like by adding empty class ids.

## Confusion matrix: (C, C), rows true, cols predicted

```
          pred 0  pred 1  pred 2  pred 3
true 0  [   1       1       1       0   ]
true 1  [   0       2       0       0   ]
true 2  [   0       0       1       0   ]
true 3  [   0       0       0       0   ]
```

Row sums = class counts in y (3, 2, 1, 0). Column sums = how often each
class was predicted (1, 3, 2, 0). Read the errors off row 0: class-0
samples scattered to classes 1 and 2. The all-zero row/column for class 3
immediately shows it was never involved.

## Majority baseline: the number every table must sit next to

Most frequent class in y: 0 appears 3 times → baseline = 3/6 = **0.5**.
Our 0.6667 accuracy beats it by 0.167 — on EPIC's long tail (open: 4851,
mix: 212 train instances) majority accuracy can be ~10-20% while the
model does nothing useful. This is why `majority_baseline` is computed
next to every number in this project.

## What each row and column means

`confusion[i, j]`: "i was the truth, j was the guess". The DIAGONAL is
agreement; everything off it is a specific kind of mistake. Row i
normalized = per-class accuracy for class i. `scores` columns are ranked
by confidence for top-k; `per_class` is indexed by TRUE class only.

## Three common mistakes

1. **Confusion matrix transposed** (pred as rows): row sums then equal
   PREDICTION counts and per-class accuracy read from the wrong axis.
   Convention here: `cm[t, p]`, rows true. (sklearn matches this; some
   tutorials don't.)
2. **Mean-class accuracy including absent classes as 0**: silently
   punishes you for classes you never evaluated.
3. **Top-k with k > C or k = 0**: always clip/validate — k=C gives 1.0
   always (degenerate), k=0 is meaningless.

## Edge cases

- Empty y: every function returns 0.0 / zeros (never NaN, never raises).
- Ties in argmax: numpy takes the FIRST max (smallest class id) —
  deterministic, fine, but know it exists.
- prob vs logits input: argmax and top-k don't care (softmax is
  monotonic); the reference documents that both are acceptable inputs.
