# Worked example: softmax loss + gradient (`reference/exercises/softmax.py`)

2 samples × 3 classes, D = 3. Paper-checkable.

## The setup

```
X = [[1, 2, 3],        W = [[0.1, 0.2, 0.3],     y = [0, 2]
     [1, 2, 5]]             [0.4, 0.1, 0.0],
                             [0.2, 0.2, 0.2]]
```

## Step 1 — scores: X @ W -> (N, D) @ (D, C) = (2, 3)

Row 1 by hand: [1·0.1+2·0.4+3·0.2, 1·0.2+2·0.1+3·0.2, 1·0.3+2·0.0+3·0.2]
= [0.1+0.8+0.6, 0.2+0.2+0.6, 0.3+0+0.6] = [1.5, 1.0, 1.3]. Code gives:

```
scores = [[1.5, 1.0, 0.9],
          [1.9, 1.4, 1.3]]
```

## Step 2 — subtract the row max, exp, normalize

Row 0 max = 1.5; row 1 max = 1.9. Note row 1 = row 0 + 0.4 — a constant
shift. Softmax is invariant to adding a constant to a whole row (it
cancels between numerator and denominator), so **both rows get identical
probabilities**. The code confirms:

```
probs = [[0.4640, 0.2814, 0.2546],
         [0.4640, 0.2814, 0.2546]]     (each row sums to 1)
```

Hand-check row 0 class 1: exp(1.0−1.5)=exp(−0.5)=0.6065;
exp(0)=1; exp(0.9−1.5)=exp(−0.6)=0.5488. Sum=2.1553.
0.6065/2.1553 = **0.2814**. ✓

**Why subtract the max:** exp(1000) overflows float64 (max ≈ 1.8e308) and
returns `inf`, and `inf/inf = nan` — training dies on confident logits.
Subtracting the row max keeps every exponent ≤ 0 (exp ≤ 1) and changes
nothing mathematically. It is a float-range fix, not a math change.

## Step 3 — loss: mean of −log p[correct]

Sample 0 (y=0): −log(0.4640) = 0.7680. Sample 1 (y=2): −log(0.2546) = 1.3686.
Mean = (0.7680+1.3686)/2 = **1.0679**. Code: `loss = 1.0679` (reg = 0). ✓

## Step 4 — gradient: (probs − onehot) / N, chained to W

dL/dscores (before the 1/N): probs with 1 subtracted at each correct class:

```
[[-0.5360, 0.2814, 0.2546],
 [ 0.4640, 0.2814, -0.7454]]
```

Check the sign story: correct classes got TOO LITTLE probability (0.464,
0.2546 < 1) → negative gradient → their scores get pushed UP next step.
dW = X.T @ dscores / N → (D, N) @ (N, C) = (3, 3). One cell by hand:
dW[1, 0] = (1·(−0.536) + 1·0.464)/2 = −0.0360.

The numerical gradient check (central difference) matches to relative
error < 1e-6 — run `tests/test_exercises.py::test_softmax_gradient_check`.

## What each row and column means

`scores`/`probs`: rows are SAMPLES, columns are CLASSES; entry (i, c) is
sample i's belief in class c. `dW`: rows are INPUT FEATURES, columns are
CLASSES; dW[d, c] says "to raise class c's logit, change feature d's
weight thus". `dscores` rows sum to 0 (softmax normalization) — a useful
sanity check when debugging.

## Three common mistakes

1. **Softmax over the wrong axis** (over N instead of C): with axis=0 each
   COLUMN sums to 1 — the loss still computes but is meaningless. Guard:
   `assert np.allclose(probs.sum(axis=1), 1)`.
2. **Forgetting the 1/N**: gradient magnitudes scale with batch size; with
   the mean loss you must divide dW by N or the effective learning rate
   changes with batch size.
3. **log(0)**: an extremely confident wrong prediction gives p ≈ 1e-30 →
   −log is finite (~69) but at exactly 0 it's −inf; the max-subtract keeps
   p ≥ exp(min−max)/C > 0 — another reason the shift is not optional.

## Edge cases

- reg > 0: loss += reg·||W||², dW += 2·reg·W — the reference shows both.
- C = 1: softmax is always 1.0, loss = 0, gradient = 0 (degenerate but
  well-defined — a nice assert to write yourself).
- Class ids out of [0, C): indexing `probs[np.arange(N), y]` silently
  wraps or raises depending on the value — validate y first.
