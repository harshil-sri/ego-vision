# Worked example: temporal attention (`reference/temporal_attention_ref.py`)

B=1 clip, T=3 frames, D=2 features per frame. Weights are set BY HAND so
every number below is paper-checkable.

## The setup

```
W = [[0.5, 0.0],     v = [1.0, 0.0]    (v has no bias)
     [0.0, 0.5]]

frames x (1, 3, 2):  frame 0 = [1, 0]   frame 1 = [0, 1]   frame 2 = [1, 1]
```

## Step 1 — project each frame: tanh(W x_t) -> (B, T, D)

With this W the projection just scales: tanh(0.5)=**0.4621** on the
"active" coordinate, 0 on the other.

```
projections = [[0.4621, 0    ],
               [0     , 0.4621],
               [0.4621, 0.4621]]
```

## Step 2 — score each frame: v·proj -> (B, T)

v looks only at coordinate 0: scores = [0.4621, 0, 0.4621]. Frames 0 and
2 (both "hot" in coordinate 0) tie for attention; frame 1 scores 0.

## Step 3 — softmax over TIME (dim=1)

exp: [1.5874, 1.0, 1.5874], sum 4.1748 →
**weights = [0.3802, 0.2395, 0.3802]**, sum = 1.0 (up to float32:
0.99999994). Softmax is over the T axis ONLY — dim=1 in (B, T, D). dim=0
would softmax across the BATCH: every sample's weights then depend on
which OTHER samples share the batch (nondeterministic predictions).

## Step 4 — weighted sum over time: pooled = Σ w_t · x_t -> (B, D)

```
0.3802·[1,0] + 0.2395·[0,1] + 0.3802·[1,1]
= [0.3802 + 0 + 0.3802, 0 + 0.2395 + 0.3802]
= [0.7604, 0.6197]        code: [0.7605, 0.6198]  ✓ (float rounding)
```

Compare plain averaging (TSN): [2/3, 2/3] = [0.667, 0.667]. Attention
UP-weighted the frames its score liked — with trained v/W it learns which
moments of the ~2 s window carry the verb ("the instant the knife starts
moving" vs "holding still").

## What each row and column means

`weights` (B, T): row b is a probability distribution over TIME for clip
b — entry t = "how much frame t contributed". `pooled` (B, D): one
feature vector per CLIP (the (B, T, D) stack collapses over T). The learn
lives in W (D→D, "what to look for") and v (D→1, "how strongly this frame
counts").

## Three common mistakes

1. **Softmax over the wrong dim** (batch or feature axis instead of time)
   — see step 3; shapes don't catch it, only the sum-to-1-over-T assert
   does.
2. **Weighted sum that drops a frame's feature dims** — the broadcast is
   `weights.unsqueeze(-1) * x`: (B, T, 1) * (B, T, D) -> (B, T, D), then
   `.sum(dim=1)`. Doing `weights @ x` directly fails or silently
   transposes depending on shapes.
3. **Forgetting eval()/no_grad at inference**: attention itself has no
   BatchNorm, but the backbone feeding it does — the whole model inherits
   the eval-mode rule (docs/worked/batchnorm.md).

## Edge cases

- T = 1: softmax of one score = [1.0]; pooled = that frame. Degenerate
  but exact (live startup with one frame in the buffer).
- All-equal scores: uniform weights — attention gracefully becomes
  averaging.
- Untrained attention at init: scores ≈ 0 → near-uniform → behaves like
  TSN until v/W learn something. (Useful sanity: if attention underforms
  TSN badly EARLY in training, suspect a bug, not the architecture.)
