# Worked example: BatchNorm forward (`reference/exercises/batchnorm.py`)

4 samples, 2 channels, per-channel scalars (so the (N, C) path; the
(N, C, H, W) path just reshapes to (N·H·W, C) and does the same math per
column).

## The setup

```
x (4, 2) = [[1, 1],      gamma = [1, 1]    beta = [0, 0]
            [3, 1],      running_mean = [0, 0]   (fresh start)
            [5, 3],      running_var  = [1, 1]
            [7, 3]]
```

Channel 0 values: 1, 3, 5, 7 → mean **4**, biased var **5**
(unbiased: 5·4/3 = **6.667**).
Channel 1 values: 1, 1, 3, 3 → mean **2**, biased var **1** (unbiased **1.333**).

## Train mode: normalize with THIS batch's stats

xhat (channel 0, eps=1e-5 ≈ 0 for hand math):
(1−4)/√5 = **−1.3416**, (3−4)/√5 = **−0.4472**, (5−4)/√5 = **+0.4472**,
(7−4)/√5 = **+1.3416**.

```
out = [[-1.3416, -1.0000],
       [-0.4472, -1.0000],
       [+0.4472, +1.0000],
       [+1.3416, +1.0000]]
```

(gamma=1, beta=0, so out == xhat.)

**The PyTorch subtlety the test catches:** normalization used BIASED var
(5), but the RUNNING variance is updated with the UNBIASED one (6.667):

```
running_mean = 0.9·[0,0] + 0.1·[4,2]           = [0.4, 0.2]
running_var  = 0.9·[1,1] + 0.1·[6.667, 1.333]  = [1.5667, 1.0333]
```

Use the biased one in the update and eval mode drifts from
`nn.BatchNorm2d` — `test_batchnorm_matches_pytorch_train_and_eval`
compares both modes and fails on exactly this.

## Eval mode: normalize with the RUNNING stats

Same x through eval mode with running stats (0.4, 1.5667) for channel 0:
(1−0.4)/√(1.5667+eps) ≈ 0.479 — different from train mode's −1.3416, as
it must be: eval uses a fixed population estimate, not the batch in hand.

## The batch-size-1 trap (why model.eval() is law in this project)

Train mode at N=1, (1, C) input: the batch mean IS the sample →
xhat = (x−x)/√(0+eps) = **0 exactly** → out = beta. The layer ERASED the
input; nothing downstream can recover it. Eval mode: out = (x−rm)/√(rv)
with fixed constants — information preserved. Run
`batchnorm_train_vs_eval_demo` after one night's sleep and it still shows
train_out ≡ 0, eval_out = x. (For conv inputs at N=1 the stats pool over
the H·W positions so the collapse is partial — but the statistics still
come from ONE image: unrepresentative. Same conclusion: eval mode.)

## What each row and column means

In the (N·H·W, C) view: each ROW is one spatial position of one image;
each COLUMN is a channel with its OWN mean/var/gamma/beta. BN never mixes
channels. The running arrays are per-channel constants in eval mode.

## Three common mistakes

1. **Normalizing over the wrong axis** (per-sample across channels instead
   of per-channel across samples): shapes still check out, training
   limps, accuracy stays bad. Always state which axis is "the batch".
2. **eps inside vs outside the sqrt**: /√(var+eps) (standard, PyTorch) vs
   /√var+eps — differs by eps≈1e-5, invisible in training, visible when
   comparing against PyTorch to 1e-6 tolerance.
3. **Forgetting running-stat updates need `train=True` only**: updating
   them during eval silently shifts the population as you evaluate.

## Edge cases

- Batch of 1 in train mode (2D path): xhat = 0, out = beta (the trap
  above). PyTorch REFUSES (raises "Expected more than 1 value per
  channel"); the reference shows the degeneracy instead of raising —
  that's the point of the demo.
- var = 0 (constant channel): xhat = 0/√eps = 0 — safe, no NaN, thanks
  to eps.
- Momentum semantics: PyTorch `running = (1−momentum)·running +
  momentum·batch` — "new weight", not "decay rate of old" (the tf-style
  inversion is a classic porting bug).
