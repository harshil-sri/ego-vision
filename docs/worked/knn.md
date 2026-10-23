# Worked example: kNN distances (`reference/exercises/knn.py`)

Close this after studying; write your own version tomorrow. Everything
below is computable on paper.

## The setup

3 train points, 2 test points, 2 features:

```
X_train = [[0, 0],     y_train = [0, 1, 2]
           [1, 0],
           [0, 1]]
X_test  = [[3, 4],
           [0.1, 0]]
```

**The rule that unblocks the whole file:** the distance matrix is
`(num_test, num_train)` = **(2, 3)**. `dists[i, j]` pairs TEST row `i`
with TRAIN row `j`. Row = test, column = train. Fix the output shape
first; every loop then writes one cell or one row.

## The distances, by hand

`dists[i, j] = sqrt(sum((X_test[i] - X_train[j])**2))` — the plain L2 norm.

| | train 0 = (0,0) | train 1 = (1,0) | train 2 = (0,1) |
|---|---|---|---|
| **test 0 = (3,4)** | sqrt(9+16) = **5.0** | sqrt(4+16) = **4.4721** | sqrt(9+9) = **4.2426** |
| **test 1 = (0.1,0)** | sqrt(0.01) = **0.1** | sqrt(0.81) = **0.9** | sqrt(0.01+1) = **1.0050** |

Verify a cell: test 0 vs train 1: (3−1, 4−0) = (2, 4) → sqrt(4+16) = sqrt(20) ≈ 4.4721. ✓

Running the code (`compute_distances_two_loops(X_test, X_train)`) prints
exactly:

```
[[5.     4.4721 4.2426]
 [0.1    0.9    1.005 ]]
```

## The no-loops version: one identity, no magic

`||a − b||² = ||a||² + ||b||² − 2·a·b`. For test 0 (norm² 25):

```
25 + [0, 1, 1] − 2·[0·3+0·4, 1·3+0·4, 0·3+1·4]  =  [25, 20, 18]
sqrt -> [5.0, 4.4721, 4.2426]                     ✓ matches the loops
```

In code: `test_sq` is (2, 1), `train_sq.T` is (1, 3), `X_test @ X_train.T`
is (2, 2)→wait—(2, 3): (num_test, D) @ (D, num_train). Broadcasting
(2,1) + (1,3) -> (2,3). The matmul is the ONLY O(N·D·T) work; everything
else is O(N·T).

## Predictions

`predict_labels(dists, y_train, k=1)`: per test row, argsort ascending,
take first k columns, majority vote (ties -> smallest label id).

- test 0: nearest column is 2 (4.2426) -> label **2**
- test 1: nearest column is 0 (0.1) -> label **0**

With k=3 both rows see all labels [0,1,2] (each appears once; bincount tie
-> smallest id wins) -> **[0, 0]**. Ties are why "k is odd" is folklore.

## What each row and column means

Row i answers: "how far is test point i from EVERY train point?" Column j
answers: "how far is train point j from every test point?" kNN prediction
for test i only ever reads row i — one argsort of 3 numbers here, of
71,585 numbers for the full EPIC train set (which is why kNN is not the
verb model — it is the SCAN-PHASE matcher's idea: distance to a
prototype).

## Three common mistakes

1. **Transposed output**: writing (num_train, num_test) and then indexing
   `dists[j, i]` — predictions come out silently wrong, not crashy. Guard:
   assert `dists.shape == (num_test, num_train)` before predicting.
2. **Forgetting the sqrt** (returning squared distances): predictions are
   UNCHANGED for k=1 (monotonic) but wrong for any threshold on distance
   (the matcher's similarity cutoffs live in a different range).
3. **`np.sqrt` of the expansion going negative**: float error makes
   `25 + 18 − 2·18` occasionally −1e-13 → sqrt gives NaN. The reference
   clips with `np.maximum(sq, 0)` before the sqrt.

## Edge cases

- k > num_train: argsort slice silently gives fewer than k — decide
  (reference: takes what exists).
- Duplicate points at distance 0: fine; ties resolved by smallest id.
- float32 vs float64: the two-loop and no-loop versions agree to ~1e-6 in
  float32; tests use `atol=1e-8` with float64. Know which you're holding.
