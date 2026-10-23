# Worked example: scan-phase matcher (`reference/enrollment_ref.py`)

Three ingredients enrolled from pure-colour frames; the toy embedder is
"mean colour of the frame, L2-normalized" so cosine similarities are
arithmetic on unit vectors.

## Enroll: frames -> prototype

`embed_fn([frame])` on a pure red frame (255, 0, 0): mean colour
(1.0, 0.0, 0.0) (after /255), already unit length →
prototype_red = **(1, 0, 0)**.
Blue: prototype_blue = **(0, 0, 1)**.

Enrolling 4 red frames with slight noise: each embeds to ≈(0.999, ±0.03,
±0.03); the prototype is the MEAN re-normalized — noise cancels, the
direction survives. That averaging IS the whole "scan phase learns your
ingredients in 10 seconds" trick: no training, just a mean on the sphere.

## Match: cosine similarity = dot product of unit vectors

Query: an orange frame (255, 128, 0) → /255 = (1, 0.502, 0) →
normalize (÷1.1186) → (0.894, 0.449, 0).

```
sim(orange, red)  = 0.894·1 + 0.449·0 + 0·0   = 0.894
sim(orange, blue) = 0.894·0 + 0.449·0 + 0·1   = 0.000
```

With threshold 0.9: best score 0.894 < 0.9 → **("unknown", 0.894)**.
Lower the threshold to 0.85 and orange becomes "red" with the SAME
score — the threshold is the false-accept/false-reject dial (D7 evidence
sweeps exactly this; see scripts/eval_matcher.py).

The matrix form the code uses: (1, D) @ (K, D).T -> (1, K); argmax picks
the column. K prototypes = K dot products, no loops.

## What each row and column means

`prototypes` (K, D): row k is ingredient k's unit direction in embedding
space; `names[k]` is its label. `sims` (1, K): column k is the query's
closeness to ingredient k. A match is an argmax OVER COLUMNS with a floor
under it. D = 384 for DINOv2 ViT-S/14 (default, D7); the math is
dimension-agnostic.

## Three common mistakes

1. **Forgetting to re-normalize the mean**: the mean of unit vectors is
   SHORTER than 1 (0.9 for perpendicular pairs, → 0 for opposite ones);
   dot products against a short prototype shrink and everything starts
   reading "unknown". Re-normalize after averaging, always.
2. **Comparing un-normalized embeddings with dots**: cosine needs unit
   vectors; a bright frame has a longer raw embedding and wins by
   magnitude, not direction. The reference defensively re-normalizes
   embed_fn output.
3. **Threshold on the NAME instead of the score**: returning
   ("unknown", 0.0) hides the actual similarity; the UI/debug drawer
   wants the real score to show "almost red (0.894)". Return the best
   score with the unknown name.

## Edge cases

- match() with nothing enrolled: ("unknown", 0.0) — the app shows "scan
  phase not done", never crashes.
- Retake: enroll(same name) REPLACES the prototype (the scan UI's retake
  button relies on this).
- Two visually identical ingredients (two brands of salt): both
  prototypes near-identical → scores tie; the argmax picks whichever came
  first — a real limitation of prototype matching the owner should be
  able to state out loud (it is WHY the demo recipe uses visually
  distinct ingredients).
