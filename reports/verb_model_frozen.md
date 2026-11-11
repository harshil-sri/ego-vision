# Verb model: frozen features + small head (2026-11-11)

All numbers are from commands run on this machine (RTX 4060, Windows) on
the V1 split: participants P22 and P30 held out, 55 EPIC-256 videos.
Commands are at the end.

## 1. Result in one table

Validation = trimmed action clips of the held-out participants (1,715
clips, 8 verbs). "Last epoch" numbers: no epoch was picked on the
validation set.

| Model | Trainable part | Accuracy | Mean-class accuracy | Top-3 (best epoch) |
|---|---|---|---|---|
| Majority class ("take") | - | 0.319 | 0.125 | - |
| ResNet-34 fine-tuned, single frame (old `runs/baseline`, old labels, 593 clips, best epoch) | whole network | - | 0.22 | - |
| DINOv2 ViT-S/14 frozen, mean head + motion | head | 0.41 | 0.33 | 0.75 |
| DINOv2 frozen, Transformer head + motion | head | 0.42 | 0.33 | 0.74 |
| **VideoMAE ViT-B (Something-Something v2) frozen, MLP head** | head | **0.67** | **0.65** | **0.91** |

DINOv2 and VideoMAE rows: mean of 3 seeds, standard deviation at most
0.016. The saved model `runs/videomae_mlp` (seed 42): accuracy 0.681,
mean-class accuracy 0.655.

Per-verb recall of the saved model (last epoch): close 0.66, cut 0.76,
mix 0.71, open 0.78, pour 0.41, put 0.64, take 0.75, wash 0.54.

## 2. What changed, and why it worked

1. **Labels.** The old pipeline kept a clip only if the raw narration word
   was one of the 8 verbs ("take" but not "pick up"). Using the official
   verb class instead gives 5,435 clips where the old rule gave 2,995, on
   the same 55 videos.
2. **Frozen backbone.** 3,720 training clips are too few to fine-tune a
   backbone: the old model overfit after one epoch. Here the backbone is
   never trained. Each clip becomes a fixed vector once
   (`scripts/extract_features.py`, about 16 minutes for all clips); only a
   small head is trained (seconds).
3. **Motion-pretrained backbone.** DINOv2 sees single frames. Verbs such
   as open/close or take/put differ in motion, not appearance. VideoMAE
   fine-tuned on Something-Something v2 was trained on exactly such
   motion-defined labels. It has never seen EPIC-KITCHENS, so the held-out
   participants stay unseen. Open/close confusions: 17 of 381 clips.
4. **Fusion does not help.** A linear probe on VideoMAE + DINOv2 features
   is not better than VideoMAE alone (0.61 against 0.65 mean-class
   accuracy), so the model uses VideoMAE only.

## 3. Under live conditions the numbers are lower

The table above uses clips cut exactly at the start and stop of each
action. The app does not know where actions start. It sees the last 2 s
at every tick. `scripts/eval_sliding.py` replays the full held-out videos
that way (10,205 ticks, one every 0.5 s; 41% of ticks have no demo verb).

Saved model, 2 s window (`reports/sliding_eval_trimmed_w2.0.md`):

- On ticks where a demo verb is happening: accuracy 0.56 (trimmed clips:
  0.68).
- The model has no "nothing is happening" answer. It is confident
  (probability >= 0.5) on 57% of other-verb ticks and 63% of background
  ticks.
- A 1 s window is worse than a 2 s window (accuracy 0.50).

Fires, as the engine counts them (a fire = the stabilised verb changes).
Lenient precision = an action with the fired verb overlaps the window.
Action recall = share of the 1,715 annotated demo-verb actions during
which the stabilised verb was correct at least once.

| Smoother, min_conf | Fires per minute | False fires per minute | Precision (lenient) | Action recall |
|---|---|---|---|---|
| none, 0.5 | 27.3 | 14.0 | 0.49 | 0.76 |
| none, 0.7 | 14.3 | 5.5 | 0.62 | 0.58 |
| EMA 0.5, 0.5 (demo preset) | 14.3 | 5.4 | 0.62 | 0.61 |
| EMA 0.5, 0.6 | 9.3 | 2.8 | 0.70 | 0.48 |
| EMA 0.5, 0.7 | 5.3 | 1.1 | 0.79 | 0.33 |

For scale: these videos contain 20 demo-verb actions per minute. The
user's own demo is slower and more deliberate than EPIC kitchen footage,
so these are pessimistic for the demo. That statement is an expectation,
not a measurement.

## 4. Tried and rejected: a head trained on sliding windows with a "none" class

Idea: train on the same windows the app sees, with a ninth class "none"
(`scripts/extract_sliding.py`, `scripts/train_window_head.py`).

Result: it does not beat the simple model with a higher threshold.

| Head, raw output | False fires per minute | Precision (lenient) | Action recall |
|---|---|---|---|
| Trimmed-clip head, min_conf 0.7 | 5.5 | 0.62 | 0.58 |
| Window head with "none", min_conf 0.5 | 6.0 | 0.60 | 0.52 |
| Trimmed-clip head, min_conf 0.8 | 2.5 | 0.71 | 0.45 |
| Window head with "none", min_conf 0.7 | 2.1 | 0.72 | 0.36 |

Also tried, no gain (tick accuracy stayed at 0.48 to 0.50): adding the
1 s window to the 2 s window, adding the windows 1 s and 2 s earlier,
deciding about a moment further in the past.

Also tried (2026-11-11), no gain: adding 79,280 HoloAssist ticks to the
training set of the window head (`--mix-holoassist`; "idle" ticks as
"none", or all ticks with grab -> take, place -> put, open, close). With
EMA 0.5 and min_conf 0.5 the best variant gives 2.1 false fires per
minute, lenient precision 0.74, action recall 0.42
(`reports/sliding_eval_window_w2.0_mix_holo_all.md`). The trimmed-clip
head reaches the same trade-off with a threshold between 0.6 and 0.7
(table in section 3).

Reading: with frozen features, "is a demo verb happening at this moment"
is the hard part, and a head cannot fix it. The next step up is to
fine-tune the video backbone on windows, or to use a detector for
hand-object contact as a gate. Neither is done.

The engine support for a "none" class is kept (tested, unused by the
demo preset), because any better window model will need it.

## 5. In the app

- `src/pipeline/model_source.py` `FrozenVerbSource` loads `runs/videomae_mlp`.
  On 40 held-out clips its probabilities equal the cached-feature
  probabilities (difference below 0.0001).
- One prediction: 88 ms in isolation, about 110 ms inside the running app
  (GPU, 16 frames, 3 crops).
- Clip mode ran end to end on Windows on P30_110: scan, enrol, cooking,
  verbs firing. This checks the mechanics, not the accuracy.
- Recipe-loop replay on trimmed clips (`reports/loop_eval_videomae_optimistic.md`):
  advance rate 0.47 (old model 0.13), 6 of 10 pseudo-recipes completed
  (old model 1 of 10).
- Demo preset: `configs/app_demo_videomae.yaml`.

Not done: live mode with the head-mounted camera on a real recipe. That
is the test that matters and it needs you.

## 6. Commands

```powershell
python -m scripts.extract_features --device cuda                      # DINOv2 features
python -m scripts.extract_features --encoder videomae --device cuda   # VideoMAE features
python -m scripts.train_head --sweep                                  # DINOv2 heads, 3 seeds
python -m scripts.train_head --feats /mnt/ssd/epic-kitchens/features/videomae_ssv2_T16 --sweep
python -m scripts.train_head --feats /mnt/ssd/epic-kitchens/features/videomae_ssv2_T16 --out runs/videomae_mlp
python -m scripts.extract_sliding --windows 2.0 1.0                   # about 50 minutes
python -m scripts.eval_sliding --run-dir runs/videomae_mlp --feats /mnt/ssd/epic-kitchens/features/videomae_ssv2_sliding --window 2.0 --rebuild
python -m scripts.train_window_head --window 2.0
python -m scripts.run_app --config configs/app_demo_videomae.yaml --mode clip --clip <video.mp4> --run-dir runs/videomae_mlp
```
