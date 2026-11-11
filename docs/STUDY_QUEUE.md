# Study queue

The owner's study order. For each item: study the reference file, trace
the worked example ON PAPER, close it, write your own version the next
day (25 min), test it against the shared vectors
(`python3 -m pytest tests/equivalence --impl core` after copying the
stub into core/), then cold-redo at +1 day and +3-4 days.

| # | Topic | Reference file | Worked example | External resource |
|---|---|---|---|---|
| 1 | kNN + vectorized numpy (the matcher's idea) | reference/exercises/knn.py | docs/worked/knn.md | CS231n A1: https://cs231n.github.io/assignments2025/assignment1/ |
| 2 | Softmax, cross-entropy, gradients | reference/exercises/softmax.py | docs/worked/softmax.md | same CS231n A1 |
| 3 | Metrics (every result is judged by these) | reference/metrics_ref.py + stub exercises/stubs/metrics.py | docs/worked/metrics.md | EPIC-100 paper metrics section: https://arxiv.org/abs/2006.13256 |
| 4 | BatchNorm, train vs eval | reference/exercises/batchnorm.py | docs/worked/batchnorm.md | d2l.ai: https://d2l.ai/chapter_convolutional-modern/batch-norm.html + ESIEE lab: https://perso.esiee.fr/~chierchg/deep-learning/concepts/residual-cnn-student.html |
| 5 | Frame sampling | reference/frame_sampler_ref.py + stub | docs/worked/frame_sampler.md | TSN sections of the TSM paper: https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html |
| 6 | Training loop (read the real one) | src/train/loop.py + src/train/config.py | — (trace one epoch by hand: loss, grad, step, log row) | PyTorch with examples: https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html |
| 7 | Transfer learning (the backbone) | src/models/backbones.py + src/models/variants.py | — (check feature dims and the (B*T,C,H,W) reshape) | https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html |
| 8 | Embeddings + prototypes (the scan phase) | reference/enrollment_ref.py + src/pipeline/embedder.py | docs/worked/enrollment.md | DINOv2: https://github.com/facebookresearch/dinov2 + https://blog.roboflow.com/how-to-classify-images-with-dinov2/ |
| 9 | Temporal aggregation (TSN avg vs single frame) | src/models/variants.py (tsn_avg, single_frame) | — | TSM code ops/temporal_shift.py: https://github.com/mit-han-lab/temporal-shift-module |
| 10 | Attention over time | reference/temporal_attention_ref.py + stub | docs/worked/temporal_attention.md | d2l attention chapters: https://d2l.ai/ |
| 11 | State machine (yours already conceptually) | reference/state_machine_ref.py + stub | docs/worked/state_machine.md | — |

Core pieces to rewrite in core/ (stub -> fill -> equivalence test):
metrics (3), frame_sampler (5), enrollment (8), temporal_attention (10),
state_machine (11). The order above front-loads what the training runs
will show you.

Cold-rebuild exam target (roadmap W7): metrics, sampler, matcher,
attention — 25 minutes each, no help.

## Added 2026-11-11: foundations behind the frozen-feature model and the pilot

Same method as above (read, trace on paper, rewrite, cold-redo). These
have no core/ stub yet; the "check yourself" column is the exam.

| # | Topic | Where it is used here | Check yourself | External resource |
|---|---|---|---|---|
| 12 | Self-attention and the Vision Transformer: patches, tokens, class token, position embeddings | `src/models/frozen.py` (`FeatureHead` kind `tfm`), DINOv2, VideoMAE | By hand: 3 tokens of width 2, one attention head, compute the output. Explain why the `mean` head cannot tell "open" from "close" and the `tfm` head can (`tests/test_frozen.py`). | d2l.ai chapter 11; "An Image is Worth 16x16 Words" |
| 13 | Self-supervised pretraining and frozen features (linear probe) | `scripts/extract_features.py`, `scripts/train_head.py` | Explain why 3,720 clips overfit a fine-tuned ResNet-34 but train a head on frozen features well. State which weights receive gradients. | DINOv2 paper sections 1 to 3; VideoMAE paper sections 1 to 3 |
| 14 | Why motion needs a video model: appearance bias, Something-Something v2, tube masking | DINOv2 heads 0.33 against VideoMAE 0.65 mean-class accuracy on the same clips | Predict, then check in `runs/*/metrics.json`: which verb pairs does a per-frame model confuse most, and why? | Something-Something paper; TSM paper introduction |
| 15 | Train/serve mismatch: trimmed clips against sliding windows, and the missing "nothing is happening" class | `scripts/eval_sliding.py`, `reports/sliding_eval.md` | Explain why accuracy on trimmed clips overstates live accuracy. Propose two fixes. | EPIC-KITCHENS-100 paper, action detection section |
| 16 | Thresholds, precision and recall, ROC AUC, average precision, base rates | `scripts/holoassist_predict.py`, `min_conf` in `configs/app.yaml` | Base rate 2.25%, average precision 0.087: is that good? Compute the precision of a random guesser. | scikit-learn user guide, "Precision-Recall" |
| 17 | Permutation tests and null models; confidence intervals; many comparisons | `scripts/holoassist_timing.py` (circular shift) | Explain what the circular shift keeps and what it destroys. Why is the smallest possible p with 2,000 shifts 0.0005? | Any statistics text, chapter on resampling; Good, "Permutation Tests" |
| 18 | Leakage and grouped splits; selecting a model on the validation set | `GroupKFold` in the pilot scripts; "best" against "last" epoch in `scripts/train_head.py` | Why split by participant or session, not by clip? Why is the "best epoch" number optimistic? | scikit-learn user guide, "Cross-validation: group k-fold" |
| 19 | Correlation is not cause: confounders in observational data | `reports/research_gap_map.md` section 4, "What the pilot does not show" | Name the confounder between "hands idle" and "instructor speaks". Design an experiment that removes it. | Pearl, "The Book of Why", chapters 1 to 4 |
| 20 | Cross-fitting: using one model's predictions as another model's input without leakage | `scripts/holoassist_from_video.py` (inner `GroupKFold`) | The hand-state head saw the training sessions. Why must the timing model not be trained on that head's predictions for those same sessions? What goes wrong at test time? | scikit-learn user guide, "Stacked generalization" |
| 21 | Bootstrap intervals with clusters; paired differences | `--boot` in `scripts/holoassist_from_video.py` | Why resample sessions and not ticks? Why compute the difference inside each resample instead of comparing two intervals? | Efron and Tibshirani, "An Introduction to the Bootstrap", chapters 1 to 6 |
| 22 | Positive and negative controls | `scripts/holoassist_shift_control.py` against `scripts/egoextra_from_video.py`; trainee speech in `scripts/egoextra_hands.py` | A test finds "no relation". Name two reasons other than "there is none", and the control that rules each one out. | Lipsitch et al., "Negative controls", Epidemiology 2010 |
| 23 | Domain shift: a model trained on one camera and task, used on another | HoloAssist models applied to Ego-EXTRA | Why standardise each dataset with its own mean? What can still be wrong afterwards, and how would you find out without labels? | Any survey on domain adaptation, introduction only |
| 24 | HTTP range requests and the tar format | `scripts/holoassist_download_subset.py` | Draw the bytes of a tar file with two members. Which bytes does the index stage read? | RFC 9110 section 14 (Range); `man 5 tar` |

