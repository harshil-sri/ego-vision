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
| 25 | Hazard models: "what is the chance per tick that the event starts now, given that it has not started yet" | `scripts/holoassist_need_controlled.py`, `scripts/holoassist_idle_spell.py` | Why is one row per 0.5 s tick, with a 1 only in the last tick, a correct way to model waiting time? What does an odds ratio of 0.17 mean in words? | Singer and Willett, "Applied Longitudinal Data Analysis", chapters 10 and 11 |
| 26 | Censoring and missing counterfactuals | `scripts/holoassist_wait_policy.py` (limit 1 in the report) | The instructor's correction ends the mistake. Why can nobody say if the performer would have fixed it alone? Which direction does this bias "failed"? | Any survival-analysis text, chapter on censoring |
| 27 | Decision policies and operating points: two kinds of error, and comparing policies at the same rate of one error | `scripts/holoassist_wait_policy.py` (`rates`, `best`) | Trace `tests/test_research.py::test_wait_policy_rates` on paper. Why is "lowest unneeded rate at 20% failed" a fairer comparison than accuracy? | scikit-learn user guide, "Tuning the decision threshold" |

## Added 2026-11-19: the rating study, the label checks and the paper

Same method. The order goes from the model, to the experiment, to the
labels, to how to read a result. Items 28 and 29 come first because the
paper's main numbers depend on them.

| # | Topic | Where it is used here | Check yourself | External resource |
|---|---|---|---|---|
| 28 | Gradient-boosted trees (histogram version): trees, boosting, learning rate, number of iterations, early stopping | Onset model in `scripts/holoassist_predict.py`, `scripts/holoassist_from_video.py`, `scripts/wtag_from_video.py` (`HistGradientBoostingClassifier`) | Why does a tree model need no feature scaling? What do learning rate and `max_iter` trade off? With more than 10,000 rows, scikit-learn turns early stopping on by default and holds out a random 10% of the training rows. Why does that choose too many iterations when neighbouring 0.5 s ticks of one session are near copies? Why does it still not leak into the test fold? | scikit-learn user guide, "Histogram-Based Gradient Boosting"; Hastie, Tibshirani and Friedman, "The Elements of Statistical Learning", chapter 10 |
| 29 | Ratio of two shares (relative risk) and its bootstrap interval | Every "x times chance" number: `scripts/egoextra_label_check.py`, `scripts/wtag_label_check.py` | Furniture: S at 20% of 212 onsets and 34% of 110 random seconds. Compute the ratio by hand. Why must the resample be over videos and not over moments? | Agresti, "Categorical Data Analysis", chapter 2 (comparing proportions); Efron and Tibshirani, chapter 6 |
| 30 | Pre-registration and the "garden of forking paths" | `study/PREREGISTRATION.md` and its amendment (collection closed before any rating was read) | Name three choices the pre-registration fixed in advance (outcome, exclusion, test). For each, say how the result could have been nudged if it had been chosen after seeing the data. | Gelman and Loken, "The garden of forking paths" (2013); Nosek et al., "The preregistration revolution", PNAS 2018 |
| 31 | Within-subject experiments: randomised versions, the rater as the unit, interaction (difference of differences), statistical power | `scripts/rating_study_build.py`, `scripts/rating_study_analyze.py`, `reports/rating_study.md` | Routine held minus now: -0.24 (-0.60 to +0.14); interaction +0.58 (+0.02 to +1.13). Why average per rater and cell first, then resample raters? Roughly how many raters would make the routine interval half as wide? | Gelman, Hill and Vehtari, "Regression and Other Stories", chapter 16 (design and sample size) |
| 32 | Inter-rater agreement: Cohen's kappa, chance agreement, the prevalence problem | `scripts/label_agreement.py`, `reports/egoextra_label_agreement.md` | From the 3x3 table (rows first labels S, B, F: 4 3 6 / 1 4 4 / 0 5 33) compute agreement and kappa by hand (answer: 0.68 and 0.36). Why can "S against the rest" agree 83% and still have kappa 0.37? | McHugh, "Interrater reliability: the kappa statistic", Biochemia Medica 2012; Feinstein and Cicchetti, "High agreement but low kappa", J Clin Epidemiol 1990 |
| 33 | Label noise and measurement error: what wrong labels do to a ratio | The furniture result (0.59) against the checker, who calls S far less often | If the extra S calls are wrong at the same rate at onsets and at random seconds, does 0.59 move toward 1 or away from it? If they are more common at random seconds, what then? | Rothman, Greenland and Lash, "Modern Epidemiology", chapter on measurement error and misclassification |
| 34 | Blind labelling: what the labeller must not know, and why | `scripts/label_check_kit.py` (shuffled, no kind, no id, key kept apart) | List what the sheet hides and the bias each hidden item prevents. Why mix random seconds with onsets on the same sheet instead of two sheets? | Any clinical-trials text, chapter on blinding; Schulz and Grimes, "Blinding in randomised trials", Lancet 2002 |
| 35 | Replication, a second batch, and the winner's curse | Furniture: first batch 0.41, second 0.78, pooled 0.59 (`reports/egoextra_label_check.md`) | The second batch was drawn after the first was seen. Why is a first, striking estimate often too extreme? Which number should the paper report, and why not the first batch alone? | Ioannidis, "Why most discovered true associations are inflated", Epidemiology 2008 |
| 36 | Reading an interval that includes 1: absence of evidence, equivalence bounds | WTaG instructor 0.81 (0.59 to 1.08); paper section 6.2 | May the paper write "instructors do not avoid precise work in cooking"? What exactly can it write? What would a test that the ratio is above 0.8 need? | Lakens, "Equivalence tests", Social Psychological and Personality Science 2017; Altman and Bland, "Absence of evidence is not evidence of absence", BMJ 1995 |
| 37 | Stratification: checking that an effect is not caused by a third variable | Action length (`8d2052c`): screw/insert avoided at every length (0.46 under 2 s to 0.28 at 10 s and more), other verbs 0.76 to 1.02 | Why could "long actions are avoided" explain the main result, and how does splitting by length rule it out? What does stratification not rule out? | Pearl, "The Book of Why", chapter 4; Rothman et al., chapter on stratified analysis |
| 38 | Video files: frames, key frames, group of pictures, time stamps, and what a seek really does | The cooking bug: `frame()` against `frames_in_order()` in `scripts/label_check_kit.py` | A seek to 300.25 s returned the frame at 303.99 s. Explain with key frames and missing time stamps. Why does reading in order fix it, and what does it cost? Name a quick test that would have caught the bug before labelling. | FFmpeg wiki, "Seeking"; any introduction to H.264 (I, P and B frames) |

Not deep learning, but used here and worth one evening: password-based
encryption in the browser (`scripts/label_check_publish.py`: PBKDF2,
AES-GCM, why the passphrase after "#" never reaches the server). MDN, "SubtleCrypto"; OWASP Password Storage Cheat Sheet.
