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
