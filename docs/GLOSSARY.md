# Glossary

Plain-language definitions, no jargon chains. Ordered by where you meet
them in the project.

**FramePacket** — One camera frame plus a frame_id (count) and
timestamp_ns (nanosecond clock). The unit that flows through the app.

**MJPEG** — Video delivered as a stream of JPEG images. The camera's only
compressed mode under WSL2; uncompressed (YUYV) hangs.

**Sliding window** — The last ~2 seconds of frames kept in a buffer; the
verb model always classifies this window, never the whole video.

**Frame sampler** — Rule for picking exactly T frames out of a longer
span (segment-centre sampling: split into T equal parts, take each
part's middle frame).

**stop_frame convention (S1)** — Whether an annotation's stop_frame is
inside the segment (inclusive) or one past it (exclusive). EPIC is
inclusive — verified on real cuts (reports/annotation_check).

**Verb / noun / segment** — EPIC vocabulary: an action segment is one
annotated interval of video with a verb (what is done) and noun (what it
is done to). ~90k segments over 700 videos.

**Many-shot** — A class with more than 100 training instances. All 8
demo verbs clear the bar (reports/verb_counts.md).

**Backbone** — The convolutional network (ResNet-18/34, MobileNetV2)
that turns an image into a feature vector; pretrained on ImageNet.

**Feature / embedding** — A vector (e.g. 512-d or 384-d) representing an
image's content. Nearby vectors = similar content.

**L2-normalized** — Vector scaled to length 1. Then dot product = cosine
similarity.

**Cosine similarity** — Dot product of unit vectors; 1 = same direction,
0 = unrelated. The matcher's score.

**Prototype** — One vector per ingredient: the mean of its scan-phase
embeddings, re-normalized. Matching = nearest prototype.

**Scan phase** — The 10 s per ingredient before cooking where the
prototype is captured. No training; just averaging.

**Show zone (D4)** — The fixed rectangle in the frame where ingredients
are held up; only that crop is embedded.

**Softmax** — Turns scores into probabilities that sum to 1 (subtract
the row max first so exp never overflows).

**Top-k accuracy** — Fraction of samples whose true class is among the
k highest-scoring predictions.

**Mean-class accuracy** — Average of per-class accuracy over classes
present in the labels; long-tail-honest headline metric.

**Majority baseline** — Accuracy of always guessing the most frequent
class. Shown next to every number so a "good" score on imbalanced data
can't fool you.

**Confusion matrix** — (C, C) table; rows true class, columns predicted.
Diagonal = correct; everything else is a specific kind of mistake.

**BatchNorm** — Layer that normalizes its input using batch statistics
in training, fixed running statistics in eval. At batch size 1 in train
mode it collapses (mean of one sample is itself) — why inference always
runs in eval mode.

**Temporal aggregation** — Combining per-frame features into one clip
decision: TSN average, attention pooling, or single middle frame.

**Attention (temporal)** — Learned weights (softmax over T frames) that
decide how much each frame contributes to the pooled clip vector.

**Smoother (D6)** — Converts a noisy per-tick verb stream into stable
events: EMA of probabilities, majority vote, or hysteresis.

**EMA** — Exponential moving average: new = alpha * value + (1 - alpha)
* old. Small alpha = smoother, slower to react.

**Hysteresis** — Keep the current decision until a challenger beats it
by a margin; debounce for recognition.

**Confidence threshold (D3a)** — Predictions below it read "no verb";
the first defence against firing on idle footage.

**Verb edge** — The moment the smoothed verb CHANGES to a new stable
value; the engine fires exactly one event per edge.

**State machine (recipe)** — Explicit rules walking the recipe steps on
(verb, ingredient) events: advance / wrong_order / repeat / idle / done.

**Registry** — src/registry.py: loads each core piece from reference/ or
core/ per configs/impl.yaml, so the owner's versions drop in everywhere.

**Equivalence vectors** — Shared test inputs+expected outputs
(tests/equivalence/vectors/) generated from the reference; the owner's
core/ versions must reproduce them.

**Tier 1/2/3** — Demo ladder: keyboard-simulated verbs on webcam; trained
model on a recorded clip; everything live.

**EPIC-100 internal split (V1)** — The HF mirror has only train videos,
so validation = held-out participants (P22, P30) from the downloaded
train set. Documented deviation; official val unreachable at usable
speed.

**Run dir** — runs/<id>/: config copy, verb_map.json, log.csv per epoch,
best.pt checkpoint, metrics.json, confusion.png.

**Session log** — runs/sessions/<ts>.jsonl: every event of one demo run,
plus optional annotated mp4 for reels.
