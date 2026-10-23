# Architecture

Data flow, shapes at every arrow, and the concept each module uses. Read
top to bottom; every box names the file that implements it.

```
                        SCAN PHASE (before cooking)
 webcam /dev/video0 (MJPG 640x480, ~19 FPS measured)
   |
   v
 CameraWorker (src/camera_worker.py)  --thread--> LatestFrameBuffer (src/frame_buffer.py)
   |                                                  | newest FramePacket only
   v                                                  v
 CookingEngine loop (src/pipeline/engine.py)   pulls ~stream rate
   |  frames (H,W,3) uint8 BGR
   |  crop = show_zone (D4) -> (h,w,3)
   |  embedder (src/pipeline/embedder.py: DINOv2 ViT-S/14 default, D7)
   |      list[(h,w,3) uint8] -> (N, 384) float32 unit rows
   v
 IngredientMatcher.enroll (registry -> reference/enrollment_ref.py)
   |  prototype = renormalized mean of (N,384) -> (384,)   [kNN idea, 1 prototype/class]
   v
 EventBus "scan_progress" -> UI wizard + SessionLogger (runs/sessions/*.jsonl)


                        COOKING PHASE (live loop, tick_hz=5)
 LatestFrameBuffer -> SlidingWindowBuffer (src/pipeline/window.py)
   |  keeps last ~2.0 s (D5 window)
   |  .sample(T=8) via registry frame_sampler (reference/frame_sampler_ref.py)
   |      -> list[FramePacket] len T       [(start,stop,T,fps) -> (T,) int64]
   v
 VerbSource (src/pipeline/model_source.py)
   |  keyboard: none (UI injects)          Tier 1
   |  model: VerbModel (src/models/variants.py, Tier 2/3)
   |      (T,H,W,3) uint8 -> (1,T,3,224,224) float -> backbone (B*T,3,H,W)
   |      -> features (B,T,D) -> variant head -> logits (B,num_verbs)
   v
 softmax -> probs (num_verbs,) -> Smoother (src/pipeline/smoothers.py, D6)
   |  ema | majority | hysteresis -> (verb|None, confidence)  [D3a threshold]
   v
 on stabilized verb EDGE:
   IngredientMatcher.match(show-zone crop) -> (name, cosine score)
   |
   v
 RecipeStateMachine.update(verb, ingredient, t_ns)  (registry ->
   |  reference/state_machine_ref.py) -> Event(advance|wrong_order|repeat|idle|done)
   v
 EventBus (src/pipeline/events.py) -> WebSocket (src/app/server.py) -> UI
                             \-> SessionLogger jsonl  \-> overlay (src/pipeline/overlay.py)
                                                      \-> MJPEG stream / cv2 window


                        TRAINING (offline, Kaggle T4 / local CPU smoke)
 EPIC_100_train.csv + EPIC_100_video_info.csv  (src/data/annotations.py)
   |  rows filtered to candidate verbs + downloaded videos (src/data/splits.py)
   |  V1 internal split: heldout_participants=[P22,P30] act as validation
   v
 EpicVerbDataset (src/data/dataset.py)
   |  per segment: sample_indices(start,stop,T,fps) -> decode_frames
   |  -> clip float32 (T,3,H,W) in [0,1] + int label (alphabetical verb map)
   |  DataLoader -> (B,T,3,H,W)
   v
 train loop (src/train/loop.py: AdamW/SGD, cosine/step, AMP, class weights,
   |  overfit-32 ritual) -> best-by-val-mean-class-accuracy checkpoint
   v
 evaluate (src/train/eval_model.py + registry metrics) -> runs/<id>/metrics.json
   (top-1/top-5, per-class, mean-class, majority baseline, confusion PNG)
```

## Module -> concept map

| Module | Concept it uses | Study material |
|---|---|---|
| reference/enrollment_ref.py | embeddings, cosine similarity, prototypes (kNN with 1 prototype) | docs/worked/knn.md, enrollment.md |
| reference/frame_sampler_ref.py | segment-based temporal sampling (TSN) | docs/worked/frame_sampler.md |
| reference/metrics_ref.py | top-k, per-class, confusion, majority baseline | docs/worked/metrics.md |
| reference/temporal_attention_ref.py | additive attention over time, softmax pooling | docs/worked/temporal_attention.md |
| reference/state_machine_ref.py | explicit state + event rules (no learning) | docs/worked/state_machine.md |
| reference/exercises/knn.py | L2 distances, broadcasting | docs/worked/knn.md |
| reference/exercises/softmax.py | cross-entropy, gradients, numerical checks | docs/worked/softmax.md |
| reference/exercises/batchnorm.py | normalization, running stats, train/eval modes | docs/worked/batchnorm.md |
| src/models/variants.py | transfer learning (ImageNet backbones), temporal aggregation | PyTorch transfer tutorial (docs/STUDY_QUEUE.md) |
| src/pipeline/smoothers.py | EMA, voting, hysteresis (D6) | — |
| src/pipeline/window.py | sliding window, frame-index arithmetic | docs/worked/frame_sampler.md |

## The three demo tiers, same code path

| Tier | verb source | frame source | command |
|---|---|---|---|
| 1 | keyboard (UI buttons / number keys) | webcam (or --synthetic) | `python -m scripts.run_app --mode keyboard` |
| 2 | trained checkpoint | recorded clip | `python -m scripts.run_app --mode clip --clip x.mp4 --run-dir runs/<id>` |
| 3 | trained checkpoint | webcam | `python -m scripts.run_app --mode live --run-dir runs/<id>` |

Registry (src/registry.py, configs/impl.yaml) swaps any of the five core
pieces between `reference/` and the owner's `core/` at import time; the
app, training and tests all go through it.
