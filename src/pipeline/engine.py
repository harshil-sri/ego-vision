"""The cooking engine: one thread pulls frames, runs scan phase or live
recognition, and publishes typed events. All UIs (web + OpenCV) are thin
consumers of engine.state() + engine.annotated_frame() + the event bus.

Phases: IDLE -> SCANNING (one ingredient at a time, UI-driven) -> COOKING
-> DONE. Transitions emit events; nothing else mutates engine state.

Decision knobs (D3a/D4/D5/D6) come from the engine config, not from code
branches the owner cannot see:
  smoother: ema | majority | hysteresis (+ alpha/k/margin, min_conf)
  ingredient_crop: show_zone (default) | whole_frame
  T, window_seconds, tick_hz, matcher_hz, idle_timeout_s

Keyboard mode (Tier 1): verbs arrive via inject_verb() from the UI; the
matcher still runs for real, so (verb, ingredient) events are genuine.
Model modes (Tier 2/3): a smoother-stabilized verb edge triggers the same
path — one code path for all tiers.
"""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import src.registry as registry
from src.pipeline.events import EventBus
from src.pipeline.recipe import load_recipe
from src.pipeline.session_log import SessionLogger
from src.pipeline.smoothers import make_smoother


@dataclass
class EngineConfig:
    """All tunables in one place (loaded from configs/app.yaml)."""
    mode: str = "keyboard"                 # keyboard | clip | live
    window_seconds: float = 2.0
    T: int = 8
    tick_hz: float = 5.0                   # recognition attempts per second
    matcher_hz: float = 2.0                # ingredient match attempts/s
    smoother: str = "ema"                  # ema | majority | hysteresis
    smoother_args: dict = field(default_factory=lambda: {"alpha": 0.3})
    min_conf: float = 0.5                  # D3a default
    recipe_prior: bool = False             # mask probs to remaining-step verbs
    prior_leak: float = 0.05               # mass illegal verbs keep (0 = hard)
    policy: str = "strict"                 # strict | optimistic (D9)
    embedder: str = "toy"                  # toy | dinov2 | resnet | clip (D7)
    matcher_threshold: float = 0.5
    ingredient_crop: str = "show_zone"     # show_zone | whole_frame (D4)
    show_zone: tuple = (0.30, 0.25, 0.40, 0.50)  # rel x, y, w, h of frame
    scan_seconds: float = 10.0
    scan_fps: float = 3.0
    idle_timeout_s: float = 45.0
    metrics_every_s: float = 2.0
    window_sample_every_s: float = 5.0   # debug drawer T-strip cadence
    record_video: bool = False
    device: str = "cpu"


IDLE, SCANNING, COOKING, DONE = "idle", "scanning", "cooking", "done"


def _dummy_frame():
    """Tiny black frame for embedder warm-up (HxWx3 uint8)."""
    import numpy as np
    return np.zeros((64, 64, 3), np.uint8)


class CookingEngine:
    """Owns the frame source, matcher, verb source, smoother, state machine."""

    def __init__(self, frame_source, config: EngineConfig,
                 recipe_path: str, verb_source=None, embedder=None,
                 bus: EventBus | None = None) -> None:
        """
        frame_source: object with start()/stop() and either latest() (camera)
            or next_frame() (clip/synthetic) returning FramePacket|None.
        verb_source: KeyboardVerbSource | ModelVerbSource (default: keyboard).
        embedder: callable for the matcher (default: config.embedder name).
        """
        self.cfg = config
        self.bus = bus or EventBus()
        self.source = frame_source
        self.recipe = load_recipe(recipe_path)
        self.recipe_path = recipe_path
        self.verb_source = verb_source  # may be None -> lazy keyboard
        self._embedder = embedder

        matcher_cls = registry.get_class("enrollment")
        self.matcher = matcher_cls(self._get_embedder,
                                   threshold=self.cfg.matcher_threshold)
        self.smoother = make_smoother(self.cfg.smoother, self._smoother_verbs(),
                                      **self._smoother_args())
        sm_cls = registry.get_class("state_machine")
        self.machine = sm_cls(self.recipe, policy=self.cfg.policy)

        self.phase = IDLE
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_annotated: np.ndarray | None = None
        self._frame_times: list[float] = []          # for measured FPS
        self._last_infer_ms: float = 0.0
        self._last_verb: tuple[str | None, float] = (None, 0.0)
        self._ingredient: tuple[str, float] = ("unknown", 0.0)
        self._warning_until: float = 0.0
        self._warning_msg: str = ""
        self._last_activity: float = time.monotonic()
        self._idle_notified_at: float = 0.0
        self._last_metrics_at: float = 0.0
        self._last_sample_at: float = 0.0
        self._stopped_published = False
        self._dropped_at_start: int = 0
        # scan-phase state
        self._scan_ingredient: str | None = None
        self._scan_frames: list[np.ndarray] = []
        self._scan_started: float = 0.0
        self._enrolled: set[str] = set()
        self.logger: SessionLogger | None = None

    # ------------------------------------------------------------ helpers

    def _get_embedder(self, frames):
        """Lazily build the embedder the first time the matcher needs it."""
        if self._embedder is None:
            from src.pipeline.embedder import make_embedder
            self._embedder = make_embedder(self.cfg.embedder, self.cfg.device)
        return self._embedder(frames)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        """Apply the ingredient crop policy (D4): show_zone rect or whole."""
        if self.cfg.ingredient_crop != "show_zone":
            return image
        H, W = image.shape[:2]
        rx, ry, rw, rh = self.cfg.show_zone
        x0, y0 = int(rx * W), int(ry * H)
        x1, y1 = int((rx + rw) * W), int((ry + rh) * H)
        return image[y0:y1, x0:x1]

    def _thumb_b64(self, image: np.ndarray, size: int = 96) -> str:
        """Small JPEG data-URI for scan_progress events (UI thumbnails)."""
        thumb = cv2.resize(image, (size, size))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""

    def _get_frame(self):
        """Camera sources expose latest(); clip/synthetic expose next_frame()."""
        if hasattr(self.source, "latest"):
            return self.source.latest(timeout=0.2)
        return self.source.next_frame()

    def _frames_pushed_per_s(self) -> float:
        """Measured frame rate over the last ~3 s of pushes."""
        now = time.monotonic()
        self._frame_times = [t for t in self._frame_times if now - t < 3.0]
        return len(self._frame_times) / 3.0 if self._frame_times else 0.0

    # ------------------------------------------------------- public API

    @property
    def is_running(self) -> bool:
        """True while the engine loop thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start source + engine loop thread + session logger."""
        if self.is_running:
            return  # already running: a repeat POST /api/session/start is a no-op
        self._stop.clear()
        self._fresh_run()
        self.source.start()
        buf = getattr(self.source, "buffer", None)
        self._dropped_at_start = buf.dropped_frames if buf else 0
        from src.pipeline.session_log import attach
        self.logger = attach(self.bus, record_video=self.cfg.record_video)
        self.logger.start()
        self._thread = threading.Thread(target=self._loop, name="engine",
                                        daemon=True)
        self._thread.start()
        # Pre-warm the embedder OFF the engine loop: the first DINOv2/CLIP
        # use downloads ~84 MB of weights, which would otherwise stall the
        # scan-phase capture for tens of seconds (UI subagent finding).
        if self.cfg.embedder != "toy":
            threading.Thread(target=self._get_embedder,
                             args=([_dummy_frame()],),
                             name="embedder-warmup", daemon=True).start()
        self.bus.publish("session_started", recipe=self.recipe.name,
                         mode=self.cfg.mode, phase=self.phase,
                         smoother=self.cfg.smoother, T=self.cfg.T,
                         embedder=self.cfg.embedder,
                         impls=registry.active_impls())

    def _publish_session_stopped(self) -> None:
        """Announce the end of a session exactly once (stop or crash)."""
        if not self._stopped_published:
            self._stopped_published = True
            self.bus.publish("session_stopped", mode=self.cfg.mode,
                             phase=self.phase)

    def stop(self) -> None:
        """Stop the loop thread and the source; close the session log."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.source.stop()
        # publish before closing the logger so the session log records its end
        self._publish_session_stopped()
        if self.logger is not None:
            self.logger.close()

    def _smoother_args(self) -> dict:
        args = dict(self.cfg.smoother_args)
        args.setdefault("min_conf", self.cfg.min_conf)
        return args

    def _smoother_verbs(self) -> list[str]:
        """Smoother indices MUST align with the probability vector it sees:
        the MODEL's verb_map order in clip/live modes (8 classes), the
        recipe's verbs in keyboard mode (verbs arrive by name, no probs)."""
        vs = getattr(self.verb_source, "verbs", None)
        return list(vs) if vs else list(self.recipe.verbs)

    def _fresh_run(self) -> None:
        """Reset per-run state so (re)starting is a clean cooking run.

        Enrolled ingredient prototypes survive (no re-scan after restart);
        the state machine, smoother and verb/ingredient readouts do not.
        """
        self.smoother = make_smoother(self.cfg.smoother, self._smoother_verbs(),
                                      **self._smoother_args())
        sm_cls = registry.get_class("state_machine")
        self.machine = sm_cls(self.recipe, policy=self.cfg.policy)
        self.phase = (COOKING
                      if self._enrolled == set(self.recipe.ingredients)
                      else IDLE)
        self._last_verb = (None, 0.0)
        self._ingredient = ("unknown", 0.0)

    # ----- scan phase -------------------------------------------------

    def begin_ingredient_scan(self, ingredient: str) -> None:
        """UI: start the 10 s capture for one ingredient."""
        if ingredient not in self.recipe.ingredients:
            self.bus.publish_error(f"unknown ingredient '{ingredient}'")
            return
        with self._lock:
            self._scan_ingredient = ingredient
            self._scan_frames = []
            self._scan_started = time.monotonic()
        self.bus.publish("scan_progress", ingredient=ingredient,
                         captured=0,
                         target=int(self.cfg.scan_seconds * self.cfg.scan_fps))

    def retake_ingredient(self, ingredient: str) -> None:
        """UI: retake button — same as begin (enroll replaces the prototype)."""
        self.begin_ingredient_scan(ingredient)

    def skip_ingredient(self) -> None:
        """UI: skip the current ingredient (marked not-enrolled)."""
        with self._lock:
            name, self._scan_ingredient = self._scan_ingredient, None
            self._scan_frames = []
        self.bus.publish("scan_progress", ingredient=name, skipped=True,
                         captured=0)

    def finish_scan(self) -> None:
        """UI: all done -> ready to cook."""
        with self._lock:
            self.phase = COOKING if len(self._enrolled) else IDLE
        self.bus.publish("scan_complete", enrolled=sorted(self._enrolled),
                         phase=self.phase)

    # ----- cooking ----------------------------------------------------

    def inject_verb(self, verb: str) -> None:
        """Keyboard mode: a verb happened NOW (UI number key / button)."""
        if verb not in self.recipe.verbs:
            self.bus.publish_error(f"unknown verb '{verb}'")
            return
        self._consume_verb(verb, 1.0)

    def _consume_verb(self, verb: str, conf: float) -> None:
        """One stabilized (verb, current ingredient) event into the machine."""
        name, score = self._ingredient
        ev = self.machine.update(verb, name, time.monotonic_ns())
        self._last_activity = time.monotonic()
        self._last_verb = (verb, conf)
        payload = dict(verb=verb, confidence=conf, ingredient=name,
                       ingredient_score=score,
                       step_index=ev.step_index,
                       next_step=(ev.next_step.verb + " " + ev.next_step.ingredient
                                  ) if ev.next_step else None,
                       message=ev.message)
        kind_map = {"advance": "step_advanced", "wrong_order": "wrong_order",
                    "repeat": "repeat", "idle": "idle_nudged", "done": "done"}
        self.bus.publish(kind_map.get(ev.kind, "idle_nudged"), **payload)
        if ev.kind == "wrong_order":
            self._warning_until = time.monotonic() + 3.0
            self._warning_msg = ev.message
        if ev.kind == "done":
            with self._lock:
                self.phase = DONE

    # ----- state snapshot ----------------------------------------------

    def state(self) -> dict:
        """Full UI snapshot (REST /api/state + WS catch-up base)."""
        with self._lock:
            cur = self.machine.current_index if self.phase in (COOKING, DONE) else None
            steps = [{"verb": s.verb, "ingredient": s.ingredient, "note": s.note,
                      "done": i < cur if cur is not None else False,
                      "current": i == cur if cur is not None else False}
                     for i, s in enumerate(self.recipe.steps)]
            scan = None
            if self._scan_ingredient is not None:
                elapsed = time.monotonic() - self._scan_started
                remaining = max(self.cfg.scan_seconds - elapsed, 0.0)
                scan = {"ingredient": self._scan_ingredient,
                        "captured": len(self._scan_frames),
                        "target": int(self.cfg.scan_seconds * self.cfg.scan_fps),
                        "remaining_s": round(remaining, 1),
                        "enrolled": sorted(self._enrolled)}
            return {
                "phase": self.phase, "mode": self.cfg.mode,
                "recipe": {"name": self.recipe.name, "steps": steps,
                           "ingredients": self.recipe.ingredients,
                           "verbs": self.recipe.verbs},
                "scan": scan if scan else {"enrolled": sorted(self._enrolled)},
                "verb": {"name": self._last_verb[0],
                         "confidence": round(self._last_verb[1], 3)},
                "ingredient": {"name": self._ingredient[0],
                               "score": round(self._ingredient[1], 3)},
                "warning": ({"message": self._warning_msg}
                            if time.monotonic() < self._warning_until else None),
                "metrics": {"fps": round(self._frames_pushed_per_s(), 1),
                            "inference_ms": round(self._last_infer_ms, 1),
                            "dropped": self._dropped(),
                            "smoother": self.cfg.smoother,
                            "min_conf": self.cfg.min_conf},
            }

    def _dropped(self) -> int:
        buf = getattr(self.source, "buffer", None)
        return (buf.dropped_frames - self._dropped_at_start) if buf else 0

    def annotated_frame(self) -> np.ndarray | None:
        """Latest frame with overlay drawn (MJPEG + recording source)."""
        return self._last_annotated

    # ------------------------------------------------------- engine loop

    def _loop(self) -> None:
        """Single thread: pull frames; run scan capture or recognition."""
        from src.pipeline.overlay import draw_overlay
        from src.pipeline.window import SlidingWindowBuffer
        window = SlidingWindowBuffer(self.cfg.window_seconds)
        next_tick = 0.0
        next_match = 0.0
        try:
            while not self._stop.is_set():
                packet = self._get_frame()
                if packet is None:
                    if self.phase != IDLE and not hasattr(self.source, "latest"):
                        break  # clip ended
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                self._frame_times.append(now)
                window.push(packet)

                # -------- scan capture
                if self._scan_ingredient is not None:
                    self._scan_capture(packet)
                elif self.phase == COOKING:
                    if now >= next_match:
                        next_match = now + 1.0 / self.cfg.matcher_hz
                        self._match_ingredient(packet.image)
                    if now >= next_tick:
                        next_tick = now + 1.0 / self.cfg.tick_hz
                        self._recognize_verb(window.sample(self.cfg.T))

                # -------- idle nudge + metrics
                if self.phase == COOKING:
                    quiet = now - self._last_activity
                    if (quiet > self.cfg.idle_timeout_s
                            and now - self._idle_notified_at > self.cfg.idle_timeout_s):
                        self._idle_notified_at = now
                        step = self.recipe.steps[self.machine.current_index]
                        self.bus.publish("idle",
                                         message=f"Still waiting for: "
                                                 f"{step.verb} {step.ingredient}",
                                         quiet_seconds=round(quiet, 1))
                if now >= self._last_metrics_at + self.cfg.metrics_every_s:
                    self._last_metrics_at = now
                    st = self.state()
                    self.bus.publish("metrics_tick", **st["metrics"],
                                     verb=self._last_verb[0],
                                     ingredient=self._ingredient[0],
                                     ingredient_score=self._ingredient[1])

                # -------- sampled-frame strip (debug drawer): the exact T
                # frames a model tick would see, as small thumbnails
                if (self.phase == COOKING
                        and now >= self._last_sample_at
                        + self.cfg.window_sample_every_s):
                    self._last_sample_at = now
                    frames = window.sample(self.cfg.T)
                    if frames:
                        self.bus.publish("window_sample",
                                         thumbs=[self._thumb_b64(p.image, 48)
                                                 for p in frames])

                # -------- overlay + record
                annotated = draw_overlay(packet.image.copy(), self)
                self._last_annotated = annotated
                if self.logger is not None and self.cfg.record_video:
                    self.logger.frame(annotated)
        except Exception as exc:  # a dead loop = frozen demo with no notice;
            # surface it and end the session cleanly instead
            self.bus.publish_error(f"engine loop crashed: {exc!r}")
            self._publish_session_stopped()

    # -------- scan capture internals

    def _scan_capture(self, packet) -> None:
        """Collect frames for the active ingredient until time/target met."""
        now = time.monotonic()
        target_n = int(self.cfg.scan_seconds * self.cfg.scan_fps)
        elapsed = now - self._scan_started
        want_n = min(int(elapsed * self.cfg.scan_fps), target_n)
        with self._lock:
            name = self._scan_ingredient
        if name is None:
            return
        if len(self._scan_frames) < want_n:
            self._scan_frames.append(self._crop(packet.image))
            self.bus.publish("scan_progress", ingredient=name,
                             captured=len(self._scan_frames), target=target_n,
                             thumb=self._thumb_b64(packet.image))
        if elapsed >= self.cfg.scan_seconds or len(self._scan_frames) >= target_n:
            with self._lock:
                frames = list(self._scan_frames)
                self._scan_ingredient = None
            if frames:
                self.matcher.enroll(name, frames)   # replaces on retake
                self._enrolled.add(name)
            self.bus.publish("scan_progress", ingredient=name,
                             captured=len(frames), target=target_n,
                             done=True, enrolled=sorted(self._enrolled))
            if self._enrolled == set(self.recipe.ingredients):
                self.bus.publish("scan_complete", enrolled=sorted(self._enrolled),
                                 all_done=True)

    # -------- cooking internals

    def _match_ingredient(self, image: np.ndarray) -> None:
        """Match the crop against prototypes; publish on CHANGE only."""
        try:
            name, score = self.matcher.match(self._crop(image))
        except Exception as exc:  # matcher failure must not kill the loop
            self.bus.publish_error(f"matcher failed: {exc}")
            return
        if name != self._ingredient[0]:
            self._ingredient = (name, score)
            self.bus.publish("ingredient_matched", ingredient=name,
                             score=round(score, 3))

    def _recognize_verb(self, window) -> None:
        """Model modes: predict -> (optional recipe prior) -> smooth -> fire."""
        if self.verb_source is None or not window:
            return
        t0 = time.monotonic()
        probs = self.verb_source.predict(window)
        self._last_infer_ms = (time.monotonic() - t0) * 1000.0
        if probs is None:  # keyboard source: verbs arrive via inject_verb
            return
        if self.cfg.recipe_prior:
            from src.pipeline.prior import apply_recipe_prior, legal_verbs
            legal = legal_verbs(self.recipe.steps, self.machine.current_index)
            # probs are ordered by the MODEL's verb_map (verb_source.verbs),
            # which need not match the recipe's verb list or alphabetical order
            model_verbs = getattr(self.verb_source, "verbs", None)
            if legal and model_verbs:
                probs = apply_recipe_prior(
                    probs, list(model_verbs), legal,
                    leak=self.cfg.prior_leak)
        verb, conf = self.smoother.update(probs, time.monotonic_ns())
        if verb is not None and verb != self._last_verb[0]:
            self.bus.publish("verb_predicted", verb=verb,
                             confidence=round(conf, 3))
            self._consume_verb(verb, conf)
