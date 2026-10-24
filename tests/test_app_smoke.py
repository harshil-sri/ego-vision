"""Tier 1 smoke test: full app flow on SYNTHETIC frames, no camera, no
model (keyboard verbs). Automates the WP5 acceptance: scan phase completes,
recipe steps advance, wrong order is flagged, session is logged.

The ingredient identity is INJECTED (engine._ingredient) instead of relying
on matcher accuracy — matcher quality has its own tests; this test covers
the ENGINE wiring: events, phases, state machine, session log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.pipeline.engine import DONE, COOKING, CookingEngine, EngineConfig
from src.pipeline.events import EventBus
from src.pipeline.window import SyntheticFrameSource

RECIPE = "recipes/example_salad.yaml"


@pytest.fixture()
def engine():
    cfg = EngineConfig(mode="keyboard", embedder="toy",
                       matcher_threshold=0.3,
                       scan_seconds=0.4, scan_fps=6.0,
                       tick_hz=10.0, matcher_hz=4.0,
                       metrics_every_s=0.5, idle_timeout_s=3600)
    eng = CookingEngine(SyntheticFrameSource(fps=20), cfg, RECIPE,
                        bus=EventBus())
    eng.start()
    yield eng
    eng.stop()


def events_of(recorder, kind):
    return [e for e in recorder if e.kind == kind]


def wait_for(predicate, timeout=6.0, step=0.05):
    """Poll until predicate() is truthy; returns its final value."""
    deadline = time.monotonic() + timeout
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(step)
        result = predicate()
    return result


def test_tier1_full_flow(engine, tmp_path):
    recorder: list = []
    engine.bus.subscribe(recorder.append)

    # -- session log exists and session_started was published (the fixture
    #    starts the engine BEFORE this test subscribes, so session_started
    #    lives in the bus history, not the recorder)
    assert any(e.kind == "session_started" for e in engine.bus.history())
    assert engine.logger.path is not None and engine.logger.path.exists()

    # -- scan phase: enroll all three ingredients
    for ingredient in engine.recipe.ingredients:
        engine.begin_ingredient_scan(ingredient)
        ok = wait_for(lambda ing=ingredient: any(
            e.kind == "scan_progress" and e.payload.get("done")
            and e.payload["ingredient"] == ing for e in recorder))
        assert ok, f"scan for {ingredient} did not finish"
    engine.finish_scan()
    assert wait_for(lambda: engine.phase == COOKING)
    assert wait_for(lambda: events_of(recorder, "scan_complete"))

    # -- wrong order first: pour dressing (a LATER step) before take tomato
    engine._ingredient = ("dressing", 0.9)
    engine.inject_verb("pour")
    assert wait_for(lambda: events_of(recorder, "wrong_order"))

    # -- walk the recipe in order
    for step in engine.recipe.steps:
        engine._ingredient = (step.ingredient, 0.9)
        engine.inject_verb(step.verb)
    assert wait_for(lambda: engine.phase == DONE)
    assert wait_for(lambda: events_of(recorder, "done"))
    kinds = [e.kind for e in events_of(recorder, "step_advanced")]
    assert len(events_of(recorder, "wrong_order")) >= 1
    assert len(kinds) == len(engine.recipe.steps) - 1  # last step -> done

    # -- metrics ticks flowed
    assert wait_for(lambda: events_of(recorder, "metrics_tick"))

    # -- annotated frames were produced (overlay path exercised)
    assert wait_for(lambda: engine.annotated_frame() is not None)

    # -- session jsonl is valid JSON lines with the expected kinds
    lines = engine.logger.path.read_text().strip().splitlines()
    logged = [json.loads(line) for line in lines]
    logged_kinds = {entry["kind"] for entry in logged}
    assert {"session_started", "scan_progress", "scan_complete",
            "step_advanced", "wrong_order", "done"} <= logged_kinds


def test_scan_retake_replaces_prototype(engine):
    engine.begin_ingredient_scan("tomato")
    assert wait_for(lambda: "tomato" in engine._enrolled)
    engine.retake_ingredient("tomato")  # must not raise or duplicate
    assert wait_for(lambda: len(
        [e for e in engine.matcher.names if e == "tomato"]) == 1)


def test_unknown_verb_publishes_error(engine):
    engine.inject_verb("teleport")
    assert wait_for(lambda: [e for e in engine.bus.history()
                             if e.kind == "error"])


def test_stop_then_restart_runs_new_loop(engine):
    """Regression: stop() used to leave _stop set, so a following start()
    spawned a loop that exited instantly (dead "restarted" session)."""
    recorder: list = []
    engine.bus.subscribe(recorder.append)

    engine.stop()
    assert events_of(recorder, "session_stopped"), "stop must announce itself"

    recorder.clear()
    engine.start()
    assert wait_for(lambda: events_of(recorder, "session_started"))
    assert wait_for(lambda: events_of(recorder, "metrics_tick")), \
        "restarted engine loop is not producing events"


def test_double_start_is_noop(engine):
    import threading
    engine.start()  # already running from the fixture
    time.sleep(0.3)
    loops = [t for t in threading.enumerate() if t.name == "engine"]
    assert len(loops) == 1, "second start() must not spawn a second loop"
    assert wait_for(lambda: events_of(engine.bus.history(), "metrics_tick"))


def test_restart_after_enrollment_resumes_cooking(engine):
    """Enrolled prototypes survive a restart; recipe progress resets."""
    for ingredient in engine.recipe.ingredients:
        engine.begin_ingredient_scan(ingredient)
        assert wait_for(lambda ing=ingredient: ing in engine._enrolled)
    engine.finish_scan()
    assert wait_for(lambda: engine.phase == COOKING)

    engine.stop()
    engine.start()
    assert wait_for(lambda: engine.phase == COOKING), \
        "fully-enrolled restart should resume in COOKING, not force a re-scan"
    assert engine.machine.current_index == 0
