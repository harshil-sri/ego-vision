"""Typed event bus: every state change in the app flows through here as a
JSON-serializable event; subscribers (WebSocket pusher, session logger,
UI state snapshot) receive everything.

Event kinds (from the build brief, WP5):
  session_started, scan_progress, scan_complete, ingredient_matched,
  verb_predicted, step_advanced, wrong_order, repeat, idle, done,
  metrics_tick, error, app_state
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


def now_ns() -> int:
    """Monotonic-ish wall clock in nanoseconds (matches FramePacket's
    time.monotonic_ns source closely enough for logging; events also carry
    wall-clock `at` for human reading)."""
    return time.monotonic_ns()


@dataclass
class Event:
    """One application event.

    kind: str (list above); payload: JSON-serializable dict; at: ISO wall
    time string; t_ns: monotonic ns timestamp for ordering/latency math.
    """
    kind: str
    payload: dict = field(default_factory=dict)
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S",
                                                          time.localtime()))
    t_ns: int = field(default_factory=now_ns)

    def to_json(self) -> str:
        """One-line JSON for the WebSocket and the .jsonl session log."""
        return json.dumps({"kind": self.kind, "at": self.at,
                           "t_ns": self.t_ns, "payload": self.payload},
                          ensure_ascii=False)


class EventBus:
    """Thread-safe publish/subscribe with a bounded replay history.

    publish() never blocks on a slow subscriber: handlers are called
    synchronously but must be quick (pushers queue; loggers write line by
    line). history() replays the last N events so a newly connected
    WebSocket can catch up.
    """

    def __init__(self, history_size: int = 256) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self._history: deque[Event] = deque(maxlen=history_size)

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        """handler(event) is called for every event from now on."""
        self._subscribers.append(handler)

    def publish(self, kind: str, **payload) -> Event:
        """Create an event, store it, deliver to subscribers; return it."""
        event = Event(kind=kind, payload=payload)
        self._history.append(event)
        for handler in list(self._subscribers):  # copy: allow unsub via raise
            handler(event)
        return event

    def history(self, n: int | None = None) -> list[Event]:
        """Most recent events, oldest first (for WebSocket catch-up)."""
        items = list(self._history)
        return items if n is None else items[-n:]

    def publish_error(self, message: str, **extra) -> Event:
        return self.publish("error", message=message, **extra)
