"""When to hand a message to the user: routine messages wait for the hands.

EXPERIMENTAL. This applies the pattern measured on HoloAssist
(reports/research_gap_map.md, part 6): a human instructor holds a routine
instruction back while the performer does fine-motor work, and does not
hold a correction back. Nothing shows that this helps a user; that needs
a study with participants.

Rules
    urgent message    delivered at once (a wrong-order warning).
    routine message   delivered at once when the hands are not in
                      fine-motor work (or when no hand state is known).
                      Otherwise it is held and delivered as soon as the
                      fine-motor probability falls below `fine_threshold`,
                      or after `max_hold_s` seconds, whichever comes first.
    A newer routine message replaces a held one (the old one is stale).

The class has no clock and no threads: the caller passes `now` in seconds.
`fine_threshold` and `max_hold_s` are first guesses, not tuned values.
"""

from __future__ import annotations


class HandAwareDelivery:
    def __init__(self, fine_threshold: float = 0.5, max_hold_s: float = 4.0) -> None:
        self.fine_threshold = fine_threshold
        self.max_hold_s = max_hold_s
        self._fine: float | None = None          # newest fine-motor probability
        self._held: dict | None = None           # {"kind", "text", "since"}
        self.superseded = 0                      # held messages replaced by a newer one

    def reset(self) -> None:
        self._fine, self._held, self.superseded = None, None, 0

    @property
    def held(self) -> dict | None:
        return self._held

    def _busy(self) -> bool:
        return self._fine is not None and self._fine >= self.fine_threshold

    @staticmethod
    def _out(kind: str, text: str, urgent: bool, held_s: float, reason: str) -> dict:
        return {"source": kind, "text": text, "urgent": urgent, "held_s": round(held_s, 2), "reason": reason}

    def submit(self, kind: str, text: str, urgent: bool, now: float) -> list[dict]:
        """A new message. -> the messages to deliver now (empty if it is held)."""
        if urgent:
            return [self._out(kind, text, True, 0.0, "urgent")]
        if not self._busy():
            self._held = None                    # a held older message is stale now
            return [self._out(kind, text, False, 0.0, "hands free")]
        if self._held is not None:
            self.superseded += 1
        self._held = {"kind": kind, "text": text, "since": now}
        return []

    def tick(self, fine: float | None, now: float) -> list[dict]:
        """A new hand-state reading. -> the held message, if it is released now."""
        self._fine = fine
        h = self._held
        if h is None:
            return []
        waited = now - h["since"]
        if not self._busy():
            reason = "hands free"
        elif waited >= self.max_hold_s:
            reason = "held too long"
        else:
            return []
        self._held = None
        return [self._out(h["kind"], h["text"], False, waited, reason)]
