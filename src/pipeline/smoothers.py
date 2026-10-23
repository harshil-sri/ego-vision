"""Live prediction smoothing (decision D6 — implemented as config choice):
EMA of probabilities, majority vote over the last k windows, hysteresis.

Contract (brief section 5.2):
    Smoother.update(probs: np.ndarray (num_verbs,), t_ns: int)
        -> (verb: str | None, confidence: float)

All smoothers are stateful and single-threaded (the engine calls them from
its one inference loop). "None" means "no confident verb this tick" — the
confidence threshold (D3a) is applied INSIDE update via min_conf.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class EmaSmoother:
    """Exponential moving average of the probability vector (D6 default).

    ema = alpha * probs + (1 - alpha) * ema   # (C,) both sides
    Verb = argmax(ema); None while its value stays under min_conf.
    """

    def __init__(self, verbs: list[str], alpha: float = 0.3,
                 min_conf: float = 0.5) -> None:
        self.verbs = verbs
        self.alpha = float(alpha)
        self.min_conf = float(min_conf)
        self.ema: np.ndarray | None = None   # (C,) float

    def update(self, probs: np.ndarray, t_ns: int):
        """probs (C,) float; output (verb name or None, smoothed confidence)."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.ema is None:
            self.ema = probs.copy()
        else:
            self.ema = self.alpha * probs + (1 - self.alpha) * self.ema  # (C,)
        i = int(np.argmax(self.ema))
        conf = float(self.ema[i])
        return (self.verbs[i], conf) if conf >= self.min_conf else (None, conf)


class MajorityVoteSmoother:
    """Majority vote of the argmax verb over the last k windows.

    k=1 degenerates to no smoothing. Ties resolve to the alphabetically
    smallest verb (np.bincount + argmax over verb indices, verbs sorted).
    """

    def __init__(self, verbs: list[str], k: int = 5,
                 min_conf: float = 0.5) -> None:
        self.verbs = sorted(verbs)
        self.k = int(k)
        self.min_conf = float(min_conf)
        self._votes: deque[int] = deque(maxlen=self.k)

    def update(self, probs: np.ndarray, t_ns: int):
        probs = np.asarray(probs, dtype=np.float64)
        idx = int(np.argmax(probs))
        self._votes.append(idx)
        counts = np.bincount(np.array(self._votes, dtype=np.int64),
                             minlength=len(self.verbs))       # (C,) votes
        best = int(counts.argmax())
        conf = float(counts[best] / len(self._votes))         # vote share
        return (self.verbs[best], conf) if conf >= self.min_conf else (None, conf)


class HysteresisSmoother:
    """Switch verbs only when a challenger beats the incumbent by `margin`
    (or when the incumbent itself collapses below `drop`).

    Classic debounce for recognition streams: fast to lock on, slow to
    flip-flop. Confidence returned is the incumbent's EMA value.
    """

    def __init__(self, verbs: list[str], alpha: float = 0.3, margin: float = 0.15,
                 drop: float = 0.35, min_conf: float = 0.5) -> None:
        self.verbs = verbs
        self.alpha = alpha
        self.margin = margin
        self.drop = drop
        self.min_conf = min_conf
        self.ema: np.ndarray | None = None
        self.current: str | None = None

    def update(self, probs: np.ndarray, t_ns: int):
        probs = np.asarray(probs, dtype=np.float64)
        if self.ema is None:
            self.ema = probs.copy()
        else:
            self.ema = self.alpha * probs + (1 - self.alpha) * self.ema
        i = int(np.argmax(self.ema))
        conf = float(self.ema[i])
        candidate = self.verbs[i]
        if self.current is None:
            if conf >= self.min_conf:
                self.current = candidate
        elif candidate != self.current:
            # Challenger must beat the incumbent's OWN smoothed value by margin.
            inc_i = self.verbs.index(self.current)
            if conf - float(self.ema[inc_i]) >= self.margin:
                self.current = candidate
        elif conf < self.drop:
            self.current = None
        if self.current is None:
            return None, conf
        cur_conf = float(self.ema[self.verbs.index(self.current)])
        return (self.current, cur_conf) if cur_conf >= self.min_conf else (None, cur_conf)


def make_smoother(name: str, verbs: list[str], **kwargs):
    """Factory: 'ema' | 'majority' | 'hysteresis' (config string -> object)."""
    table = {"ema": EmaSmoother, "majority": MajorityVoteSmoother,
             "hysteresis": HysteresisSmoother}
    if name not in table:
        raise ValueError(f"unknown smoother '{name}'; valid: {sorted(table)}")
    return table[name](verbs, **kwargs)
