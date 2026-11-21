"""Tests for scripts/timing_audit.py that need no dataset (synthetic video)."""

from __future__ import annotations

import numpy as np

from scripts.timing_audit import counts, read_speech

# one video, 0..100 s: idle 0-40 (state 3 -> group 0), fine 40-70 (state 0 -> group 1), other 70-100 (state 2 -> group 3)
V = {"edges": np.array([0.0, 40.0, 70.0, 100.0]), "state": np.array([3, 0, 2])}
GID = np.array([1, 2, 3, 0])          # state id -> hand-state group (0 idle, 1 fine, 2 hold, 3 other)


def test_counts_observed_and_chance():
    t = np.array([10.0, 20.0, 50.0, 80.0, 150.0])          # the last one is outside the window
    obs, exp = counts(V, GID, t, 0.0, 100.0, np.random.default_rng(0).uniform(0, 1, 4000))
    assert obs.tolist() == [2, 1, 0, 1]
    # chance follows the share of time: 40%, 30%, 0%, 30% of 4 onsets
    assert np.allclose(exp, [1.6, 1.2, 0.0, 1.2], atol=0.08)


def test_counts_empty():
    obs, exp = counts(V, GID, np.array([]), 0.0, 100.0, np.zeros(3))
    assert obs.sum() == 0 and exp.sum() == 0


def test_read_speech(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("video,t,end\nA,1.5,30\nA,0.5,40\nB,,20\n", encoding="utf-8")
    times, ends = read_speech(p)
    assert times["A"].tolist() == [0.5, 1.5] and len(times["B"]) == 0
    assert ends == {"A": 40.0, "B": 20.0}
