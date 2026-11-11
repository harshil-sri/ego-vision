"""Tests for the pure functions of the research-pilot scripts.

No dataset is needed: every test builds a tiny input by hand, so the
expected output can be checked on paper.
"""

import numpy as np

from scripts.egoextra_hands import DT, to_states
from scripts.egoextra_speech import expert_turns, load_transcript
from scripts.holoassist_download_subset import _octal
from scripts.holoassist_from_video import run_lengths


def test_load_transcript_parses_times_and_speakers(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("1 [00:00:02,000 --> 00:00:02,560] E: At that time,\n"
                 "2 [00:01:10,500 --> 00:01:12,480] A: Ok, this here?\n"
                 "not a transcript line\n", encoding="utf-8")
    u = load_transcript(p)
    assert u.shape == (2, 3)
    assert np.allclose(u[0], [2.0, 2.56, 1.0])           # expert
    assert np.allclose(u[1], [70.5, 72.48, 0.0])         # trainee


def test_expert_turns_merge_and_unasked():
    # columns: start, end, is_expert
    u = np.array([
        [0.0, 1.0, 1],      # turn 1: nobody spoke before -> unasked
        [1.5, 3.0, 1],      # 0.5 s after turn 1 ended -> same turn (gap < 2 s)
        [10.0, 11.0, 0],    # trainee asks
        [12.0, 12.5, 1],    # turn 2: trainee spoke 1 s before -> asked
        [30.0, 33.0, 1],    # turn 3: trainee silent for 19 s -> unasked
    ], dtype=float)
    starts, unasked, speech = expert_turns(u, turn_gap=2.0, ask_window=5.0)
    assert starts.tolist() == [0.0, 12.0, 30.0]
    assert unasked.tolist() == [True, False, True]
    assert np.allclose(speech, [2.5, 0.5, 3.0])          # seconds of expert speech per turn


def test_to_states_thirds_and_missing():
    # 300 bins: slow, medium, fast blocks of 100 bins, then 50 bins with no hand
    speed = np.r_[np.full(100, 0.1), np.full(100, 1.0), np.full(100, 5.0), np.full(50, np.nan)]
    state, recent = to_states(speed)
    assert len(state) == 350
    # well inside each block (the 1 s average needs 10 bins to settle)
    assert state[50] == 1 and state[150] == 2 and state[250] == 3      # still, mid, moving
    assert state[-1] == 0 and np.isnan(recent[-1])                     # none
    assert abs(recent[250] - 5.0) < 1e-9
    assert DT == 0.1


def test_run_lengths_reset_on_change_and_new_video():
    arg = np.array([3, 3, 3, 1, 1, 1, 1])
    vid = np.array(["a", "a", "a", "a", "a", "b", "b"])
    #                same state continues: 0,1,2; change: 0,1; new video: 0,1
    assert run_lengths(arg, vid).tolist() == [0, 1, 2, 0, 1, 0, 1]


def test_tar_octal_field():
    assert _octal(b"0000644\0") == 0o644
    assert _octal(b"00000001750 ") == 1000               # 1750 octal = 1000 bytes
    assert _octal(b"\0" * 12) == 0


def test_wait_policy_rates():
    """Part 8: unneeded / failed rates of the wait and gate policies on four hand-made mistakes."""
    from scripts.holoassist_wait_policy import DELAYS, THRESH, best, rates

    silent = np.array([1, 1, 0, 0])
    D = np.array([1.0, 4.0, 9.0, 9.0])                   # self-corrected ones last 1 s and 4 s
    L = np.array([np.nan, np.nan, 0.6, 3.0])             # instructor corrects after 0.6 s and 3 s
    p = np.array([0.1, 0.6, 0.4, 0.9])                   # model: probability that the instructor corrects
    U, F = rates(p, silent, D, L)
    k2 = int(np.flatnonzero(DELAYS == 2.0)[0])
    assert U[0, 0] == 1.0 and F[0, 0] == 0.0             # speak at once: always unneeded, never late
    assert U[0, k2] == 0.5 and F[0, k2] == 0.5           # wait 2 s: one of each
    i = int(np.flatnonzero(np.isclose(THRESH, 0.5))[0])  # gate fires when p > 0.5
    assert U[i, -1] == 0.5 and F[i, -1] == 0.5           # gate alone: speaks on mistakes 2 and 4
    assert best(U, F, "wait d", 0.0)[0] == 1.0           # never late only by not waiting
    u, thr, d = best(U, F, "gate + wait d", 0.0)         # a gate at p > 0.3 catches both corrections
    assert u == 0.5 and np.isinf(d) and 1 - thr < 0.4


def test_need_controlled_episodes():
    """Part 6 episode builders on a hand-made video."""
    from scripts.holoassist_need_controlled import episodes_a, episodes_b, group_ids

    top = ["screw", "grab"]                              # state ids: 0 screw, 1 grab, 2 other, 3 idle
    gid = group_ids(top)
    assert gid.tolist() == [1, 3, 3, 0]                  # fine, other action, other action, idle
    V = {"lo": 0.0, "hi": 30.0,
         "edges": np.array([0.0, 10.0, 11.0, 30.0]),     # idle 0-10, screw 10-11, idle 11-30
         "state": np.array([3, 0, 3]),
         "coarse": np.array([10.0]),                     # a step boundary at 10 s
         "onsets": {"high_level": np.array([11.2])},     # next instruction 1.2 s later
         "wrong": [(20.0, 22.0, "screw")], "corr": np.array([21.2])}
    a = episodes_a(V, gid)
    # ticks at 10.0, 10.5, 11.0 s; the instruction falls in the tick that starts at 11.0
    assert a[:, 0].tolist() == [0.0, 0.5, 1.0]
    assert a[:, 1].tolist() == [1, 1, 0]                 # fine, fine, idle
    assert a[:, 2].tolist() == [0, 0, 1]
    b = episodes_b(V)
    # ticks at 20.0, 20.5, 21.0 s, all before the action ends at 22 s; correction in the last tick
    assert b[:, 0].tolist() == [0.0, 0.5, 1.0]
    assert b[:, 1].tolist() == [1, 1, 1] and b[:, 2].tolist() == [1, 1, 1]
    assert b[:, 3].tolist() == [0, 0, 1]

