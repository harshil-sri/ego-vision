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
