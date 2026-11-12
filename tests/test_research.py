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


def test_delivery_replay():
    """The simulation drives the app's delivery class: at once, hands free, held too long, replaced."""
    from scripts.holoassist_delivery_sim import replay, score

    t = np.arange(0.0, 12.0)                                  # one reading per second
    fine = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    r = replay(t, fine, np.array([0.5, 1.5, 4.5, 5.5]), 0.5, 4.0)
    assert r[0].tolist() == [0.5, 0.0, 0.0]                   # reading 0.1: delivered at once
    assert r[1].tolist() == [3.0, 1.5, 1.0]                   # held until the reading at 3 s is low
    assert np.isnan(r[2, 0]) and r[2, 2] == 3                 # replaced by the message at 5.5 s
    assert r[3].tolist() == [10.0, 4.5, 2.0]                  # first reading 4 s or more after 5.5 s
    s = score([(r, None, lambda x: np.nan_to_num(x) >= 10.0)])
    assert s["held"] == 0.75 and s["in_fine_work"] == 1 / 3 and abs(s["mean_delay"] - 2.0) < 1e-9
    assert abs(s["timeout"] - 1 / 3) < 1e-9 and abs(s["replaced"] - 1 / 3) < 1e-9


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
    assert u == 0.5 and 1 - thr < 0.4 and d >= 1.0       # a wait of 1 s or more adds no speech


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



def test_rating_study_analysis(tmp_path):
    """Result codes and files are read once per rater; cell means and differences are right."""
    import base64
    import json

    from scripts.rating_study_analyze import cell_means, load, summarize

    type_of = {"c01": "routine", "c02": "routine", "c03": "warning", "c04": "warning"}

    def rater(i, routine_now, routine_held, warn_now, warn_held):
        return {"id": f"r{i}", "sound": "yes", "answers": [
            {"c": "c01", "k": "now", "r": routine_now, "ms": 1000}, {"c": "c02", "k": "held", "r": routine_held, "ms": 1000},
            {"c": "c03", "k": "now", "r": warn_now, "ms": 1000}, {"c": "c04", "k": "held", "r": warn_held, "ms": 1000}]}

    a, b = rater(1, 2, 5, 4, 1), rater(2, 3, 4, 5, 2)
    (tmp_path / "a.json").write_text(json.dumps(a))
    code = base64.b64encode(json.dumps(b).encode()).decode()
    (tmp_path / "codes.txt").write_text(code + "\n" + code + "\nnot-a-code\n")       # the same rater twice
    got = load(tmp_path)
    assert [r["id"] for r in got] == ["r1", "r2"]
    M = np.stack([cell_means(r, type_of) for r in got])
    assert M.tolist() == [[2, 5, 4, 1], [3, 4, 5, 2]]
    s = summarize(M, boot=200)
    assert s["routine: held - now"]["mean"] == 2.0 and s["warning: held - now"]["mean"] == -3.0
    assert s["interaction: (routine held - now) - (warning held - now)"]["mean"] == 5.0
    assert s["routine: held - now"]["raters_above_0"] == 2


def test_robustness_shift_ratio():
    """Onsets that never fall in the state give ratio 0; the state read one tick earlier gives a high ratio."""
    from scripts.holoassist_robustness import family, shift_ratio, slices

    vid = np.array(["a"] * 40 + ["b"] * 60)
    sl = slices(vid)
    assert sl == [(0, 40), (40, 100)]
    state = np.tile([True, False], 50)                     # even ticks are in the state
    event = np.zeros(100, dtype=bool)
    event[1::10] = True                                    # every event on an odd tick
    r = shift_ratio(state, event, sl, np.random.default_rng(0), shifts=300)
    assert r[0] == 0.0 and r[3] == 10 and r[1] < 1 < r[2]
    r1 = shift_ratio(state, event, sl, np.random.default_rng(0), shifts=300, lag=1)
    assert 1.7 < r1[0] < 2.3                               # one tick earlier: always in the state (chance is half)
    assert family("disassemble tray table") == "tray table" and family("setup big printer") == "printer"


def test_rating_study_instruction_filter():
    from scripts.rating_study_build import clean

    assert clean("start unscrewing the top of the legs") == "Start unscrewing the top of the legs."
    assert clean("And you lift that up now please") is None            # fragment
    assert clean("put it on, on the table over there") is None         # repeated word
    assert clean("um take the screw and turn it") is None and clean("can you take the screw out now?") is None
    assert clean("too short") is None


def test_official_split_onset_f1():
    """A prediction counts if a true onset lies from 1 tick before to 2 ticks after it; each onset once."""
    from scripts.holoassist_official_split import onset_f1

    y = np.zeros(20); y[[5, 12]] = 1
    s = np.zeros(20); s[[4, 5, 14, 18]] = 1.0          # 4 -> onset 5; 5 is inside the 3 s gap; 14 -> too late for 12; 18 -> none
    f1, p, r = onset_f1(s, y, [(0, 20)], 0.5)
    assert (p, r) == (1 / 3, 1 / 2) and abs(f1 - 0.4) < 1e-9
    s2 = np.zeros(20); s2[[6, 10]] = 1.0               # 6 -> onset 5 (1 tick before); 10 -> onset 12 (2 ticks after)
    assert onset_f1(s2, y, [(0, 20)], 0.5) == (1.0, 1.0, 1.0)
