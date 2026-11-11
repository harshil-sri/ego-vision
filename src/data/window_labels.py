"""Labels for sliding windows (what the live app sees), from EPIC annotations.

A window is "the last W seconds" ending at frame f. Its label is decided at
ONE reference moment inside it, by default the window centre:

    t_ref = f - ref * W * fps          (ref = 0.5 -> centre)

    label = the demo verb of the annotated action that covers t_ref
            V       if only other verbs cover t_ref        ("other")
            V + 1   if no annotated action covers t_ref    ("background")

Why the centre: the model then sees what led up to the moment and what
followed it. The price is delay: a decision about time t is made W/2
seconds later (1 s for a 2 s window).

When several actions cover t_ref (EPIC actions overlap), a demo verb wins
over other verbs, and among demo verbs the action that started last wins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import NONE_VERB as NONE   # merged "other verb" + "background" class


def load_actions(ann_dir, verbs: list[str]) -> pd.DataFrame:
    """All annotated actions with a demo-verb id column.

    Output columns: video_id, start_frame, stop_frame, verb_id
    (0..V-1 for the demo verbs in `verbs` order, V for any other verb).
    """
    ann_dir = Path(ann_dir)
    df = pd.read_csv(ann_dir / "EPIC_100_train.csv",
                     usecols=["video_id", "start_frame", "stop_frame", "verb_class", "noun"])
    classes = pd.read_csv(ann_dir / "EPIC_100_verb_classes.csv")
    key = dict(zip(classes["id"].astype(int), classes["key"].astype(str)))
    idx = {v: i for i, v in enumerate(verbs)}
    df["verb_id"] = [idx.get(key[int(c)], len(verbs)) for c in df["verb_class"]]
    return df


def point_labels(video: np.ndarray, frame: np.ndarray, fps: dict, window_s: float,
                 verbs: list[str], actions: pd.DataFrame, ref: float = 0.5):
    """Label every window.

    Inputs:
        video (N,) str, frame (N,) int: last frame of each window
        fps: dict video_id -> frames per second
        verbs: the demo verbs (WITHOUT "none"), in label order
        actions: output of load_actions
    Outputs:
        label (N,) int64: 0..V-1 demo verb, V other verb, V+1 background
        any_verb (N, V) bool: an action with that demo verb overlaps the
            window anywhere (used for a lenient precision)
        segs: list of (video, start_frame, stop_frame, verb_id) for every
            demo-verb action in these videos
    """
    V = len(verbs)
    label = np.full(len(frame), V + 1, dtype=np.int64)
    any_verb = np.zeros((len(frame), V), dtype=bool)
    segs = []
    for v in np.unique(video):
        m = np.flatnonzero(video == v)
        d = actions[actions["video_id"] == v]
        s = d["start_frame"].to_numpy(); e = d["stop_frame"].to_numpy(); c = d["verb_id"].to_numpy()
        if len(d) == 0:
            continue
        wlen = int(round(window_s * fps[v]))
        f1 = frame[m][:, None]; f0 = f1 - wlen                              # (n, 1)
        t_ref = f1 - ref * wlen
        cover = (s[None] <= t_ref) & (e[None] >= t_ref)                      # (n, actions)
        demo = cover & (c[None] < V)
        # among covering demo actions take the latest start; else other; else background
        start_if_demo = np.where(demo, s[None], -1)
        best = start_if_demo.argmax(1)
        has_demo, has_any = demo.any(1), cover.any(1)
        label[m] = np.where(has_demo, c[best], np.where(has_any, V, V + 1))
        overlap = (s[None] <= f1) & (e[None] >= f0)
        for k in range(V):
            if (c == k).any():
                any_verb[m, k] = overlap[:, c == k].any(1)
        segs += [(v, int(a), int(b), int(k)) for a, b, k in zip(s, e, c) if k < V]
    return label, any_verb, segs
