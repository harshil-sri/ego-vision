"""Picture rows for the human relabelling of Ego-EXTRA and WTaG moments.

A row is the 4 frames of the 2 s that end at a moment (0.5 s apart), the
same view the first labeller had (scripts.label_check_kit):

    Ego-EXTRA: central 500 x 500 of the 704 x 704 view, shown at 300 x 300,
               read with an ffmpeg seek (exact for these files)
    WTaG:      384 x 216, read in order from the start of the file,
               because a seek in these files lands on the last key frame

This module reads many moments at once: Ego-EXTRA in parallel, WTaG one
pass per recording instead of one pass per moment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.label_check_kit import SETS, frame

OFFSETS = (1.75, 1.25, 0.75, 0.25)


def _egoextra_row(root: Path, rec: str, t: float) -> np.ndarray:
    cfg = SETS["egoextra"]
    video = root / "videos" / f"{rec}.mp4"
    return np.concatenate([frame(video, t - o, cfg["vf"], cfg["size"]) for o in OFFSETS], 1)


def _wtag_rows(root: Path, rec: str, times: list[float]) -> list[np.ndarray]:
    size = SETS["wtag"]["size"]
    cap = cv2.VideoCapture(str(root / "videos" / f"{rec}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want: dict[int, list[tuple[int, int]]] = {}
    for i, t in enumerate(times):
        for j, o in enumerate(OFFSETS):
            want.setdefault(int(round(max(t - o, 0) * fps)), []).append((i, j))
    out = [[np.zeros((*size, 3), np.uint8) for _ in OFFSETS] for _ in times]
    for n in range(max(want) + 1):
        if n in want:
            ok, f = cap.read()
            if ok:
                for i, j in want[n]:
                    out[i][j] = cv2.resize(f, (size[1], size[0]))
        elif not cap.grab():
            break
    cap.release()
    return [np.concatenate(r, 1) for r in out]


def rows(dataset: str, root: str | Path, moments: list[tuple[str, float]]) -> list[np.ndarray]:
    """One picture row per (recording, time in s), in the order given."""
    root = Path(root)
    if dataset == "egoextra":
        with ThreadPoolExecutor(12) as ex:
            return list(ex.map(lambda m: _egoextra_row(root, m[0], m[1]), moments))
    by_rec: dict[str, list[int]] = {}
    for i, (rec, _) in enumerate(moments):
        by_rec.setdefault(rec, []).append(i)
    res: list[np.ndarray | None] = [None] * len(moments)

    def one(rec: str) -> None:
        idx = by_rec[rec]
        for i, r in zip(idx, _wtag_rows(root, rec, [moments[i][1] for i in idx])):
            res[i] = r

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, by_rec))
    return res  # type: ignore[return-value]


def duration(root: str | Path, rec: str) -> float:
    cap = cv2.VideoCapture(str(Path(root) / "videos" / f"{rec}.mp4"))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps if fps else 0.0
