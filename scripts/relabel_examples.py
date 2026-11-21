"""Draw candidate moments for the guide and practice of the relabelling site.

The guide shows labelled examples, and the practice round gives feedback.
Neither may show a moment that the labellers will label. So the
candidates are random seconds in the same recordings, at least 10 s from
every moment of the sample. WTaG has few screw/insert moments, so some
WTaG candidates are drawn 10 to 25 s away from a moment the first
labeller marked S (the same activity, often still going on).

The candidates are written as contact sheets (numbered picture rows) to
runs/label_check/relabel/candidates/ (not in git: dataset frames).
A person looks at them and writes the clear cases, with the reason, into
study/label/examples.json. Only recording ids and times go into git.

Usage:  python -m scripts.relabel_examples --dataset egoextra --root C:/ssd-data/ego-extra
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.relabel_frames import duration, rows

GAP = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["egoextra", "wtag"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--near_s", type=int, default=24, help="WTaG only: candidates near S moments")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="runs/label_check/relabel/candidates")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lab = pd.read_csv(f"reports/{a.dataset}_hand_labels.csv", dtype={"recording": str})
    taken = lab.groupby("recording").t_s.apply(np.asarray).to_dict()
    dur = {r: duration(a.root, r) for r in taken}
    far = lambda r, t: 5 < t < dur[r] - 2 and np.abs(taken[r] - t).min() >= GAP

    cands: list[tuple[str, float, str]] = []
    recs = sorted(taken)
    while len(cands) < a.n:
        r = recs[rng.integers(len(recs))]
        t = round(float(rng.uniform(0, dur[r])), 2)
        if far(r, t):
            cands.append((r, t, "random"))
    if a.dataset == "wtag":
        s = lab[lab.label == "S"].reset_index(drop=True)
        tries = 0
        while sum(c[2] == "near_S" for c in cands) < a.near_s and tries < 5000:
            tries += 1
            m = s.iloc[rng.integers(len(s))]
            t = round(float(m.t_s + rng.choice([-1, 1]) * rng.uniform(GAP, 25)), 2)
            if far(m.recording, t):
                cands.append((m.recording, t, "near_S"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pics = rows(a.dataset, a.root, [(r, t) for r, t, _ in cands])
    per = 8
    for p in range(0, len(pics), per):
        sheet = []
        for i in range(p, min(p + per, len(pics))):
            im = pics[i].copy()
            cv2.rectangle(im, (0, 0), (70, 34), (0, 0, 0), -1)
            cv2.putText(im, str(i + 1), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            sheet.append(im)
        img = np.concatenate(sheet, 0)
        img = cv2.resize(img, (1000, int(img.shape[0] * 1000 / img.shape[1])))
        cv2.imwrite(str(out / f"{a.dataset}_{p // per + 1:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    (out / f"{a.dataset}_candidates.json").write_text(json.dumps(
        [{"n": i + 1, "recording": r, "t_s": t, "source": s} for i, (r, t, s) in enumerate(cands)], indent=1))
    print(f"{len(cands)} candidates, {(len(pics) + per - 1) // per} sheets in {out}")


if __name__ == "__main__":
    main()
