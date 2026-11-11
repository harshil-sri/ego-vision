"""Train the hand-state head for the app, and check it on kitchen video.

The research pilot (reports/research_gap_map.md, part 5) found that a
small classifier on the frozen VideoMAE vector can tell fine-motor hand
states from free hands on HoloAssist, and that human instructors avoid
speaking during the fine-motor ones. The app's verb model computes the
same VideoMAE vector at every tick. This script trains that classifier on
all downloaded HoloAssist videos and saves it, so the app can report "are
the hands busy with fine work" at no extra cost
(src/pipeline/hand_state.py).

States: 20 HoloAssist verbs, other verb, idle. Groups:
    fine   screw, unscrew, insert
    free   idle, hold, place, inspect
    other  the rest

Check on kitchen video (EPIC-KITCHENS held-out and training ticks from
scripts.extract_sliding; the head never saw them). EPIC has no hand-state
labels, so the check is indirect: at ticks with NO annotated action the
"free" probability should be higher than at ticks inside an action.
Reported as ROC AUC (0.5 = the head sees no difference).

Usage:
    python -m scripts.train_hand_state_head
Outputs: runs/hand_state_holoassist/{last.pt, metrics.json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.egoextra_from_video import FINE, FREE, GROUPS, holoassist_ticks
from scripts.holoassist_from_video import fit_head, predict_head
from src.paths import resolve_data_path

EPIC_SLIDING = "/mnt/ssd/epic-kitchens/features/videomae_ssv2_sliding"
DEMO_VERBS = sorted(["take", "put", "wash", "cut", "open", "close", "mix", "pour"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--out", default="runs/hand_state_holoassist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    X, st, _, names, _ = holoassist_ticks(args.window)                 # raw features: the app feeds raw vectors
    head = fit_head(X, st, len(names), args.device)
    gid = [0 if n in FINE else 1 if n in FREE else 2 for n in names]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": head.state_dict(), "head": head.config(), "states": names,
                "group_of_state": gid, "groups": GROUPS, "window_s": args.window,
                "encoder": "videomae_base_ssv2", "trained_on": f"HoloAssist, {len(st)} ticks"}, out / "last.pt")

    # ---- indirect check on EPIC-KITCHENS -------------------------------
    from src.data.annotations import load_video_info
    from src.data.window_labels import load_actions, point_labels
    ann = Path(resolve_data_path(yaml.safe_load(Path("configs/data.yaml").read_text())["annotations_dir"]))
    fd = Path(resolve_data_path(EPIC_SLIDING))
    meta = pd.read_csv(fd / "meta.csv")
    feats = np.load(fd / f"feats_{args.window}.npy").astype(np.float32)
    fps = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    label, _, _ = point_labels(meta["video_id"].to_numpy(str), meta["frame"].to_numpy(), fps,
                               args.window, DEMO_VERBS, load_actions(ann, DEMO_VERBS), 0.5)
    V = len(DEMO_VERBS)                                                # label: 0..V-1 demo verb, V other verb, V+1 background
    P = predict_head(head, feats, args.device)
    G = np.stack([P[:, np.array(gid) == k].sum(1) for k in range(3)], 1)   # (n, 3) fine, free, other
    background = label == V + 1
    res = {"holoassist_ticks": int(len(st)), "epic_ticks": int(len(label)),
           "epic_background_share": round(float(background.mean()), 3),
           "epic_auc_free_prob_background_vs_action": round(float(roc_auc_score(background, G[:, 1])), 4),
           "epic_mean_group_prob": {
               "background": [round(float(x), 3) for x in G[background].mean(0)],
               "other verb": [round(float(x), 3) for x in G[label == V].mean(0)],
               **{v: [round(float(x), 3) for x in G[label == i].mean(0)] for i, v in enumerate(DEMO_VERBS)}}}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("columns of epic_mean_group_prob:", GROUPS)


if __name__ == "__main__":
    main()
