"""Train the verb head on SLIDING windows, with a "none" class.

scripts.train_head trains on clips cut exactly at each action. The live app
shows the model something else: the last W seconds, at every tick, with or
without an action in them. This trainer uses those windows.

Data: sliding-window VideoMAE features from scripts.extract_sliding
    feats_<W>.npy (N, 768), meta.csv (video_id, frame, fps)
Labels (src/data/window_labels.py, read at the window centre): the 8 demo
verbs plus "none" (another verb, or no annotated action). Class ids are
alphabetical over all 9 names, so "none" is id 3.
Split: whole participants held out (P22 + P30), as everywhere else.

Options that matter:
    --mix-trimmed   also train on the trimmed action clips
                    (scripts.extract_features, mean of the 3 crop vectors).
                    They are clean examples of each verb.
    --mix-holoassist  also train on HoloAssist ticks (other dataset, other
                    tasks; features from scripts.holoassist_video_features).
                    "idle": only ticks labelled idle, as "none".
                    "all": also grab -> take, place -> put, open, close,
                    and every other HoloAssist verb -> "none".
    --none-weight   multiply the loss weight of "none". Above 1: fewer
                    false fires, more missed actions. Below 1: the reverse.

Outputs in --out (loadable by the app, same format as scripts.train_head):
    last.pt, verb_map.json, config.yaml, log.csv, metrics.json
The final epoch is saved. No epoch is picked on the validation set.

Usage:
    python -m scripts.train_window_head --window 2.0
    python -m scripts.eval_sliding --run-dir runs/videomae_window_w2.0 --feats /mnt/ssd/epic-kitchens/features/videomae_ssv2_sliding --window 2.0 --rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src import registry
from src.paths import resolve_data_path

DEMO_VERBS = sorted(["take", "put", "wash", "cut", "open", "close", "mix", "pour"])
TRIMMED = "/mnt/ssd/epic-kitchens/features/videomae_ssv2_T16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/mnt/ssd/epic-kitchens/features/videomae_ssv2_sliding")
    ap.add_argument("--window", type=float, default=2.0, help="which window length of --feats to train on")
    ap.add_argument("--out", default=None, help="default: runs/videomae_window_w<window>")
    ap.add_argument("--ref", type=float, default=0.5, help="label moment inside the window (0.5 = centre)")
    ap.add_argument("--mix-trimmed", action="store_true")
    ap.add_argument("--mix-holoassist", choices=["idle", "all"], default=None)
    ap.add_argument("--none-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--heldout", nargs="+", default=["P22", "P30"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from src.data.annotations import load_video_info
    from src.data.window_labels import NONE, load_actions, point_labels
    from src.models.frozen import FeatureHead

    data_cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    ann = Path(resolve_data_path(data_cfg["annotations_dir"]))
    fd = Path(resolve_data_path(args.feats))
    info = json.loads((fd / "info.json").read_text())
    window = float(args.window)
    meta = pd.read_csv(fd / "meta.csv")
    feats = np.load(fd / f"feats_{window}.npy").astype(np.float32)             # (N, 768)
    fps = {str(v): float(f) for v, f in load_video_info(ann)["fps"].items()}
    V = len(DEMO_VERBS)
    # Class ids are alphabetical over ALL names, "none" included, because the
    # app's smoothers sort their verb list. "none" is therefore id 3.
    verbs = sorted(DEMO_VERBS + [NONE])
    verb_map = {v: i for i, v in enumerate(verbs)}
    NONE_ID = verb_map[NONE]
    to_id = np.array([verb_map[v] for v in DEMO_VERBS] + [NONE_ID, NONE_ID])   # point label -> class id
    actions = load_actions(ann, DEMO_VERBS)
    label, _, _ = point_labels(meta["video_id"].to_numpy(str), meta["frame"].to_numpy(), fps,
                               window, DEMO_VERBS, actions, args.ref)
    y = to_id[label]                                                           # other + background -> none
    is_val = meta["participant_id"].isin(args.heldout).to_numpy()
    Xtr, ytr, Xva, yva = feats[~is_val], y[~is_val], feats[is_val], y[is_val]
    n_window = len(ytr)
    if args.mix_trimmed:                                                        # clean per-action examples
        td = Path(resolve_data_path(TRIMMED))
        tm = pd.read_csv(td / "meta.csv")
        classes = pd.read_csv(ann / "EPIC_100_verb_classes.csv")
        key = dict(zip(classes["id"].astype(int), classes["key"].astype(str)))
        ty = tm["verb_class"].astype(int).map(key).map(verb_map).fillna(-1).astype(int).to_numpy()
        keep = (ty >= 0) & ~tm["participant_id"].isin(args.heldout).to_numpy() & (tm["ok"].to_numpy() == 1)
        tx = np.load(td / "feats.npy", mmap_mode="r")
        Xtr = np.concatenate([Xtr, np.asarray(tx[keep], np.float32).mean(1)])
        ytr = np.concatenate([ytr, ty[keep]])
    if args.mix_holoassist:
        from scripts.egoextra_from_video import holoassist_ticks
        hx, hst, _, hnames, _ = holoassist_ticks(window)
        same = {"grab": "take", "place": "put", "open": "open", "close": "close"}
        hid = np.array([verb_map[same[n]] if n in same else NONE_ID for n in hnames])
        keep = (np.array(hnames)[hst] == "idle") if args.mix_holoassist == "idle" else np.ones(len(hst), bool)
        Xtr = np.concatenate([Xtr, hx[keep]])
        ytr = np.concatenate([ytr, hid[hst[keep]]])
    C = len(verbs)
    counts = np.bincount(ytr, minlength=C).astype(np.float64)
    print(f"window {window} s  train {len(ytr)} ({n_window} windows)  val {len(yva)}")
    print("classes     ", verbs)
    print("train counts", counts.astype(int).tolist())
    print("val counts  ", np.bincount(yva, minlength=C).tolist())

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xv = torch.from_numpy(Xva).to(dev)
    w = (counts.sum() / (C * np.maximum(counts, 1))) ** 0.5                    # softened inverse frequency
    w[NONE_ID] *= args.none_weight
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev),
                                  label_smoothing=args.label_smoothing)
    model = FeatureHead(Xtr.shape[1], C, kind="mlp", hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(ytr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    m = registry.get("metrics")

    def evaluate():
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(Xv).float(), dim=1).cpu().numpy()
        pred = p.argmax(1)
        demo = yva != NONE_ID
        return {"acc": float((pred == yva).mean()),
                "mca": float(m.mean_class_accuracy(pred, yva, C)),
                "demo_recall": float((pred[demo] == yva[demo]).mean()),           # demo ticks named correctly
                "none_recall": float((pred[~demo] == NONE_ID).mean()),            # none ticks kept quiet
                "per_class": m.per_class_accuracy(pred, yva, C).tolist(),
                "confusion": m.confusion_matrix(pred, yva, C).tolist()}

    rows = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(ytr))).to(dev)
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            loss = loss_fn(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(b)
        ev = evaluate()
        rows.append({"epoch": epoch, "train_loss": round(total / len(ytr), 4), "val_acc": round(ev["acc"], 4),
                     "val_mca": round(ev["mca"], 4), "demo_recall": round(ev["demo_recall"], 4),
                     "none_recall": round(ev["none_recall"], 4)})

    out = Path(args.out or f"runs/videomae_window_w{window}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "head": model.config(), "epoch": args.epochs - 1,
                "verb_map": verb_map, "kind": "frozen_videomae", "encoder": info["model"], "T": 16,
                "window_s": window}, out / "last.pt")
    (out / "verb_map.json").write_text(json.dumps(verb_map, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump(
        {"model": "frozen_videomae", "encoder": info["model"], "T": 16, "window_s": window,
         "head": model.config(), "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
         "heldout_participants": args.heldout, "trained_on": "sliding windows",
         "mix_trimmed": args.mix_trimmed, "mix_holoassist": args.mix_holoassist, "none_weight": args.none_weight, "label_ref": args.ref}))
    with open(out / "log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    ev.update(verbs=verbs, train_counts=counts.astype(int).tolist(), n_train=int(len(ytr)), n_val=int(len(yva)),
              majority=float(np.bincount(yva).max() / len(yva)))
    (out / "metrics.json").write_text(json.dumps(ev, indent=2))
    print(f"\nfinal epoch: val_acc {ev['acc']:.4f} (always-'none' baseline {ev['majority']:.4f})  "
          f"val_mca {ev['mca']:.4f}")
    print(f"demo-verb ticks named correctly {ev['demo_recall']:.4f}   none ticks kept quiet {ev['none_recall']:.4f}")
    for v, a in zip(verbs, ev["per_class"]):
        print(f"  {v:<6} recall {a:.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
