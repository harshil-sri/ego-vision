"""Kaggle GPU kernel: train the ego-vision verb baseline on EPIC-KITCHENS.

Runs on Tanish's account (mahajantanish1), GPU T4/P100, internet ON.

Flow:
  1. clone the repo (branch glm-window, public GitHub)
  2. clone the EPIC-100 annotations repo (small)
  3. stream the selected video subset from the HF mirror: per video,
     download -> downscale to short side 256 (ffmpeg) -> delete the
     original (keeps peak disk low; corpus lands in /kaggle/working/EPIC-256)
  4. overfit-32 gate on REAL clips (the training ritual; must reach ~100%)
  5. baseline training run (config from the repo, paths patched)
  6. outputs persist in /kaggle/working/runs/<id> (metrics.json, best.pt,
     log.csv, verb_map.json, confusion.png)

Everything the kernel prints lands in the kernel log — numbers are real or
absent, never guessed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "ego-vision"
BRANCH = os.environ.get("EGO_BRANCH", "glm-window")
EPOCHS = os.environ.get("EGO_EPOCHS", "12")
SKIP_DOWNLOAD = os.environ.get("EGO_SKIP_DOWNLOAD", "0") == "1"

# kaggle/variant.txt in the REPO selects the model variant for a pushed run
# (pushed kernels cannot receive env vars, so the file IS the switch).
# Absent/empty -> single_frame, identical to the original baseline.
TAG_FOR_VARIANT = {"single_frame": "baseline", "tsn_avg": "tsn",
                   "temporal_attn": "temporal_attn"}

ANN = WORK / "epic-annotations"
EPIC256 = WORK / "EPIC-256"


def sh(cmd: str) -> None:
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    t0 = time.time()

    # ---- 1-2. code + annotations
    if not REPO.exists():
        sh(f"git clone -q --depth 1 -b {BRANCH} "
           f"https://github.com/harshil-sri/ego-vision.git {REPO}")
    if not ANN.exists():
        sh(f"git clone -q --depth 1 "
           f"https://github.com/epic-kitchens/epic-kitchens-100-annotations.git {ANN}")
    os.chdir(REPO)

    variant_path = REPO / "kaggle" / "variant.txt"
    variant = (variant_path.read_text().strip()
               if variant_path.exists() else "") or "single_frame"
    run_tag = os.environ.get("EGO_RUN_TAG") or TAG_FOR_VARIANT.get(variant, variant)
    print(f"[variant] {variant} -> runs/{run_tag}", flush=True)

    # ---- 3. video subset: download -> downscale -> delete original
    import pandas as pd
    from huggingface_hub import hf_hub_download

    selected = pd.read_csv("reports/selected_videos.csv")
    total_gb = selected["size_gb"].sum()
    print(f"selected subset: {len(selected)} videos, {total_gb:.1f} GB "
          f"(raw); downscaled copies are ~2% of that", flush=True)
    if not SKIP_DOWNLOAD:
        done = 0
        for row in selected.itertuples():
            vid = row.video_id
            out = EPIC256 / vid[:3] / "videos" / f"{vid}.MP4"
            if out.exists():
                done += 1
                continue
            raw = None
            try:
                raw = hf_hub_download(
                    repo_id="awsaf49/epic_kitchens_100", repo_type="dataset",
                    filename=f"{vid[:3]}/videos/{vid}.MP4",
                    local_dir=WORK / "raw")
                out.parent.mkdir(parents=True, exist_ok=True)
                sh(f"ffmpeg -y -loglevel error -i '{raw}' -vf scale=-2:256 "
                   f"-c:v libx264 -crf 23 -preset veryfast -an '{out}'")
                done += 1
                if done % 10 == 0:
                    print(f"[data] {done}/{len(selected)} downscaled "
                          f"({time.time() - t0:.0f}s elapsed)", flush=True)
            except Exception as exc:
                print(f"[data] WARNING {vid} failed: {exc}", flush=True)
            finally:
                if raw is not None and Path(raw).exists():
                    Path(raw).unlink()  # keep peak disk small
                # hf_hub also leaves blobs under raw/.cache — prune
                cache = WORK / "raw" / ".cache"
                if cache.exists():
                    shutil.rmtree(cache, ignore_errors=True)
        print(f"[data] downscale complete: {done} videos at {EPIC256}", flush=True)

    # ---- 4-5. overfit gate, then baseline train
    cfg = {
        "backbone": "resnet34", "variant": variant, "T": 8,
        "img_size": 224, "optimizer": "adamw", "lr": 0.0003,
        "weight_decay": 0.0001, "lr_schedule": "cosine", "epochs": int(EPOCHS),
        "batch_size": 16, "amp": True, "class_weights": True, "seed": 42,
        "grad_clip": 1.0, "out_dir": str(WORK / "runs" / run_tag),
        "overfit_batch": None, "train_limit": None, "device": "cuda",
        "data_root": str(WORK), "annotations_dir": str(ANN),
        "videos_root": str(EPIC256), "heldout_participants": ["P22", "P30"],
    }
    cfg_path = WORK / f"train_{run_tag}.yaml"
    cfg_path.write_text(json.dumps(cfg))  # JSON is valid YAML
    print(f"[cfg] {cfg_path}", flush=True)

    overfit_cfg = dict(cfg, epochs=25, overfit_batch=32, train_limit=32,
                       batch_size=8, img_size=160, amp=False,
                       out_dir=str(WORK / "runs" / "overfit32"))
    (WORK / "train_overfit.yaml").write_text(json.dumps(overfit_cfg))

    print("[gate] overfit-32 on real clips", flush=True)
    r = subprocess.run([sys.executable, "-m", "scripts.train",
                        "--config", str(WORK / "train_overfit.yaml"),
                        "--workers", "2"], capture_output=True, text=True)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        print("GATE FAILED — long run skipped", flush=True)
        return

    print(f"[train] baseline run ({run_tag}, epochs={EPOCHS})", flush=True)
    r = subprocess.run([sys.executable, "-m", "scripts.train",
                        "--config", str(cfg_path), "--workers", "4"],
                        capture_output=True, text=True)
    print(r.stdout[-5000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-5000:], flush=True)
        print("TRAIN FAILED", flush=True)
        return

    # ---- 6. results stay under /kaggle/working/runs/*
    metrics = WORK / "runs" / run_tag / "metrics.json"
    if metrics.exists():
        print("[done] metrics:", metrics.read_text(), flush=True)
    print(f"[done] total kernel time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
