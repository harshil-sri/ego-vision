"""Harvest a finished Kaggle training kernel's outputs into runs/.

Pulls the kernel output bundle, verifies the overfit gate result and the
baseline metrics exist, and prints a summary suitable for pasting into
STATUS.md / HANDOFF.md. Refuses to report numbers that are not in the
pulled files (anti-fabrication guard).

Usage (Tanish's account):
    export KAGGLE_API_TOKEN=$(cat ~/.kaggle/tanish_access_token)
    python3 -m scripts.harvest_kaggle [--kernel mahajantanish1/ego-vision-baseline] \
        [--out runs/kaggle-baseline]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", default="mahajantanish1/ego-vision-baseline")
    p.add_argument("--out", default="runs/kaggle-baseline")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"pulling output of {args.kernel} -> {out}")
    subprocess.run(["kaggle", "kernels", "output", args.kernel, "-p", str(out)],
                   check=True)

    # ---- verify + summarize (only what the files actually say)
    summary: list[str] = [f"# {args.kernel} harvest summary"]
    gate = out / "runs" / "overfit32" / "metrics.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        summary.append(f"overfit-32 gate: best train acc "
                       f"{g.get('best_val_mean_class_acc', 'MISSING')} "
                       f"(gate passes at >= 0.99)")
    else:
        summary.append("overfit-32 gate metrics: NOT FOUND in output")

    base = out / "runs" / "baseline" / "metrics.json"
    if base.exists():
        b = json.loads(base.read_text())
        for key in ("best_epoch", "best_val_mean_class_acc",
                    "majority_baseline_val"):
            summary.append(f"baseline {key}: {b.get(key, 'MISSING')}")
    else:
        summary.append("baseline metrics.json: NOT FOUND — check the kernel "
                       "log; the gate may have failed or training errored")

    log = out / "runs" / "baseline" / "log.csv"
    if log.exists():
        lines = log.read_text().strip().splitlines()
        summary.append(f"baseline epochs logged: {len(lines) - 1} "
                       f"(last: {lines[-1][:120]})")

    ckpt = out / "runs" / "baseline" / "best.pt"
    summary.append(f"checkpoint: {'present (' + str(ckpt.stat().st_size // 1024 // 1024) + ' MB)' if ckpt.exists() else 'MISSING'}")

    text = "\n".join(summary)
    print(text)
    (out / "HARVEST_SUMMARY.md").write_text(text + "\n")
    if not base.exists():
        sys.exit(2)


if __name__ == "__main__":
    main()
