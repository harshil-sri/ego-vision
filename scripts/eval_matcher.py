"""Evaluate the scan-phase matcher on recorded ingredient frames (D7 + D4
evidence): held-out top-1 per embedder and crop option, plus a threshold
sweep (false-accept / false-reject) -> reports/matcher_eval.md.

Protocol: for each ingredient, half the frames (rounded down) enroll the
prototype (the "scan phase"); the REST are queries (the "cooking phase").
Top-1 = fraction of queries whose best match is the right ingredient and
the score clears the threshold. Unknown-queries: a held-OUT ingredient
(registered by --unknown or the alphabetically-last one is held out
entirely) must read "unknown" — false accepts are measured on it.

Usage:
    python3 -m scripts.eval_matcher --root data/ingredients \
        --embedders toy,dinov2 --crops show_zone,whole_frame
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import src.registry as registry
from src.pipeline.embedder import make_embedder

SHOW_ZONE = (0.30, 0.25, 0.40, 0.50)  # must mirror configs/app.yaml


def crop(image: np.ndarray, mode: str) -> np.ndarray:
    """Apply crop policy; image HxWx3 uint8 -> cropped view."""
    if mode != "show_zone":
        return image
    H, W = image.shape[:2]
    rx, ry, rw, rh = SHOW_ZONE
    return image[int(ry * H):int((ry + rh) * H), int(rx * W):int((rx + rw) * W)]


def load_split(root: Path, unknown_name: str | None):
    """Return (enroll_map, query_map): name -> list[HxWx3].

    Half of each ingredient's frames enroll, half query (deterministic
    split by sorted filename index parity). The `unknown_name` ingredient
    (if given) contributes ONLY queries — it was never scanned.
    """
    enroll, query = {}, {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name
        files = sorted(folder.glob("*.jpg"))
        imgs = [cv2.imread(str(f)) for f in files]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            continue
        if unknown_name and name == unknown_name:
            query[name] = imgs
            continue
        half = len(imgs) // 2
        enroll[name] = imgs[:half]
        query[name] = imgs[half:]
    return enroll, query


def evaluate(embedder_name: str, crop_mode: str, enroll, query,
             unknown_name: str | None) -> dict:
    """One (embedder, crop) cell: top-1 across thresholds."""
    embedder = make_embedder(embedder_name)
    embed = lambda frames: embedder([crop(f, crop_mode) for f in frames])
    matcher_cls = registry.get_class("enrollment")
    matcher = matcher_cls(embed, threshold=0.0)  # sweep handles thresholds
    for name, frames in enroll.items():
        matcher.enroll(name, frames)

    # Raw best-match scores per query: (true_name, best_name, best_score)
    triples = []
    for name, frames in query.items():
        for f in frames:
            best_name, score = matcher.match(crop(f, crop_mode))
            triples.append((name, best_name, float(score)))

    known = [t for t in triples if not (unknown_name and t[0] == unknown_name)]
    unknown = [t for t in triples
               if unknown_name and t[0] == unknown_name] or None
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = []
    for th in thresholds:
        correct = sum(1 for n, b, s in known if b == n and s >= th)
        rejects = sum(1 for n, b, s in known if s < th)
        total = len(known)
        fa = (sum(1 for _, b, s in unknown if b != "unknown" or s >= th)
              / len(unknown)) if unknown else float("nan")
        sweep.append({
            "threshold": th,
            "top1": correct / total if total else float("nan"),
            "rejected_pct": rejects / total if total else float("nan"),
            "unknown_false_accept": fa,
            "n_known": total, "n_unknown": len(unknown) if unknown else 0,
        })
    return {"embedder": embedder_name, "crop": crop_mode, "sweep": sweep}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/ingredients")
    p.add_argument("--embedders", default="toy")
    p.add_argument("--crops", default="show_zone,whole_frame")
    p.add_argument("--unknown", default=None,
                   help="ingredient name held out entirely (false-accept test)")
    p.add_argument("--out", default="reports/matcher_eval.md")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists() or not any(root.iterdir()):
        (Path(args.out)).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            "# Matcher evaluation\n\nNOT RUN — no recorded frames at "
            f"{root}. Record some first:\n"
            "  python -m scripts.record_ingredient_frames "
            "--ingredients tomato,cucumber,onion\n")
        print(f"no frames at {root}; wrote NOT RUN note to {args.out}")
        return

    enroll, query = load_split(root, args.unknown)
    if len(enroll) < 2:
        print(f"need >= 2 enrolled ingredients, found {len(enroll)}; aborting")
        return

    lines = ["# Matcher evaluation (D7 embedder / D4 crop evidence)", "",
             f"Frames root: `{root}`. Split: half enroll / half query per "
             "ingredient (deterministic, sorted-by-filename parity).",
             f"Held-out unknown ingredient: {args.unknown or 'none'}.", ""]
    for embedder_name in args.embedders.split(","):
        for crop_mode in args.crops.split(","):
            try:
                result = evaluate(embedder_name, crop_mode, enroll, query,
                                  args.unknown)
            except Exception as exc:
                lines.append(f"## {embedder_name} / {crop_mode}\n\nFAILED: "
                             f"{exc}\n")
                continue
            lines.append(f"## {embedder_name} / {crop_mode}\n")
            lines.append("| threshold | top-1 | rejected % | unknown "
                         "false-accept | n known / unknown |")
            lines.append("|---|---|---|---|---|")
            for row in result["sweep"]:
                fa = row["unknown_false_accept"]
                lines.append(
                    f"| {row['threshold']:.2f} | {row['top1']:.3f} | "
                    f"{row['rejected_pct'] * 100:.1f}% | "
                    f"{'n/a' if fa != fa else f'{fa * 100:.1f}%'} | "
                    f"{row['n_known']} / {row['n_unknown']} |")
            lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
