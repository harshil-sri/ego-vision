"""Record labelled ingredient frames from the webcam into
data/ingredients/<name>/*.jpg — EVALUATION ONLY (never training data).

Usage (interactive, one ingredient at a time):
    python3 -m scripts.record_ingredient_frames --ingredients tomato,cucumber
Each ingredient gets an N-second capture window (hold the ingredient in
the show zone); frames are saved as jpg plus an index.csv.

--synthetic writes deterministic coloured frames instead (offline smoke).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.capture import CameraCapture, CameraConfig


def synthetic_frame(color: tuple[int, int, int], size: int = 64,
                    noise: int = 8, rng=None) -> np.ndarray:
    """Deterministic near-constant-colour frame (HxWx3 uint8)."""
    rng = rng or np.random.default_rng(0)
    base = np.ones((size, size, 3), np.uint8) * np.array(color, np.uint8)
    return np.clip(base + rng.integers(-noise, noise + 1, base.shape),
                   0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ingredients", required=True,
                   help="comma-separated ingredient names")
    p.add_argument("--out", default="data/ingredients")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="capture window per ingredient")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--source", default="/dev/video0")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    names = [x.strip() for x in args.ingredients.split(",") if x.strip()]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    cam: CameraCapture | None = None
    if not args.synthetic:
        cfg = CameraConfig(source=args.source, width=640, height=480, fps=30,
                           pixel_format="MJPG")
        cam = CameraCapture(cfg)
        cam.open()

    palette = [(230, 60, 60), (60, 230, 60), (60, 60, 230), (230, 230, 60),
               (230, 60, 230), (60, 230, 230), (200, 130, 40), (240, 240, 240)]
    try:
        for i, name in enumerate(names):
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            target = int(args.seconds * args.fps)
            print(f"[{name}] capture {target} frames "
                  f"({args.seconds:.0f}s at {args.fps:.0f} fps)...")
            rng = np.random.default_rng(1000 + i)
            got = 0
            t0 = time.monotonic()
            while got < target:
                if args.synthetic:
                    frame = synthetic_frame(palette[i % 8], rng=rng)
                    time.sleep(1.0 / args.fps)
                else:
                    packet = cam.read()
                    if packet is None:
                        continue
                    frame = packet.image
                    if (got / args.fps) > (time.monotonic() - t0):
                        continue  # pace to fps
                path = folder / f"{name}_{got:03d}.jpg"
                import cv2
                cv2.imwrite(str(path), frame)
                rows.append({"ingredient": name, "path": str(path),
                             "t": round(time.monotonic() - t0, 3)})
                got += 1
            print(f"[{name}] saved {got} frames -> {folder}")
    finally:
        if cam is not None:
            cam.release()

    with open(root / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ingredient", "path", "t"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"index: {root / 'index.csv'} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
