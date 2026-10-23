"""Run the cooking demo app.

Tier 1 (no model):   python -m scripts.run_app --mode keyboard
Tier 2 (clip):       python -m scripts.run_app --mode clip --clip <mp4> --run-dir runs/<id>
Tier 3 (live):       python -m scripts.run_app --mode live --run-dir runs/<id>
UIs:                 --ui web (default, browser) | --ui cv (OpenCV window)
No camera attached?  --synthetic frames come from a procedural source.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["keyboard", "clip", "live"],
                   default="keyboard")
    p.add_argument("--ui", choices=["web", "cv"], default="web")
    p.add_argument("--recipe", default="recipes/example_salad.yaml")
    p.add_argument("--config", default="configs/app.yaml")
    p.add_argument("--source", default=None, help="camera device (default /dev/video0)")
    p.add_argument("--clip", default=None, help="video file for --mode clip")
    p.add_argument("--run-dir", default=None, help="runs/<id> with a checkpoint")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--synthetic", action="store_true",
                   help="procedural frames instead of a camera (smoke tests)")
    return p.parse_args()


def build_engine(args):
    """Wire frame source + verb source + engine from CLI/config."""
    from src.pipeline.engine import CookingEngine, EngineConfig
    from src.pipeline.model_source import make_verb_source

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    cam = cfg_raw.pop("camera", {})
    cfg_raw.pop("mode", None)  # CLI --mode wins over the config file
    cfg = EngineConfig(mode=args.mode, **cfg_raw)

    if args.synthetic:
        from src.pipeline.window import SyntheticFrameSource
        source = SyntheticFrameSource()
    elif args.mode == "clip":
        from src.pipeline.window import ClipFrameSource
        if not args.clip:
            raise SystemExit("--mode clip needs --clip <video.mp4>")
        source = ClipFrameSource(args.clip, real_time=True)
    else:
        from src.pipeline.window import CameraFrameSource
        source = CameraFrameSource(
            source=args.source or cam.get("source", "/dev/video0"),
            width=cam.get("width", 640), height=cam.get("height", 480),
            fps=cam.get("fps", 30), pixel_format=cam.get("pixel_format", "MJPG"))

    try:
        verb_source = make_verb_source(args.mode, args.run_dir, cfg.device)
    except FileNotFoundError as exc:
        raise SystemExit(f"cannot start: {exc}")

    return CookingEngine(source, cfg, args.recipe, verb_source=verb_source)


def run_cv_ui(engine):
    """OpenCV fallback UI: same overlay, number keys emit verbs, q quits."""
    import cv2
    engine.start()
    keys = "12345678"
    print("CV UI: keys 1-8 emit verbs:", engine.recipe.verbs)
    print("      d toggles nothing here (web UI has the debug drawer); q quits")
    while True:
        frame = engine.annotated_frame()
        if frame is not None:
            cv2.imshow("ego-kitchen", frame)
        k = cv2.waitKey(80) & 0xFF
        if k == ord("q"):
            break
        if ord("1") <= k <= ord("8"):
            idx = k - ord("1")
            if idx < len(engine.recipe.verbs):
                engine.inject_verb(engine.recipe.verbs[idx])
        elif k == ord(" "):  # space: advance scan wizard manually
            st = engine.state()
            if st["phase"] == "idle" or st["scan"].get("ingredient") is None:
                not_done = [i for i in engine.recipe.ingredients
                            if i not in engine._enrolled]
                if not_done:
                    engine.begin_ingredient_scan(not_done[0])
                else:
                    engine.finish_scan()
    cv2.destroyAllWindows()
    engine.stop()


def run_web(engine, port: int):
    """FastAPI + uvicorn; serves the static UI from src/app/static."""
    import uvicorn
    from src.app.server import create_app
    static_dir = Path(__file__).resolve().parent.parent / "src" / "app" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(engine, static_dir)
    engine.start()
    print(f"UI: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    args = parse_args()
    engine = build_engine(args)
    if args.ui == "cv":
        run_cv_ui(engine)
    else:
        run_web(engine, args.port)


if __name__ == "__main__":
    main()
