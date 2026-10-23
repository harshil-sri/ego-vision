# ego-vision — EGO-KITCHEN

A head-mounted webcam watches you cook a fixed recipe: a scan phase
learns your ingredients in 10 seconds each, a trained verb model reads
the last ~2 s of video, and a recipe state machine tracks progress in a
live UI.

Built on EPIC-KITCHENS-100 (personal demo use). DINOv2 weights: check
license note in DECISIONS_NEEDED.md D7 before any commercial use.

## Install (WSL2 / Linux)

```bash
python3 -m venv ~/.venvs/ego && source ~/.venvs/ego/bin/activate
pip install -r requirements.txt        # torch: use the cu124 index if training on GPU
sudo apt install ffmpeg aria2          # or use static builds
```

## Run (under 10 lines)

```bash
make test                              # unit tests
make smoke                             # Tier 1 flow, no camera needed
make app                               # Tier 1 on the webcam -> http://127.0.0.1:8000
python3 -m scripts.run_app --mode keyboard --synthetic   # headless, no camera
python3 -m scripts.run_app --mode clip --clip vid.mp4 --run-dir runs/<id>  # Tier 2
python3 -m scripts.run_app --mode live --run-dir runs/<id>                # Tier 3
python3 -m scripts.run_app --mode keyboard --ui cv       # OpenCV fallback UI
```

Number keys 1-8 emit verbs (keyboard mode); `d` debug drawer, `p`
presentation mode, `?` help. Webcam must be MJPEG (WSL2 requirement).

## Layout

- `src/pipeline/` — engine, sliding window, smoothers, event bus, session log
- `src/app/` — FastAPI server + static web UI
- `src/models`, `src/train/` — verb model variants, training/eval
- `src/data/` — EPIC annotations, dataset, video decoding
- `reference/` — study-material implementations (+ `exercises/`)
- `exercises/stubs/` — copy to `core/` and fill in (owner's five pieces)
- `docs/` — ARCHITECTURE, GLOSSARY, STUDY_QUEUE, worked examples
- `recipes/`, `configs/` — recipe YAML, app/train/impl switches
- `runs/`, `reports/` — logged outputs and evidence tables
- STATUS.md, DECISIONS_NEEDED.md, BLOCKERS.md, HANDOFF.md — the paper trail

## Verify everything

```bash
bash scripts/check_all.sh
```
