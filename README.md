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

## Install (Windows 11, native)

The global uv venv is already active in every shell. Data lives at
`C:\ssd-data`; configs keep the old `/mnt/ssd/...` spelling and
`src/paths.py` translates it (override with the `EGO_DATA_ROOT`
environment variable).

```powershell
uv pip install --python $env:USERPROFILE\.venvs\global\Scripts\python.exe -r requirements.txt
python -m pytest tests -q
```

The camera opens through DirectShow; `/dev/video0` in a config means
camera index 0.

## Verb model (frozen VideoMAE features + small head)

```powershell
python -m scripts.extract_features --encoder videomae --device cuda   # once, ~16 min
python -m scripts.train_head --feats /mnt/ssd/epic-kitchens/features/videomae_ssv2_T16   # seconds
python -m scripts.run_app --mode clip --clip <video.mp4> --run-dir runs/videomae_mlp --device cuda
```

Results and the comparison with the earlier models: `reports/verb_model_frozen.md`.
Research notes (when should an assistant speak): `reports/research_gap_map.md`.

## Research pilot (when should an assistant speak?)

Question, evidence and limits: `reports/research_gap_map.md`. Every number
there comes from one of these commands. Data goes to `C:\ssd-data`.

```powershell
# HoloAssist, labels only (117 MB of annotations): parts 1 to 4
python -m scripts.holoassist_timing
python -m scripts.holoassist_predict
python -m scripts.holoassist_adjusted
python -m scripts.holoassist_aftermath
python -m scripts.holoassist_silence
python -m scripts.holoassist_need_controlled     # part 6
python -m scripts.holoassist_idle_spell          # part 6c
python -m scripts.holoassist_student_speech      # part 7
python -m scripts.holoassist_wait_policy         # part 8
python -m scripts.holoassist_wait_policy --video # part 8 from video (needs the features of part 5)
python -m scripts.holoassist_delivery_sim       # replay of the app's message-hold rule (needs the features)
# HoloAssist from video: part 5 (downloads 22 GB out of a 155 GB archive)
python -m scripts.holoassist_download_subset index
python -m scripts.holoassist_download_subset download --videos 300
python -m scripts.holoassist_video_features            # about 80 min on the GPU
python -m scripts.holoassist_from_video
python -m scripts.holoassist_shift_control
# Ego-EXTRA, second dataset (transcripts 1 MB, hand tracking 2.2 GB, gaze 0.8 GB, videos 23.7 GB)
python -m scripts.egoextra_speech
python -m scripts.egoextra_hands --signal wrist        # also: fingers, gaze, gaze_depth
python -m scripts.holoassist_video_features --root /mnt/ssd/ego-extra --center 320
python -m scripts.egoextra_from_video
python -m scripts.egoextra_predict
# the hand-state signal inside the app (experimental)
python -m scripts.train_hand_state_head
python -m scripts.run_app --config configs/app_demo_videomae.yaml --mode clip --clip <video.mp4> --run-dir runs/videomae_mlp --hand-state-dir runs/hand_state_holoassist
```

In Git Bash, put `MSYS_NO_PATHCONV=1` before a command that takes a
`/mnt/ssd/...` argument; PowerShell needs nothing.

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
