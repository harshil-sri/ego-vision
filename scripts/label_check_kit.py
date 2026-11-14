"""Build a blind check sheet for a second, human labeller.

The hand labels of Ego-EXTRA (reports/egoextra_hand_labels.csv) and WTaG
(reports/wtag_hand_labels.csv) come from one labeller, the AI assistant.
This script draws a random part of them and writes ONE html file that a
person opens in a browser. The person sees the same 4 frames that the
first labeller saw (the 2 s that end at the moment, 0.5 s apart), chooses
S, B or F for each moment, and downloads the answers as a file.

The sheet does not show the first label, the kind of the moment (speech
onset or random second), the recording or the original id. The moments
are shuffled and numbered 1 to n. The key (new number -> original id) is
written next to the sheet and is not sent to the checker.

The sample: the same number of moments of each kind, drawn at random.
Ego-EXTRA: furniture assembly only (the scenario with a result).

The output holds frames of the datasets, so it goes to runs/label_check/,
which is not in git. Send the html file to the checker privately.

Usage:
    python -m scripts.label_check_kit --dataset egoextra --root C:/ssd-data/ego-extra
    python -m scripts.label_check_kit --dataset wtag --root C:/ssd-data/wtag
Then: python -m scripts.label_agreement --dataset egoextra --answers <file from the checker>
Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SETS = {
    "egoextra": {
        "vf": "crop=500:500:102:170,scale=300:300", "size": (300, 300), "in_order": False,
        "task": "furniture assembly",
        "S": "A screw, unscrew or insert action is in progress: a screwdriver or hex key is turning, "
             "a screw or dowel is going into its hole, a cap is being turned.",
        "B": "Other precise work is in progress: hammering, wiping, aligning two parts, using pliers.",
        "F": "Everything else: hold, carry, reach, pick up, look, gesture, or no hands in view.",
    },
    "wtag": {
        "vf": "scale=384:216", "size": (216, 384), "in_order": True,
        "task": "cooking",
        "S": "A screw, unscrew or insert action is in progress: twisting a lid, putting the filter "
             "into the cone, pushing toothpicks into the wrap.",
        "B": "Other precise work is in progress: pour, spread, cut, scoop or measure, stir, fold, squeeze, wipe.",
        "F": "Everything else: idle, hold, reach, carry, place, look, or hands not in view.",
    },
}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Hand label check</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1240px;margin:0 auto;padding:16px;background:#f6f6f4;color:#1a1a1a;line-height:1.5}
.box{background:#fff;border:1px solid #ccc;border-radius:8px;padding:14px 18px;margin:14px 0}
.item img{width:100%;height:auto;display:block;border-radius:4px}
.item.done{border-color:#2a7}
label{display:inline-block;margin:10px 18px 0 0;padding:6px 14px;border:1px solid #999;border-radius:6px;cursor:pointer;font-weight:600}
input:checked+span{text-decoration:underline}
#bar{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:8px 14px;border-radius:6px;z-index:5}
button{font-size:1rem;padding:10px 18px;border-radius:6px;border:0;background:#1a5fb4;color:#fff;cursor:pointer}
button:disabled{background:#999}
</style></head><body>
<h1>Hand label check (__TASK__)</h1>
<div class="box">
<p>Each row shows 4 pictures from 2 seconds of a video, in time order from left to right
(0.5 seconds apart). The camera is on the head of a person who does a task.
For each row, answer one question: <b>what are the hands doing at the end of the row?</b></p>
<p><b>S</b>: __S__</p>
<p><b>B</b>: __B__</p>
<p><b>F</b>: __F__</p>
<p>If you are not sure, choose the label that fits best. Do not go back to change many answers.
There are __N__ rows. It takes about 15 minutes. At the end, press the button and send the
downloaded file back.</p>
<p>Your name or initials: <input id="who" style="font-size:1rem;padding:4px"></p>
</div>
<div id="bar">Answered: <span id="n">0</span> of __N__</div>
__ITEMS__
<div class="box"><button id="save" disabled>Download answers</button>
<p id="msg"></p></div>
<script>
const N=__N__, DS="__DS__", ans={};
document.querySelectorAll('input[type=radio]').forEach(r=>r.addEventListener('change',()=>{
  ans[r.name]=r.value; r.closest('.item').classList.add('done');
  const k=Object.keys(ans).length; document.getElementById('n').textContent=k;
  document.getElementById('save').disabled=k<N;}));
document.getElementById('save').addEventListener('click',()=>{
  const out={dataset:DS,who:document.getElementById('who').value,time:new Date().toISOString(),answers:ans};
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out)],{type:'application/json'}));
  a.download=DS+'_check_answers.json'; a.click();
  document.getElementById('msg').textContent='Saved. Please send the file '+a.download+' back. Thank you.';});
</script></body></html>"""


def frame(video: Path, t: float, vf: str, size) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", vf, "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    b = subprocess.run(cmd, capture_output=True).stdout
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
    return im if im is not None else np.zeros((*size, 3), np.uint8)


def frames_in_order(video: Path, times: list[float], size) -> list[np.ndarray]:
    """Frames at the given times, read in order from the start of the file.

    The WTaG videos have packets without a time stamp. A seek (ffmpeg -ss)
    then returns the last key frame, up to some seconds off, and the 4
    pictures of a row are often the same picture. Reading in order is
    slow and exact."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = {}
    for j, t in enumerate(times):
        want.setdefault(int(round(max(t, 0) * fps)), []).append(j)
    out = [np.zeros((*size, 3), np.uint8) for _ in times]
    for i in range(max(want) + 1):
        if i in want:
            ok, f = cap.read()
            if ok:
                for j in want[i]:
                    out[j] = cv2.resize(f, (size[1], size[0]))
        elif not cap.grab():
            break
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SETS), required=True)
    ap.add_argument("--root", required=True, help="dataset folder with videos/<recording>.mp4")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/label_check")
    a = ap.parse_args()
    cfg, rng = SETS[a.dataset], np.random.default_rng(a.seed)
    lab = pd.read_csv(f"reports/{a.dataset}_hand_labels.csv", dtype={"recording": str})
    if a.dataset == "egoextra":
        lab = lab[lab.recording.str[:2] == "03"]
    kinds = sorted(lab.kind.unique())
    per = a.n // len(kinds)
    pick = pd.concat([lab[lab.kind == k].sample(per, random_state=int(rng.integers(1 << 31))) for k in kinds])
    pick = pick.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, key, repeated = [], {}, 0
    for i, r in pick.iterrows():
        video = Path(a.root) / "videos" / f"{r.recording}.mp4"
        offs = (1.75, 1.25, 0.75, 0.25)
        if cfg["in_order"]:
            fr = frames_in_order(video, [r.t_s - o for o in offs], cfg["size"])
        else:
            fr = [frame(video, r.t_s - o, cfg["vf"], cfg["size"]) for o in offs]
        row = np.concatenate(fr, 1)
        if not row.any():
            raise SystemExit(f"no frames for {video} at {r.t_s}")
        if any(np.abs(fr[j].astype(float) - fr[j + 1]).mean() < 0.5 for j in range(3)):
            repeated += 1
        ok, jpg = cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n = i + 1
        key[str(n)] = int(r.id)
        radios = "".join(f'<label><input type="radio" name="{n}" value="{v}"> <span>{v}</span></label>' for v in "SBF")
        items.append(f'<div class="box item"><b>{n}</b><img alt="row {n}" '
                     f'src="data:image/jpeg;base64,{base64.b64encode(jpg.tobytes()).decode()}">{radios}</div>')
    html = PAGE
    for k, v in (("__ITEMS__", "\n".join(items)), ("__N__", str(len(items))), ("__DS__", a.dataset),
                 ("__TASK__", cfg["task"]), ("__S__", cfg["S"]), ("__B__", cfg["B"]), ("__F__", cfg["F"])):
        html = html.replace(k, v)
    sheet = out / f"{a.dataset}_check.html"
    sheet.write_text(html, encoding="utf-8")
    (out / f"{a.dataset}_check_key.json").write_text(json.dumps(key))
    print(f"{sheet} ({sheet.stat().st_size / 1e6:.1f} MB), {len(items)} moments, {per} of each kind: {kinds}; "
          f"{repeated} rows with a repeated picture")


if __name__ == "__main__":
    main()
