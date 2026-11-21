"""Render a shareable demo video: the HoloAssist minute with the synced timeline under it.

    python -m scripts.demo_video          # needs site/media/demo.mp4 and demo.json (scripts.demo_data)
    -> runs/demo/demo_timeline.mp4  (1280x1080 at 25 fps, 3 s title card, no sound: the HoloAssist mp4
       files carry none, so the instructor's transcribed words and ProAssist's words are shown as captions)

The timeline shows the labelled hand state, the hand-state model's probability of fine-motor work
(trained without this recording session), and when the instructor and ProAssist start to speak.
A label flashes when a speech start passes the playhead.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MEDIA = Path("site/media")
W, H, VH, FPS, TITLE_S = 1280, 1080, 720, 25, 3.0
BG, INK, INK2, RULE = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 223)
BLUE, ORANGE, VIOLET = (42, 120, 214), (235, 104, 52), (98, 80, 214)
SEG = {"idle": (239, 238, 233), "fine": ORANGE, "hold": (201, 200, 194), "other": (221, 220, 215)}
LW, PX0, PX1 = 230, 250, 1240                      # label column, plot x range


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def base_panel(D) -> tuple[Image.Image, callable, dict]:
    """The static part of the timeline panel (H - VH tall)."""
    img = Image.new("RGB", (W, H - VH), BG)
    d = ImageDraw.Draw(img)
    X = lambda t: PX0 + t / D["duration_s"] * (PX1 - PX0)
    rows = {"inst": (40, 78), "pa": (92, 130), "hand": (150, 200), "model": (214, 292)}
    f = font(24)
    for key, lab in (("inst", "instructor starts"), ("pa", "ProAssist starts"), ("hand", "hands (labels)"),
                     ("model", "model: fine work?")):
        y0, y1 = rows[key]
        d.text((24, (y0 + y1) / 2), lab, fill=INK2, font=f, anchor="lm")
    for s, e, g in D["segments"]:
        d.rectangle([X(s), rows["hand"][0], max(X(s) + 1, X(e) - 2), rows["hand"][1]], fill=SEG[g])
    m0, m1 = rows["model"]
    d.line([PX0, m1, PX1, m1], fill=RULE, width=2)
    for x in np.arange(PX0, PX1, 14):
        d.line([x, (m0 + m1) / 2, x + 6, (m0 + m1) / 2], fill=RULE, width=2)
    pts = [(X(t), m1 - p * (m1 - m0)) for t, p in D["p_fine"]]
    d.line(pts, fill=ORANGE, width=4, joint="curve")
    for t in D["instructor"]:
        d.rectangle([X(t) - 2, rows["inst"][0], X(t) + 2, rows["inst"][1]], fill=BLUE)
    for t in D["proassist"]:
        d.rectangle([X(t) - 2, rows["pa"][0], X(t) + 2, rows["pa"][1]], fill=VIOLET)
    lf, lx = font(20), 24
    for lab, c in (("screw, unscrew, insert", ORANGE), ("hold, place, inspect", SEG["hold"]),
                   ("other action", SEG["other"]), ("idle", SEG["idle"])):
        d.rectangle([lx, 318, lx + 18, 336], fill=c, outline=RULE)
        d.text((lx + 26, 327), lab, fill=INK2, font=lf, anchor="lm")
        lx += 40 + d.textlength(lab, font=lf)
    d.text((24, 336), "", font=lf)
    return img, X, rows


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w_ in text.split():
        nxt = (cur + " " + w_).strip()
        if d.textlength(nxt, font=f) > width and cur:
            lines.append(cur); cur = w_
        else:
            cur = nxt
    return lines + ([cur] if cur else [])


def caption(img: Image.Image, t: float, D) -> None:
    """Draw the words being spoken at time t over the bottom of the video frame."""
    d = ImageDraw.Draw(img, "RGBA")
    f, small = font(30), font(22, True)
    boxes = [("INSTRUCTOR", x, BLUE) for s, e, x in D["instructor_text"] if s <= t < max(e, s + 1.5)]
    boxes += [("PROASSIST", x, VIOLET) for s, x in D["proassist_text"] if s <= t < s + 3.0]
    y = VH - 24
    for who, text, c in reversed(boxes):
        lines = wrap(d, text.replace("*unintelligible*", "[unclear]"), f, W - 220)[:2]
        h = 34 * len(lines) + 46
        y -= h
        d.rounded_rectangle([60, y, W - 60, y + h - 10], radius=12, fill=(11, 11, 11, 200))
        d.text((84, y + 10), who, fill=c if c != BLUE else (110, 167, 236), font=small)
        for i, line in enumerate(lines):
            d.text((84, y + 38 + 34 * i), line, fill=(255, 255, 255), font=f)
        y -= 8


def title_card(D) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 360), "Hands before words", fill=INK, font=font(72, True))
    d.multiline_text((80, 470), "Human instructors rarely start talking while the hands\nscrew, unscrew or insert. "
                     "Does an assistant need to wait too?", fill=INK2, font=font(36), spacing=14)
    d.text((80, 640), f"HoloAssist video {D['video']}: one minute, with the instructor's speech and ProAssist's",
           fill=INK2, font=font(26))
    d.text((80, 990), "Video: HoloAssist (Wang et al., ICCV 2023), CDLA-Permissive-2.0", fill=(138, 137, 132), font=font(22))
    return np.array(img)


def main() -> None:
    D = json.loads((MEDIA / "demo.json").read_text())
    panel, X, rows = base_panel(D)
    cap = cv2.VideoCapture(str(MEDIA / "demo.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Path("runs/demo/demo_timeline.mp4")          # for sharing; too large for git
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23",
                             "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE)
    card = title_card(D)
    for _ in range(int(TITLE_S * FPS)):
        proc.stdin.write(card.tobytes())
    big = font(28, True)
    frames = []
    ok, fr = cap.read()
    while ok:
        frames.append(fr)
        ok, fr = cap.read()
    n_out = int(D["duration_s"] * FPS)
    for i in range(n_out):
        t = i / FPS
        fr = frames[min(int(t * src_fps), len(frames) - 1)]
        top = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        caption(top, t, D)
        top = np.array(top)
        p = panel.copy()
        d = ImageDraw.Draw(p)
        x = X(t)
        d.rectangle([x - 2, 20, x + 2, rows["model"][1] + 6], fill=INK)
        for arr, lab, c in ((D["instructor"], "instructor speaks", BLUE), (D["proassist"], "ProAssist speaks", VIOLET)):
            if any(0 <= t - s < 1.2 for s in arr):
                bw = d.textlength(lab, font=big) + 32
                y = 8 if c == BLUE else 46
                fx = min(max(x + 10, PX0), W - bw - 10)
                d.rounded_rectangle([fx, y, fx + bw, y + 36], radius=8, fill=c)
                d.text((fx + 16, y + 18), lab, fill=(255, 255, 255), font=big, anchor="lm")
        proc.stdin.write(np.concatenate([top, np.array(p)], 0).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB, {n_src} source frames)")


if __name__ == "__main__":
    main()
