"""Overlay drawing shared by the web MJPEG stream and the OpenCV fallback
UI (--ui cv). One drawing function -> identical overlays in both UIs.

Style: dark translucent panels, warm accent, large legible type for
screen recording. Text is templated only — no LLM anywhere in the loop
(out-of-scope rule).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

ACCENT = (94, 197, 255)      # warm accent (BGR: light orange-ish blue-free)
WARN = (64, 64, 255)         # calm red for wrong-order
OK = (128, 212, 128)         # green for done

def _panel(img, x, y, w, h, alpha=0.55):
    """Translucent dark rounded-ish panel; mutates img in place."""
    overlay = img[y:y + h, x:x + w]
    dark = np.zeros_like(overlay)
    cv2.addWeighted(dark, 1 - alpha, overlay, alpha, 0, overlay)

def draw_overlay(frame: np.ndarray, engine) -> np.ndarray:
    """Draw the current engine state onto one BGR frame.

    frame: (H, W, 3) uint8 BGR (mutated copy owned by caller); returns it.
    Shows: verb + confidence chip, ingredient chip, phase/state line,
    scan progress bar, wrong-order banner (3 s), show-zone rect (debug),
    metrics line (fps / latency / dropped).
    """
    H, W = frame.shape[:2]
    st = engine.state()
    m = st["metrics"]
    scale = max(H / 480.0, 0.8)          # keep type legible at any size
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- show zone rectangle (ingredient policy D4 visual contract)
    if engine.cfg.ingredient_crop == "show_zone":
        rx, ry, rw, rh = engine.cfg.show_zone
        x0, y0 = int(rx * W), int(ry * H)
        x1, y1 = int((rx + rw) * W), int((ry + rh) * H)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

    # --- top-left chips: verb + ingredient
    chip_y = 8
    verb, conf = st["verb"]["name"], st["verb"]["confidence"]
    ing, score = st["ingredient"]["name"], st["ingredient"]["score"]
    line1 = f"{verb or '-'} {conf:.2f}" if verb else "- "
    line2 = f"{ing} {score:.2f}" if ing != "unknown" else "unknown"
    _panel(frame, 8, chip_y, int(280 * scale), int(74 * scale))
    cv2.putText(frame, f"VERB  {line1}", (int(18 * scale), int(34 * scale)),
                font, 0.62 * scale, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"ING   {line2}", (int(18 * scale), int(64 * scale)),
                font, 0.62 * scale, (235, 235, 235), 2, cv2.LINE_AA)

    # --- state / next-step line (bottom)
    steps = st["recipe"]["steps"]
    cur = next((i for i, s in enumerate(steps) if s["current"]), None)
    if st["phase"] in ("cooking", "done"):
        if st["phase"] == "done":
            msg = f"Recipe '{st['recipe']['name']}' complete"
        else:
            s = steps[cur]
            note = f" ({s['note']})" if s["note"] else ""
            msg = f"Next: {s['verb']} {s['ingredient']}{note}"
    elif st["phase"] == "scanning" or st["scan"].get("ingredient"):
        scan_ing = st["scan"].get("ingredient")
        msg = f"Hold up: {scan_ing}" if scan_ing else "Scan phase"
    else:
        msg = "Idle - start a session"
    _panel(frame, 8, H - int(44 * scale), W - 16, int(36 * scale))
    cv2.putText(frame, msg, (int(18 * scale), H - int(18 * scale)),
                font, 0.62 * scale, (245, 245, 245), 2, cv2.LINE_AA)

    # --- scan progress bar
    scan = st["scan"]
    if scan.get("ingredient"):
        captured = int(scan.get("captured", 0))
        target = int(scan.get("target", 1)) or 1
        frac = min(captured / target, 1.0)
        bar_w = int(W * 0.5)
        x0, y0 = (W - bar_w) // 2, int(10 * scale)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + int(14 * scale)),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * frac), y0 + int(14 * scale)),
                      ACCENT, -1)
        cv2.putText(frame, f"{scan['ingredient']} {captured}/{target}",
                    (x0, y0 + int(38 * scale)), font, 0.55 * scale,
                    (235, 235, 235), 1, cv2.LINE_AA)

    # --- wrong-order banner (calm, ~3 s)
    if st.get("warning"):
        _panel(frame, 8, int(90 * scale), W - 16, int(40 * scale), alpha=0.65)
        cv2.putText(frame, st["warning"]["message"][:78],
                    (int(18 * scale), int(118 * scale)), font,
                    0.55 * scale, WARN, 2, cv2.LINE_AA)

    # --- metrics line (debug legibility; the web debug drawer has the rest)
    mline = (f"{m['fps']:.0f}fps {m['inference_ms']:.0f}ms "
             f"drop:{m['dropped']} {m['smoother']}>{m['min_conf']}")
    cv2.putText(frame, mline, (W - int(300 * scale), int(22 * scale)),
                font, 0.45 * scale, (170, 170, 170), 1, cv2.LINE_AA)
    return frame
