"""Build the guided relabelling site: every Ego-EXTRA and WTaG moment, 2 humans each.

The hand labels in reports/{egoextra,wtag}_hand_labels.csv come from one
labeller, the AI assistant. This script splits ALL moments of each
dataset into batches of about 70, in a random order (fixed seed), and
writes one page per batch. Each batch is to be labelled by two people
who work independently. A page shows:

    1. a welcome screen (full name, opt-in for the acknowledgements)
    2. a guide: the three labels, with 2 picture examples each
    3. 6 practice moments with feedback
    4. the batch, one moment at a time (S, B or F; keys 1, 2, 3)
    5. sending: to the collection endpoint (study/endpoint.js) when one
       is set, else a file to download and send

The page never shows the first label, the kind of moment (speech onset or
random second), the recording or the original id. The guide and practice
moments come from study/label/examples.json: other seconds of the same
recordings, at least 10 s from every moment of the sample.

The pages hold dataset frames, so the plain pages stay in
runs/label_check/relabel/ (not in git). The public files in study/label/
are encrypted like the check sheets (AES-256-GCM, PBKDF2 key) in a
.bin file next to a small loader page: the
passphrase is the part of the link after "#", which the browser never
sends. Full links: runs/label_check/relabel/links.md. Keys (batch
number -> original id): runs/label_check/relabel/keys/.

Usage:
    python -m scripts.relabel_build --base https://harshil-sri.github.io/ego-vision
Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from scripts.label_check_publish import ROUNDS
from scripts.relabel_frames import rows

VERSION = "relabel-v1"
BATCH = 70
ROOTS = {"egoextra": "C:/ssd-data/ego-extra", "wtag": "C:/ssd-data/wtag"}
INFO = {
    "egoextra": {
        "task": "bike workshop and furniture assembly", "aspect": "1 / 1",
        "credit": "Ego-EXTRA dataset",
        "defs": {
            "S": "Screw, unscrew or insert is happening: a screwdriver or hex key is turning, a screw or dowel is "
                 "going into its hole, a cap or valve is being turned.",
            "B": "Other precise hand work is happening: fitting a tyre or tube, hammering, using pliers, "
                 "pumping, wiping, aligning two parts.",
            "F": "Everything else: holding, carrying, reaching, picking up, putting down, looking, pointing, "
                 "or no hands in view.",
        },
    },
    "wtag": {
        "task": "cooking", "aspect": "16 / 9", "credit": "WTaG dataset",
        "defs": {
            "S": "Screw, unscrew or insert is happening: twisting a lid or cap, putting the filter into the "
                 "cone, pushing toothpicks into the wrap.",
            "B": "Other precise hand work is happening: pouring, spreading, cutting, scooping or measuring, "
                 "stirring, folding, squeezing, wiping.",
            "F": "Everything else: holding, carrying, reaching, picking up, putting down, looking, "
                 "or no hands in view.",
        },
    },
}
SHORT = {"S": "screw, unscrew, insert, twist a cap", "B": "other precise work", "F": "hold, reach, look, idle"}


SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>Hand labelling</title></head>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px">
<p id="m">Loading the pictures (about __MB__ MB). This can take some seconds.</p>
<script>
const SALT="__SALT__", ROUNDS=__ROUNDS__, DATA="__BIN__";
const b=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
(async()=>{
  const m=document.getElementById('m'), pass=decodeURIComponent(location.hash.slice(1));
  if(!pass){m.textContent='This link is not complete. Please open the full link from the message you received.';return;}
  let buf;
  try{ buf=new Uint8Array(await (await fetch(DATA)).arrayBuffer()); }
  catch(e){m.textContent='The pictures could not be loaded. Please check the internet connection and reload.';return;}
  try{
    const km=await crypto.subtle.importKey('raw',new TextEncoder().encode(pass),'PBKDF2',false,['deriveKey']);
    const key=await crypto.subtle.deriveKey({name:'PBKDF2',salt:b(SALT),iterations:ROUNDS,hash:'SHA-256'},km,
      {name:'AES-GCM',length:256},false,['decrypt']);
    const html=new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:buf.slice(0,12)},key,buf.slice(12)));
    document.open();document.write(html);document.close();
  }catch(e){m.textContent='The page could not be opened. Please copy the full link from the message again.';}
})();
</script>
</body></html>"""


def uri(img: np.ndarray, q: int = 78) -> str:
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q])
    if not ok or not img.any():
        raise SystemExit("empty picture row")
    return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode()


def encrypt(html: str, pw: str, page: Path) -> None:
    """Write page (a small loader) and page.bin (iv + AES-GCM ciphertext of html)."""
    salt, iv = os.urandom(16), os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ROUNDS, 32)
    data = iv + AESGCM(key).encrypt(iv, html.encode(), None)
    binf = page.with_suffix(".bin")
    binf.write_bytes(data)
    page.write_text(SHELL.replace("__SALT__", base64.b64encode(salt).decode()).replace("__ROUNDS__", str(ROUNDS))
                    .replace("__BIN__", binf.name).replace("__MB__", f"{len(data) / 1e6:.0f}"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="site address, e.g. https://user.github.io/repo")
    ap.add_argument("--datasets", nargs="+", default=["egoextra", "wtag"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--work", default="runs/label_check/relabel")
    ap.add_argument("--out", default="study/label")
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    for d in (work / "pages", work / "keys", out):
        d.mkdir(parents=True, exist_ok=True)
    pf = work / "passphrase.txt"
    if not pf.exists():
        pf.write_text(secrets.token_urlsafe(18))
    pw = pf.read_text().strip()
    template = Path("scripts/relabel_page.html").read_text(encoding="utf-8")
    examples = json.loads((out / "examples.json").read_text())
    links = []
    for ds in a.datasets:
        info, rng = INFO[ds], np.random.default_rng(a.seed)
        lab = pd.read_csv(f"reports/{ds}_hand_labels.csv", dtype={"recording": str})
        lab = lab.iloc[rng.permutation(len(lab))].reset_index(drop=True)
        nb = int(np.ceil(len(lab) / BATCH))
        lab["batch"] = np.arange(len(lab)) % nb + 1
        ex = examples[ds]
        ex_moments = [(x["recording"], x["t_s"]) for part in ("guide", "practice") for x in ex[part]]
        pics = rows(ds, ROOTS[ds], ex_moments + list(zip(lab.recording, lab.t_s)))
        ex_uri = [uri(p) for p in pics[:len(ex_moments)]]
        lab["img"] = [uri(p) for p in pics[len(ex_moments):]]
        ng = len(ex["guide"])
        guide = [{"img": ex_uri[i], "label": x["label"], "why": x["why"]} for i, x in enumerate(ex["guide"])]
        practice = [{"img": ex_uri[ng + i], "label": x["label"], "why": x["why"]} for i, x in enumerate(ex["practice"])]
        practice = [practice[i] for i in rng.permutation(len(practice))]  # not S, S, B, B, F, F
        for b in range(1, nb + 1):
            part = lab[lab.batch == b].reset_index(drop=True)
            items = [{"n": i + 1, "img": r.img} for i, r in part.iterrows()]
            data = {"version": VERSION, "ds": ds, "batch": b, "nb": nb, "task": info["task"], "aspect": info["aspect"],
                    "defs": info["defs"], "short": SHORT, "guide": guide, "practice": practice, "items": items}
            html = (template.replace("__DATA__", json.dumps(data)).replace("__CREDIT__", info["credit"]))
            (work / "pages" / f"{ds}-{b}.html").write_text(html, encoding="utf-8")
            (work / "keys" / f"{ds}-{b}.json").write_text(json.dumps({str(i + 1): int(r.id) for i, r in part.iterrows()}))
            page = out / f"{ds}-{b}.html"
            encrypt(html, pw, page)
            links.append((ds, b, len(part), f"{a.base.rstrip('/')}/{out.as_posix()}/{ds}-{b}.html#{pw}",
                          page.with_suffix(".bin").stat().st_size / 1e6))
        print(f"{ds}: {len(lab)} moments in {nb} batches")
    md = ["# Relabelling links (private: do not post publicly)", "",
          "Each batch needs TWO different people. Write their names here so nobody gets the same batch twice.", "",
          "| batch | moments | link | labeller 1 | labeller 2 |", "|---|---|---|---|---|"]
    md += [f"| {ds}-{b} | {n} | {url} | | |" for ds, b, n, url, _ in links]
    (work / "links.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(links)} pages in {out} (largest {max(x[4] for x in links):.1f} MB); links in {work / 'links.md'}")


if __name__ == "__main__":
    main()
