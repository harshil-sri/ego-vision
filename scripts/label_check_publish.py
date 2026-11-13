"""Put the blind check sheets on the web without publishing dataset frames.

The sheets of scripts.label_check_kit hold frames of Ego-EXTRA and WTaG.
The repository and its GitHub Pages site are public, and the terms of the
two datasets do not clearly allow public copies of frames. So the public
file holds the sheet only in encrypted form (AES-256-GCM, key from a
passphrase with PBKDF2-SHA256, 600,000 rounds). The passphrase is in the
part of the link after "#", which a browser never sends to the server.
Only a person who gets the full link can see the frames.

The passphrase is made once and kept in runs/label_check/passphrase.txt
(not in git). The full links are printed and written to
runs/label_check/links.txt. Do not put them in a public place.

Usage:  python -m scripts.label_check_publish --base https://<user>.github.io/<repo>
Output: study/check/<dataset>.html (safe to commit)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROUNDS = 600_000
SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>Hand label check</title></head>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px">
<p id="m">Opening the sheet. This can take some seconds.</p>
<script>
const SALT="__SALT__", IV="__IV__", ROUNDS=__ROUNDS__;
const b=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
(async()=>{
  const m=document.getElementById('m'), pass=decodeURIComponent(location.hash.slice(1));
  if(!pass){m.textContent='This link is not complete. Please open the full link from the message you received.';return;}
  try{
    const km=await crypto.subtle.importKey('raw',new TextEncoder().encode(pass),'PBKDF2',false,['deriveKey']);
    const key=await crypto.subtle.deriveKey({name:'PBKDF2',salt:b(SALT),iterations:ROUNDS,hash:'SHA-256'},km,
      {name:'AES-GCM',length:256},false,['decrypt']);
    const data=b(document.getElementById('d').textContent.trim());
    const html=new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:b(IV)},key,data));
    document.open();document.write(html);document.close();
  }catch(e){m.textContent='The sheet could not be opened. Please copy the full link from the message again.';}
})();
</script>
<script id="d" type="application/octet-stream">__DATA__</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="site address, for example https://user.github.io/repo")
    ap.add_argument("--src", default="runs/label_check")
    ap.add_argument("--out", default="study/check")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pf = src / "passphrase.txt"
    if not pf.exists():
        pf.write_text(secrets.token_urlsafe(18))
    pw = pf.read_text().strip()
    links = []
    for sheet in sorted(src.glob("*_check.html")):
        name = sheet.name.replace("_check.html", "")
        salt, iv = os.urandom(16), os.urandom(12)
        key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ROUNDS, 32)
        data = AESGCM(key).encrypt(iv, sheet.read_bytes(), None)
        e = lambda x: base64.b64encode(x).decode()
        page = (SHELL.replace("__SALT__", e(salt)).replace("__IV__", e(iv))
                .replace("__ROUNDS__", str(ROUNDS)).replace("__DATA__", e(data)))
        (out / f"{name}.html").write_text(page, encoding="utf-8")
        links.append(f"{name}: {a.base.rstrip('/')}/{out.as_posix()}/{name}.html#{pw}")
    (src / "links.txt").write_text("\n".join(links) + "\n")
    print(f"{len(links)} sheets written to {out}; full links in {src / 'links.txt'}")


if __name__ == "__main__":
    main()
