"""Download a SUBSET of the HoloAssist videos without fetching the 155 GB archive.

The videos ship as one tar file on a server that supports HTTP range
requests. A tar file is a chain of 512-byte headers, each followed by its
file's bytes. So:

  stage "index"     read header, jump over the file body, read the next
                    header, ... Only headers are downloaded (a few MB in
                    total). Result: name, byte offset and size of every
                    file -> <root>/video_index.json
  stage "download"  pick whole recording sessions at random (fixed seed)
                    until --videos usable videos are reached, then fetch
                    just those byte ranges -> <root>/videos/<name>.mp4

"Usable" = the video passes the same filter as the pilot analyses
(scripts.holoassist_timing.build_video). Sessions are kept whole so that
later splits by session are clean.

Usage:
    python -m scripts.holoassist_download_subset index
    python -m scripts.holoassist_download_subset download --videos 300
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.paths import resolve_data_path

URL = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_compress.tar"
ROOT = "/mnt/ssd/holoassist"
BLOCK = 512


def _octal(field: bytes) -> int:
    s = field.strip(b"\0 ").decode("ascii", "replace")
    return int(s, 8) if s else 0


def build_index(out: Path) -> None:
    """Walk the tar headers with range requests."""
    import httpx
    files, off, pax = [], 0, {}
    with httpx.Client(timeout=60, http2=False) as c:
        total = int(c.head(URL).headers["content-length"])
        t0 = time.time()
        while off + BLOCK <= total:
            # one request covers the header and, for the small metadata entries, their body
            r = c.get(URL, headers={"Range": f"bytes={off}-{off + 4 * BLOCK - 1}"})
            r.raise_for_status()
            h = r.content[:BLOCK]
            if h == b"\0" * BLOCK:                      # end-of-archive marker
                break
            name = h[:100].split(b"\0")[0].decode("utf-8", "replace")
            prefix = h[345:500].split(b"\0")[0].decode("utf-8", "replace")
            if prefix:
                name = prefix + "/" + name
            size, kind = _octal(h[124:136]), chr(h[156] or 48)
            body = off + BLOCK
            if kind == "x":                             # pax header: overrides for the NEXT entry
                data = r.content[BLOCK:BLOCK + size]
                if len(data) < size:
                    data = c.get(URL, headers={"Range": f"bytes={body}-{body + size - 1}"}).content
                pax = {}
                while data:
                    n = int(data.split(b" ", 1)[0])
                    k, v = data[:n].split(b" ", 1)[1].rstrip(b"\n").split(b"=", 1)
                    pax[k.decode()] = v.decode("utf-8", "replace")
                    data = data[n:]
            else:
                name = pax.get("path", name)
                size = int(pax.get("size", size))
                if kind in ("0", "\0"):
                    files.append({"name": name, "offset": body, "size": size})
                    if len(files) % 200 == 0:
                        print(f"{len(files)} files, at {off / total:.1%} ({time.time() - t0:.0f}s)", flush=True)
                pax = {}
            off = body + (size + BLOCK - 1) // BLOCK * BLOCK
    out.write_text(json.dumps({"url": URL, "total_bytes": total, "files": files}))
    print(f"indexed {len(files)} files -> {out}")


def download(root: Path, n_videos: int, seed: int, workers: int = 12) -> None:
    import httpx
    from scripts.holoassist_timing import LABELS, build_video
    index = json.loads((root / "video_index.json").read_text())
    by_video = {}
    for f in index["files"]:
        if f["name"].lower().endswith(".mp4"):
            by_video[f["name"].split("/")[0]] = f
    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    usable = [v["video_name"] for v in data if v["video_name"] in by_video and build_video(v, top) is not None]
    sessions = sorted({v.split("-")[0] for v in usable})
    np.random.default_rng(seed).shuffle(sessions)
    chosen = []
    for s in sessions:
        if len(chosen) >= n_videos:
            break
        chosen += [v for v in usable if v.split("-")[0] == s]
    total = sum(by_video[v]["size"] for v in chosen)
    print(f"{len(usable)} usable videos in the archive; chose {len(chosen)} videos "
          f"from {len({v.split('-')[0] for v in chosen})} sessions, {total / 2**30:.1f} GB", flush=True)
    out = root / "videos"
    out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0, done = time.time(), 0

    def fetch(v: str) -> int:
        """One video = one byte range. Returns bytes downloaded (0 if already there)."""
        f, dst = by_video[v], out / f"{v}.mp4"
        if dst.exists() and dst.stat().st_size == f["size"]:
            return 0
        tmp = dst.with_suffix(".part")
        for attempt in range(5):
            try:
                with c.stream("GET", index["url"], headers={
                        "Range": f"bytes={f['offset']}-{f['offset'] + f['size'] - 1}"}) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_bytes(1 << 20):
                            fh.write(chunk)
                if tmp.stat().st_size == f["size"]:
                    tmp.replace(dst)
                    return f["size"]
            except httpx.HTTPError as e:
                print(f"retry {attempt + 1} for {v}: {e!r}", flush=True)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"download failed: {v}")

    # the server is slow per connection, so several videos are fetched at once
    with httpx.Client(timeout=120, limits=httpx.Limits(max_connections=workers)) as c, \
            ThreadPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(fetch, v) for v in chosen])):
            done += fut.result()
            if (i + 1) % 10 == 0:
                rate = done / (time.time() - t0) / 2**20
                print(f"{i + 1}/{len(chosen)} videos  {done / 2**30:.1f} GB new  {rate:.1f} MB/s", flush=True)
    (root / "videos_subset.json").write_text(json.dumps({"seed": seed, "videos": chosen}, indent=1))
    print(f"done: {len(chosen)} videos in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["index", "download"])
    ap.add_argument("--videos", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12, help="parallel connections")
    args = ap.parse_args()
    root = Path(resolve_data_path(ROOT))
    if args.stage == "index":
        build_index(root / "video_index.json")
    else:
        download(root, args.videos, args.seed, args.workers)


if __name__ == "__main__":
    main()
