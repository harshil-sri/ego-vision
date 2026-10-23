"""Batch-1 inference latency benchmark for the verb model (WP3).

Grid: backbone {resnet18, resnet34, mobilenet_v2} x T {4, 8, 16} x
precision {fp32, fp16}. Each cell: 5 warmup + 30 timed forward passes of a
randomly-initialised VerbModel on ONE clip (batch 1, (1,T,3,img,img)),
reporting mean/min milliseconds and windows/s (= 1000 / mean_ms).

Measured variant is tsn_avg by default (the backbone runs on ALL T frames —
the conservative case); single_frame runs the backbone on 1 frame only, so
its cost is the T=1 backbone cost plus a tiny head (NOT measured here, not
claimed). fp16 is skipped on CPU (half precision has no CPU speedup and
some ops are unsupported) with a printed note.

Writes:
    reports/latency.md      human table (with the device name)
    runs/latency/latency.json  machine-readable results

Usage:
    python -m scripts.latency_bench --device cpu --img-size 96   # smoke/CPU
    python -m scripts.latency_bench --device cuda --img-size 224  # RTX 4060
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "reports" / "latency.md"
JSON_PATH = REPO_ROOT / "runs" / "latency" / "latency.json"

WARMUP_ITERS = 5   # untimed passes (lazy init, caches) before measurement
TIMED_ITERS = 30   # timed passes averaged into mean/min ms
BACKBONES = ("resnet18", "resnet34", "mobilenet_v2")
T_VALUES = (4, 8, 16)
PRECISIONS = ("fp32", "fp16")


def _device_name(device) -> str:
    """Honest device label for the report.

    Input: torch.device. Output: str — CUDA card name via torch on cuda;
    on CPU the /proc/cpuinfo 'model name' (Linux/WSL2), falling back to
    platform.machine(). Measured from the machine, never guessed.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:  # first 'model name' line = the CPU
                if line.startswith("model name"):
                    return f"CPU ({line.split(':', 1)[1].strip()})"
    except OSError:
        pass
    return f"CPU ({platform.machine()})"


def _bench_one(model, clip, device) -> tuple[float, float]:
    """Time one model/clip configuration.

    Inputs: model nn.Module already on `device` and in eval(); clip tensor
        (1,T,3,H,W) already on `device` in the right dtype; device.
    Output: (mean_ms float, min_ms float) over TIMED_ITERS forwards,
        measured with time.perf_counter around each call (CUDA paths are
        synchronised before and after each timed call).
    """
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            model(clip)  # (1,T,3,H,W) -> (1,num_verbs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms: list[float] = []
        for _ in range(TIMED_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(clip)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return sum(times_ms) / len(times_ms), min(times_ms)


def main() -> None:
    """CLI entry point. Outputs: reports/latency.md + runs/latency/
    latency.json with the measured grid; prints the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--img-size", type=int, default=224,
                        help="square input resolution (short-side proxy)")
    parser.add_argument("--variant", default="tsn_avg",
                        choices=["single_frame", "tsn_avg", "temporal_attn"],
                        help="variant to measure (default tsn_avg: backbone "
                             "runs on ALL T frames)")
    args = parser.parse_args()

    from src.models.variants import VerbModel  # torch imported at top

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")
    torch.manual_seed(0)  # fixed input/model seed; latency is seed-stable

    dev_name = _device_name(device)
    results: list[dict] = []
    skipped: list[dict] = []

    for backbone in BACKBONES:
        for T in T_VALUES:
            for precision in PRECISIONS:
                label = f"{backbone} T={T} {precision}"
                if device.type == "cpu" and precision == "fp16":
                    note = "skipped on CPU: fp16 has no CPU speedup and "
                    "several ops are unsupported"
                    print(f"{label}: {note}")
                    skipped.append({"backbone": backbone, "T": T,
                                    "precision": precision, "reason": note})
                    continue
                # Random init (pretrained=False): weights content does not
                # change latency, and no download happens.
                model = VerbModel(backbone, num_verbs=8, variant=args.variant,
                                  pretrained=False)
                model.eval()  # BN running stats; the inference-time mode
                model = model.to(device)
                dtype = torch.float16 if precision == "fp16" else torch.float32
                model = model.to(dtype)
                clip = torch.randn(1, T, 3, args.img_size, args.img_size,
                                   dtype=dtype, device=device)
                mean_ms, min_ms = _bench_one(model, clip, device)
                windows_per_s = 1000.0 / mean_ms
                results.append({
                    "backbone": backbone, "T": T, "precision": precision,
                    "mean_ms": round(mean_ms, 3), "min_ms": round(min_ms, 3),
                    "windows_per_s": round(windows_per_s, 2),
                })
                print(f"{label}: mean {mean_ms:.1f} ms | min {min_ms:.1f} ms "
                      f"| {windows_per_s:.2f} windows/s")

    # ---- reports -----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Verb-model latency (batch 1)",
        "",
        f"Measured: {timestamp}, device: `{dev_name}`, img_size "
        f"{args.img_size}x{args.img_size}, variant `{args.variant}`, "
        f"{WARMUP_ITERS} warmup + {TIMED_ITERS} timed forwards per cell.",
        "Command: `python -m scripts.latency_bench --device "
        f"{args.device} --img-size {args.img_size}`",
        "",
        "Note: measured with variant=tsn_avg, whose backbone runs on ALL T "
        "frames. single_frame runs the backbone on ONE frame, so its cost "
        "is close to the backbone cost at one frame plus a linear head "
        "(not measured separately here).",
        "",
        "| backbone | T | precision | mean ms | min ms | windows/s |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| {r['backbone']} | {r['T']} | {r['precision']} | "
                     f"{r['mean_ms']:.2f} | {r['min_ms']:.2f} | "
                     f"{r['windows_per_s']:.2f} |")
    if skipped:
        lines += ["", "Skipped cells:"]
        for s in skipped:
            lines.append(f"- {s['backbone']} T={s['T']} {s['precision']}: "
                         f"{s['reason']}")
    if device.type == "cpu":
        lines += ["",
                  "RTX 4060: NOT RUN (owner asleep; run "
                  "scripts/latency_bench.py --device cuda --img-size 224 "
                  "in the morning)"]
    else:
        lines += ["", f"GPU: {dev_name} (this run WAS on cuda)."]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "device": dev_name,
                   "device_type": device.type, "img_size": args.img_size,
                   "variant": args.variant,
                   "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
                   "results": results, "skipped": skipped}, f, indent=2)
    print(f"wrote {REPORT_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
