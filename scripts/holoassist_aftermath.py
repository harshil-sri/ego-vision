"""Pilot part 3: what do the hands do AROUND an instructor utterance?

Parts 1 and 2 looked at the single moment an utterance starts. This part
follows the performer's hands from 10 s before to 10 s after that moment.

For every instructor utterance onset t and every lag d in -10 .. +10 s:
    busy(t + d) = 1 if a fine-grained hand action covers time t + d, else 0
Averaging over onsets gives a curve "probability the hands are busy, as a
function of time relative to the onset".

Null model: the same per-video circular shift as part 1 (all onsets of a
video moved by one random offset). The null curve is what a speaker who
ignored the hands would produce.

Three conditions, each compared with its own null:
    all            every onset
    busy_at_onset  onsets that fall inside a hand action. The null is
                   shifted onsets that also fall inside a hand action, so
                   "the hands were busy at d = 0" is true for both.
    idle_at_onset  onsets that fall in a gap between actions

Two readings:
    before the onset: selection. A dip means the speaker waited for (or
        picked) a moment when the hands were winding down.
    after the onset: aftermath. A dip means the hands stopped after the
        speech began. That is either the performer stopping to listen (a
        cost of the interruption) or the instruction telling them to stop.
        These labels cannot separate the two.

Summary number per condition: "idle seconds" = sum over lags of
(1 - busy) x step, for the 5 s after and the 5 s before the onset, each
minus its null value. Positive = more idle time than chance.

Usage:
    python -m scripts.holoassist_aftermath --perms 500
Outputs: reports/holoassist_aftermath.json, reports/holoassist_aftermath.md,
         reports/figures/holoassist_aftermath.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.holoassist_timing import GROUP_NAMES, GROUPS, LABELS, build_video
from src.paths import resolve_data_path

STEP = 0.5
LAGS = np.arange(-10.0, 10.0 + STEP / 2, STEP)          # (41,)
CONDS = ["all", "busy_at_onset", "idle_at_onset"]


def busy_at(V: dict, t: np.ndarray) -> np.ndarray:
    """1.0 where a hand action covers time t, 0.0 where idle, nan outside the
    task window. t: any shape."""
    k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
    out = (~np.isnan(V["a_start"][k])).astype(np.float64)
    out[(t < V["lo"]) | (t >= V["hi"])] = np.nan
    return out


def utterance_durations(data: list) -> dict:
    """Median instructor utterance length in seconds, per group."""
    d = {g: [] for g in GROUPS}
    for v in data:
        for e in v["events"]:
            if e["label"] != "Conversation":
                continue
            p = e["attributes"].get("Conversation Purpose", "")
            for g, name in GROUPS.items():
                if p == name:
                    d[g].append(e["end"] - e["start"])
    return {g: round(float(np.median(x)), 2) for g, x in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    P, G, L, C = args.perms, len(GROUP_NAMES), len(LAGS), len(CONDS)

    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"]
                     if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    videos = [V for V in (build_video(v, top) for v in data) if V is not None]
    print(f"{len(videos)}/{len(data)} videos usable")

    # index 0 = observed, 1..P = shifted copies
    s_busy = np.zeros((C, P + 1, G, L)); s_n = np.zeros((C, P + 1, G, L))
    n_on = np.zeros((C, G))
    for V in videos:
        W = V["hi"] - V["lo"]
        off = np.concatenate([[0.0], rng.uniform(0, W, P)])                  # (P+1,)
        for g, name in enumerate(GROUP_NAMES):
            on = V["onsets"][name]
            if len(on) == 0:
                continue
            t = V["lo"] + (on[None, :] - V["lo"] + off[:, None]) % W        # (P+1, n)
            b = busy_at(V, t[:, :, None] + LAGS[None, None, :])              # (P+1, n, L)
            at0 = busy_at(V, t)                                              # (P+1, n)
            ok = ~np.isnan(b)
            b0 = np.where(ok, b, 0.0)
            for c, mask in enumerate([np.ones_like(at0, bool), at0 == 1, at0 == 0]):
                m = mask[:, :, None]
                s_busy[c, :, g] += (b0 * m).sum(axis=1)
                s_n[c, :, g] += (ok & m).sum(axis=1)
                n_on[c, g] += mask[0].sum()

    curve = s_busy / np.maximum(s_n, 1)                                      # (C, P+1, G, L)
    post = (LAGS > 0) & (LAGS <= 5.0)
    pre = (LAGS < 0) & (LAGS >= -5.0)
    idle_post = ((1 - curve)[..., post]).sum(-1) * STEP                      # (C, P+1, G)
    idle_pre = ((1 - curve)[..., pre]).sum(-1) * STEP
    asym = idle_post - idle_pre

    def summarise(x):          # x (P+1,) -> observed, null mean, 95% band, p
        obs, null = x[0], x[1:]
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (len(null) + 1)
        return {"observed": round(float(obs), 3), "null_mean": round(float(null.mean()), 3),
                "null_95": [round(float(np.percentile(null, 2.5)), 3),
                            round(float(np.percentile(null, 97.5)), 3)],
                "excess": round(float(obs - null.mean()), 3), "p": round(float(p), 5)}

    out = {"videos": len(videos), "perms": P, "lags_s": LAGS.tolist(),
           "utterance_median_s": utterance_durations(data), "groups": {}}
    for g, name in enumerate(GROUP_NAMES):
        out["groups"][name] = {}
        for c, cond in enumerate(CONDS):
            out["groups"][name][cond] = {
                "onsets": int(n_on[c, g]),
                "busy_observed": np.round(curve[c, 0, g], 4).tolist(),
                "busy_null_mean": np.round(curve[c, 1:, g].mean(0), 4).tolist(),
                "busy_null_lo": np.round(np.percentile(curve[c, 1:, g], 2.5, axis=0), 4).tolist(),
                "busy_null_hi": np.round(np.percentile(curve[c, 1:, g], 97.5, axis=0), 4).tolist(),
                "idle_s_5s_after": summarise(idle_post[c, :, g]),
                "idle_s_5s_before": summarise(idle_pre[c, :, g]),
                "after_minus_before": summarise(asym[c, :, g]),
            }
    Path("reports/holoassist_aftermath.json").write_text(json.dumps(out, indent=2))

    md = ["# HoloAssist pilot part 3: the hands around an instructor utterance", "",
          "Generated by `python -m scripts.holoassist_aftermath`. "
          f"{len(videos)} videos, null = per-video circular shift, {P} repetitions. "
          "Idle seconds = time with no annotated hand action, summed over the 5 s window. "
          "Excess = observed minus the null mean. See the script docstring for the two readings.", "",
          f"Median utterance length (s): {out['utterance_median_s']}", "",
          "![busy probability around onsets](figures/holoassist_aftermath.png)", ""]
    for cond in CONDS:
        md += [f"## {cond}", "",
               "| group | onsets | idle s in 5 s BEFORE: observed / null / excess (p) | "
               "idle s in 5 s AFTER: observed / null / excess (p) | after minus before: excess (p) |",
               "|---|---|---|---|---|"]
        for name in GROUP_NAMES:
            r = out["groups"][name][cond]
            b, a, d = r["idle_s_5s_before"], r["idle_s_5s_after"], r["after_minus_before"]
            md.append(f"| {name} | {r['onsets']} | {b['observed']} / {b['null_mean']} / {b['excess']:+.3f} ({b['p']}) | "
                      f"{a['observed']} / {a['null_mean']} / {a['excess']:+.3f} ({a['p']}) | "
                      f"{d['excess']:+.3f} ({d['p']}) |")
        md.append("")
    Path("reports/holoassist_aftermath.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    plot(out)


def plot(out: dict) -> None:
    """Small multiples: one panel per (condition, group). Same y axis everywhere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, GREY, INK, MUTED, SURFACE = "#2a78d6", "#8a8983", "#0b0b0b", "#52514e", "#fcfcfb"
    groups = list(GROUPS)                                   # 4 purposes; skip the pooled group
    titles = {"follow_up": "Follow-up instruction", "high_level": "High-level instruction",
              "confirm": "Confirmation", "correct": "Correction"}
    rows = {"all": "All onsets", "busy_at_onset": "Hands busy at onset",
            "idle_at_onset": "Hands idle at onset"}
    lags = np.array(out["lags_s"])
    fig, axes = plt.subplots(len(CONDS), len(groups), figsize=(13, 8), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for i, cond in enumerate(CONDS):
        for j, g in enumerate(groups):
            ax, r = axes[i, j], out["groups"][g][cond]
            ax.set_facecolor(SURFACE)
            ax.fill_between(lags, r["busy_null_lo"], r["busy_null_hi"], color=GREY, alpha=0.3, lw=0)
            ax.plot(lags, r["busy_null_mean"], color=GREY, lw=1.5, label="Chance (shifted onsets, 95% band)")
            ax.plot(lags, r["busy_observed"], color=BLUE, lw=2, label="Observed")
            ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c9c8c2")
            ax.tick_params(colors=MUTED, labelsize=9)
            if i == 0:
                ax.set_title(titles[g], fontsize=11, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"{rows[cond]}\nP(hands busy)", fontsize=9.5, color=INK)
            if i == len(CONDS) - 1:
                ax.set_xlabel("Seconds from utterance onset", fontsize=9.5, color=MUTED)
            ax.text(0.98, 0.04, f"n = {r['onsets']:,}", transform=ax.transAxes, ha="right",
                    fontsize=8.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Performer hand activity around instructor speech (HoloAssist)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/holoassist_aftermath.png", dpi=150, facecolor=SURFACE)
    print("wrote reports/figures/holoassist_aftermath.png")


if __name__ == "__main__":
    main()
