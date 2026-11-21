"""Make the paper's figures from the reports (and, for figure 1 and the HoloAssist intervals, the labels).

    python -m scripts.paper_figures            # all figures -> paper/figures/figN_*.pdf and .png

Figures
  1  one HoloAssist recording as a timeline: hand state and routine instructor onsets
  2  fine-motor ratio across the three datasets and both speakers, with 95% intervals
  3  (a) held, then released: odds of a pending instruction by time since the hands became free
     (b) what follows speech during fine-motor work against other actions
  4  ProAssist against the human instructors of the same videos, by hand state
  5  rating study: mean rating per cell and held minus now (round 2 if its report exists, else the pilot)

Colours: validated categorical palette (dataviz skill, light mode): blue #2a78d6, orange #eb6834,
aqua #1baf7a; grey for chance and reference lines. Speaker identity is also given by marker shape
and by direct labels, never by colour alone.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.holoassist_need_controlled import group_ids
from scripts.holoassist_student_speech import event_type
from scripts.holoassist_timing import LABELS, build_video
from src.paths import resolve_data_path

OUT = Path("paper/figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, CHANCE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
R = lambda f: json.loads(Path(f).read_text(encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def chance_line(ax, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(1.0, color=CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=0)


# --------------------------------------------------------------------------- HoloAssist intervals
def holoassist_fine_ci(shifts=300, boot=2000):
    """Ratio of onsets in fine-motor seconds, observed over circular-shift chance, with a 95% interval
    from resampling videos (the same kind of interval as the Ego-EXTRA and WTaG samples)."""
    cache = Path("reports/holoassist_fine_ci.json")
    if cache.exists():
        return R(cache)
    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    gid = group_ids(top)
    names = ["instructor starts, routine", "instructor starts, correction", "performer asks a question"]
    rows = {n: [] for n in names}                          # per video: (onsets in fine, chance onsets in fine)
    rng = np.random.default_rng(0)
    for v in data:
        V = build_video(v, top)
        if V is None:
            continue
        lo, hi = V["lo"], V["hi"]
        on = {n: [] for n in names}
        for e in v["events"]:
            if e["label"] == "Conversation" and lo <= e["start"] < hi:
                t = event_type(e["attributes"].get("Conversation Purpose", ""))
                if t in on:
                    on[t].append(e["start"])
        off = rng.uniform(0, hi - lo, size=shifts)
        for n in names:
            if not on[n]:
                continue
            t = np.array(on[n])
            k = np.clip(np.searchsorted(V["edges"], t, side="right") - 1, 0, len(V["state"]) - 1)
            ts = lo + (t[None, :] - lo + off[:, None]) % (hi - lo)
            ks = np.clip(np.searchsorted(V["edges"], ts, side="right") - 1, 0, len(V["state"]) - 1)
            rows[n].append(((gid[V["state"][k]] == 1).sum(), (gid[V["state"][ks]] == 1).sum(1).mean()))
    out = {"shifts": shifts, "boot": boot}
    for n, r in rows.items():
        r = np.array(r, float)
        pt = r[:, 0].sum() / r[:, 1].sum()
        b = [(lambda s: s[:, 0].sum() / s[:, 1].sum())(r[rng.integers(0, len(r), len(r))]) for _ in range(boot)]
        out[n] = {"videos": len(r), "onsets": int(r[:, 0].sum()),
                  "ratio": [round(pt, 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]}
    cache.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- figure 1
def fig1_timeline():
    """Pick the 60 s window with the most routine onsets around at least three fine-motor actions."""
    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    gid = group_ids(top)
    best = None
    W = 60.0
    for v in data:
        V = build_video(v, top)
        if V is None:
            continue
        on = np.array(sorted(e["start"] for e in v["events"] if e["label"] == "Conversation"
                             and event_type(e["attributes"].get("Conversation Purpose", "")) == "instructor starts, routine"))
        g = gid[V["state"]]
        for a in np.arange(V["lo"], V["hi"] - W, 5.0):
            m = (V["edges"][:-1] >= a) & (V["edges"][1:] <= a + W)
            fine_s = (np.diff(V["edges"])[m & (g == 1)]).sum()
            n_fine = int(((g == 1) & m).sum())
            o = on[(on >= a) & (on < a + W)]
            if n_fine < 3 or len(o) < 5 or not (0.3 * W <= fine_s <= 0.55 * W):
                continue
            k = np.clip(np.searchsorted(V["edges"], o, side="right") - 1, 0, len(g) - 1)
            score = len(o) - 3 * int((g[k] == 1).sum())     # many onsets, few of them inside fine work
            if best is None or score > best[0]:
                best = (score, v["video_name"], a, V, on, g)
    _, name, a, V, on, g = best
    colors = {0: "#f2f1ee", 1: ORANGE, 2: "#c9c8c2", 3: "#dddcd7"}   # idle, fine, hold/place/inspect, other
    fig, ax = plt.subplots(figsize=(7.0, 1.55))
    for k in range(len(g)):
        s, e = V["edges"][k], V["edges"][k + 1]
        if e <= a or s >= a + W:
            continue
        s, e = max(s, a), min(e, a + W)
        ax.add_patch(plt.Rectangle((s - a, 0), e - s, 1, color=colors[g[k]], lw=0))
        if g[k] == 1 and e - s > 2.6:
            ax.text((s + e) / 2 - a, 0.5, "fine", ha="center", va="center", color="white", fontsize=7, weight="bold")
    o = on[(on >= a) & (on < a + W)] - a
    ax.vlines(o, 1.12, 1.62, color=BLUE, lw=1.6)
    ax.plot(o, np.full(len(o), 1.62), "v", color=BLUE, ms=4)
    ax.set_xlim(0, W); ax.set_ylim(-0.05, 1.85)
    ax.set_yticks([0.5, 1.37]); ax.set_yticklabels(["hand state", "instructor\nspeaks"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("seconds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in (1, 2, 3, 0)]
    ax.legend(handles, ["screw, unscrew, insert", "hold, place, inspect", "other action", "idle"],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=7, handlelength=1.2)
    save(fig, "fig1_timeline")
    return name, round(float(a), 1)


# --------------------------------------------------------------------------- figure 2
def fig2_forest(holo):
    eg = R("reports/egoextra_label_check.json")["sets"]["furniture assembly"]["ratio"]
    wt = R("reports/wtag_label_check.json")["ratio"]
    rows = [  # (group, label, [r, lo, hi], speaker, filled)
        ("HoloAssist\n(labels, all videos)", "instructor, routine", holo["instructor starts, routine"]["ratio"], "i", True),
        ("", "instructor, correction", holo["instructor starts, correction"]["ratio"], "i", False),
        ("", "performer asks", holo["performer asks a question"]["ratio"], "p", True),
        ("Ego-EXTRA, furniture\n(blind sample)", "expert, unasked", eg["expert: unasked long turn | S (screw, unscrew, insert)"]["ci95"], "i", True),
        ("", "trainee turn", eg["trainee: turn | S (screw, unscrew, insert)"]["ci95"], "p", True),
        ("WTaG, cooking\n(blind sample)*", "instructor, routine", wt["instructor: routine instruction | S or B (any precise work)"], "i", True),
        ("", "user asks", wt["user: question | S or B (any precise work)"], "p", True),
    ]
    fix = lambda r: r if isinstance(r, list) else [r["ratio"], *r["ci95"]]
    eg_pt = {3: eg["expert: unasked long turn | S (screw, unscrew, insert)"]["ratio"],
             4: eg["trainee: turn | S (screw, unscrew, insert)"]["ratio"]}
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    y = 0.0
    ys, labels, gy = [], [], []
    for i, (grp, lab, r, who, filled) in enumerate(rows):
        if grp and i:
            y -= 0.6
        if grp:
            gy.append((y, grp))
        r = [eg_pt[i], *r] if i in eg_pt else fix(r)
        c = BLUE if who == "i" else ORANGE
        ax.plot([r[1], r[2]], [y, y], color=c, lw=1.4, solid_capstyle="round")
        ax.plot(r[0], y, "o" if who == "i" else "s", ms=5.5, mfc=c if filled else "white", mec=c, mew=1.4)
        ys.append(y); labels.append(lab)
        y -= 1
    chance_line(ax)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.set_xlim(0.2, 2.2)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for yy, g in gy:
        ax.text(2.35, yy, g, fontsize=6.5, color=INK2, va="top", ha="left")
    ax.set_xlabel("speech onsets in precise hand work,\nobserved / chance (log scale)")
    ax.text(1.0, ys[-1] - 0.9, "chance", color=CHANCE, fontsize=6.5, ha="center")
    ax.set_ylim(ys[-1] - 1.1, 0.6)
    save(fig, "fig2_across_datasets")


# --------------------------------------------------------------------------- figure 3
def fig3_hold_and_cost():
    idle = R("reports/holoassist_idle_spell.json")
    cost = R("reports/holoassist_speech_cost.json")["variants"]["main"]["contexts"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    bins = ["0 to 0.5 s", "0.5 to 1 s", "1 to 2 s", "2 to 4 s"]
    x = np.array([0.25, 0.75, 1.5, 3.0])
    for key, c, m, dx, lab in (("fine-motor action", ORANGE, "o", 0.06, "after screw, unscrew, insert"),
                               ("other action", BLUE, "s", -0.06, "after any other action")):
        r = np.array([idle[key][n]["odds_ratio_vs_4s"] for n in bins])
        a.plot(x + dx, r[:, 0], "-", color=c, lw=1.4)
        a.vlines(x + dx, r[:, 1], r[:, 2], color=c, lw=1.0, alpha=0.8)
        a.plot(x + dx, r[:, 0], m, color=c, ms=5, mec="white", mew=0.8, label=lab)
    chance_line(a, horizontal=True)
    a.set_xticks(x); a.set_xticklabels(["<0.5", "0.5-1", "1-2", "2-4"])
    a.set_xlabel("hands idle for (s)")
    a.set_ylabel("odds that the pending\ninstruction starts (vs idle 4 s+)")
    a.set_yscale("log"); a.set_yticks([0.5, 1, 2, 4]); a.set_yticklabels(["0.5", "1", "2", "4"]); a.set_ylim(0.4, 7)
    a.legend(loc="lower right", fontsize=6.5)
    a.set_title("(a) held, then released", loc="left")

    outs = [("end_soon", "hands stop\nwithin 1.5 s"), ("resumed", "action taken\nup again"), ("mistake", "wrong action\nwithin 15 s")]
    xx = np.arange(3)
    for ctx, c, m, dx, lab in (("fine", ORANGE, "o", 0.12, "speech during fine-motor work"),
                               ("other action", BLUE, "s", -0.12, "speech during other actions")):
        r = np.array([cost[ctx][o]["ratio"] for o, _ in outs])
        b.vlines(xx + dx, r[:, 1], r[:, 2], color=c, lw=1.4)
        b.plot(xx + dx, r[:, 0], m, color=c, ms=5.5, mec="white", mew=0.8, label=lab)
    chance_line(b, horizontal=True)
    b.set_xticks(xx); b.set_xticklabels([l for _, l in outs])
    b.set_xlim(-0.5, 2.5)
    b.set_ylabel("observed / chance")
    b.legend(loc="upper right", fontsize=6.5)
    b.set_title("(b) what follows the speech", loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_hold_and_cost")


# --------------------------------------------------------------------------- figure 4
def fig4_proassist():
    t = R("reports/proassist_baseline.json")["test"]
    states = ["idle", "fine", "hold/place/inspect", "other action"]
    series = [("real instructor, routine", BLUE, "o", "human instructor"),
              ("ProAssist speaks", ORANGE, "s", "ProAssist"),
              ("ProAssist reference", AQUA, "D", "ProAssist training dialogue")]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(4)
    for i, (k, c, m, lab) in enumerate(series):
        r = np.array([t[k][s] for s in states])
        dx = (i - 1) * 0.2
        ax.vlines(x + dx, r[:, 1], r[:, 2], color=c, lw=1.4)
        ax.plot(x + dx, r[:, 0], m, color=c, ms=5, mec="white", mew=0.8, label=lab)
    chance_line(ax, horizontal=True)
    ax.set_xticks(x); ax.set_xticklabels(["idle", "screw,\nunscrew,\ninsert", "hold,\nplace,\ninspect", "other\naction"])
    ax.set_ylabel("speech onsets, observed / chance")
    ax.set_ylim(0.3, 1.9)
    ax.legend(loc="upper center", fontsize=6.5, ncol=1, bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig4_proassist")


# --------------------------------------------------------------------------- figure 5
def fig5_rating():
    p = Path("reports/rating_study2.json")
    d = R(p) if p.exists() else R("reports/rating_study.json")
    tag = "second round" if p.exists() else "pilot"
    cells = d["all"]["cells"] if "all" in d else d["cells"]
    diffs = d["all"] if "all" in d else d
    n = cells["routine, now"]["n"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1, 1]})
    for i, (ver, c, m, dx) in enumerate((("now", BLUE, "o", -0.08), ("held", ORANGE, "s", 0.08))):
        r = np.array([[cells[f"{typ}, {ver}"][k] for k in ("mean", "low", "high")] for typ in ("routine", "warning")])
        a.vlines(np.arange(2) + dx, r[:, 1], r[:, 2], color=c, lw=1.4)
        a.plot(np.arange(2) + dx, r[:, 0], m, color=c, ms=5.5, mec="white", mew=0.8, label=f"spoken {ver}")
    a.set_xticks([0, 1]); a.set_xticklabels(["routine message", "warning"]); a.set_xlim(-0.5, 1.5)
    a.set_ylim(1, 5); a.set_ylabel("\"spoke at a good moment\" (1 to 5)")
    a.legend(loc="lower left", fontsize=6.5)
    a.set_title(f"(a) mean rating ({tag}, {n} raters)", loc="left")
    keys = [("routine: held - now", "routine"), ("warning: held - now", "warning")]
    for i, (k, lab) in enumerate(keys):
        r = diffs[k]
        b.plot([r["low"], r["high"]], [-i, -i], color=INK2, lw=1.6, solid_capstyle="round")
        b.plot(r["mean"], -i, "o", color=INK, ms=5.5)
    b.axvline(0, color=CHANCE, lw=0.8, ls=(0, (3, 2)))
    b.set_yticks([0, -1]); b.set_yticklabels([l for _, l in keys]); b.set_ylim(-1.6, 0.6)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.set_xlim(-1.5, 0.75)
    b.set_xlabel("held minus now (scale points)")
    b.text(0.03, 0.45, "held rated higher →", fontsize=6.5, color=INK2)
    b.text(-0.03, 0.45, "← held rated lower", fontsize=6.5, color=INK2, ha="right")
    b.set_title("(b) the effect of holding the message", loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_rating")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()
    want = set(args.only or [1, 2, 3, 4, 5])
    meta = {}
    if 1 in want:
        meta["fig1"] = fig1_timeline()
    if 2 in want:
        fig2_forest(holoassist_fine_ci())
    if 3 in want:
        fig3_hold_and_cost()
    if 4 in want:
        fig4_proassist()
    if 5 in want:
        fig5_rating()
    if meta:
        print(meta)


if __name__ == "__main__":
    main()
