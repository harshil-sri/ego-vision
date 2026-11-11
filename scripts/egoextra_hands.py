"""Ego-EXTRA: do experts time unasked speech around the trainee's HAND MOTION?

Second-dataset check of the HoloAssist pilot (part 1), with two changes
that answer its main weaknesses:

  * the hand signal is MEASURED (Aria glasses hand tracking), not a human
    action label, so label-boundary habits cannot create the effect;
  * the expert is a real remote expert who sees only the trainee's camera.

Inputs (scripts.egoextra_speech describes the transcripts)
    <root>/ht/hand_tracking/<video>_wrist_and_palm_poses_trimmed.csv   or
    <root>/ht/hand_tracking/<video>_hand_tracking_results_trimmed.csv
        one row per tracking sample (10 to 30 per second): time stamp in
        microseconds, per hand a confidence (-1 = hand not tracked) and 3-D
        positions in metres in the frame of the glasses.

Assumption: the first row of a "trimmed" file is time 0 of the transcript.
The files do not say so. Support: for most videos the tracking duration
equals the transcript length within a few seconds. Videos where they differ
by more than --max-mismatch seconds are left out.

Signal, on a 0.1 s grid
    position   wrist (first file type) or mean of the 21 landmarks (second)
    speed      distance moved per second, per hand; the faster hand counts
    recent     mean speed over the last 1 s (only bins with a tracked hand)
    state      none    no hand tracked in more than half of the last 1 s
               still   recent speed in the lowest third of this video
               mid     middle third
               moving  highest third
The frame moves with the head, so "speed" contains head motion too.

--signal fingers replaces the wrist speed by FINGER motion: the 210
distances between the 21 landmarks of a hand describe the hand's shape and
do not change when the head or the whole hand moves. Signal = mean change
of these distances per second (metres per second). It is high when the
fingers work (grasping, turning a screw) even if the wrist is still. Only
videos with the landmark file can be used.

--signal gaze and --signal gaze_depth use the trainee's EYES instead
(<root>/gz/gaze/<video>_general_eye_gaze_trimmed.csv, 10 samples per
second: yaw and pitch of the gaze in radians, depth of the gaze point in
metres). gaze: angular speed of the gaze (radians per second); "moving" =
the eyes jump around, "still" = they rest on one place. gaze_depth: the
states are then near / mid / far thirds of the gaze distance (named still
/ mid / moving in the tables for the same layout). "none" = no gaze sample.

Test (same as HoloAssist part 1): share of speech onsets that fall in each
state, against the share expected by chance. Chance = shift all onsets of
a video by one random offset (circular), --shifts times. Ratio 1.0 = no
relation. The 95% range of the shifted ratios is given.

Usage:
    python -m scripts.egoextra_hands
Outputs: reports/egoextra_hands.json, reports/egoextra_hands.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.egoextra_speech import expert_turns, load_transcript
from src.paths import resolve_data_path

ROOT = "/mnt/ssd/ego-extra"
STATES = ["none", "still", "mid", "moving"]
DT = 0.1


def hand_states(path: Path, signal: str = "wrist") -> tuple[np.ndarray, np.ndarray] | None:
    """-> (state id per 0.1 s bin, recent speed per bin in m/s; NaN where no hand), or None."""
    df = pd.read_csv(path)
    if len(df) < 100:
        return None
    t = (df.iloc[:, 0].to_numpy() - df.iloc[0, 0]) / 1e6                # seconds from the first row
    n = int(t[-1] / DT) + 1
    b = np.minimum((t / DT).astype(int), n - 1)                         # grid bin of every sample
    speeds = []
    for side in ("left", "right"):
        ok = df[f"{side}_tracking_confidence"].to_numpy() > 0
        wrist = [f"t{a}_{side}_wrist_device" for a in "xyz"]
        if wrist[0] in df.columns:
            pos = df[wrist].to_numpy()                                  # (rows, 3)
        else:
            pos = np.stack([df[[c for c in df.columns if c.startswith(f"t{a}_{side}_landmark_")]].to_numpy().mean(1)
                            for a in "xyz"], 1)
        cnt = np.bincount(b[ok], minlength=n)
        have = cnt > 0
        both = have[1:] & have[:-1]
        sp = np.full(n, np.nan)
        if signal == "fingers":
            lm = np.stack([df[[f"t{a}_{side}_landmark_{j}_device" for a in "xyz"]].to_numpy()
                           for j in range(21)], 1)                      # (rows, 21, 3)
            g = np.zeros((n, 21, 3))
            np.add.at(g, b[ok], lm[ok])
            g /= np.maximum(cnt, 1)[:, None, None]                      # mean landmark positions per bin
            iu = np.triu_indices(21, 1)
            dist = np.linalg.norm(g[:, :, None] - g[:, None], axis=-1)[:, iu[0], iu[1]]   # (n, 210) hand shape
            sp[1:][both] = np.abs(dist[1:] - dist[:-1]).mean(1)[both] / DT
        else:
            grid = np.stack([np.bincount(b[ok], weights=pos[ok, k], minlength=n) for k in range(3)], 1)
            grid = grid / np.maximum(cnt, 1)[:, None]                   # (n, 3) mean position per bin
            sp[1:][both] = np.linalg.norm(grid[1:] - grid[:-1], axis=1)[both] / DT
        speeds.append(sp)
    return to_states(np.fmax(speeds[0], speeds[1]))                     # faster hand; NaN only if both missing


def gaze_signal(path: Path, kind: str) -> np.ndarray | None:
    """-> per 0.1 s bin: gaze angular speed (rad/s) or gaze depth (m); NaN where no sample."""
    df = pd.read_csv(path, usecols=["tracking_timestamp_us", "left_yaw_rads_cpf", "right_yaw_rads_cpf",
                                    "pitch_rads_cpf", "depth_m"])
    if len(df) < 100:
        return None
    t = (df.iloc[:, 0].to_numpy() - df.iloc[0, 0]) / 1e6
    n = int(t[-1] / DT) + 1
    b = np.minimum((t / DT).astype(int), n - 1)
    yaw = (df["left_yaw_rads_cpf"].to_numpy() + df["right_yaw_rads_cpf"].to_numpy()) / 2
    cols = [yaw, df["pitch_rads_cpf"].to_numpy(), df["depth_m"].to_numpy()]
    ok = ~np.isnan(yaw)
    cnt = np.bincount(b[ok], minlength=n)
    g = np.stack([np.bincount(b[ok], weights=c[ok], minlength=n) for c in cols], 1) / np.maximum(cnt, 1)[:, None]
    have = cnt > 0
    out = np.full(n, np.nan)
    if kind == "gaze_depth":
        out[have] = g[have, 2]
    else:
        both = have[1:] & have[:-1]
        out[1:][both] = np.linalg.norm(g[1:, :2] - g[:-1, :2], axis=1)[both] / DT
    return out


def to_states(speed: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-bin signal (NaN = missing) -> (state id per bin, mean of the last 1 s)."""
    n = len(speed)
    s = pd.Series(speed)
    recent = s.rolling(10, min_periods=1).mean().to_numpy().copy()      # last 1 s
    seen = s.notna().rolling(10, min_periods=1).mean().to_numpy()
    state = np.zeros(n, dtype=int)                                      # 0 = none
    vis = (seen >= 0.5) & ~np.isnan(recent)
    if vis.sum() < 100:
        return None
    lo, hi = np.percentile(recent[vis], [100 / 3, 200 / 3])
    state[vis] = 1 + (recent[vis] > lo) + (recent[vis] > hi)
    recent[~vis] = np.nan
    return state, recent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--max-mismatch", type=float, default=30.0)
    ap.add_argument("--turn-gap", type=float, default=2.0)
    ap.add_argument("--ask-window", type=float, default=5.0)
    ap.add_argument("--long", type=float, default=1.5)
    ap.add_argument("--signal", choices=["wrist", "fingers", "gaze", "gaze_depth"], default="wrist")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    root = Path(resolve_data_path(ROOT))
    proto = {v["ID"]: v["Protocol"] for v in json.loads((root / "protocol_split_wacv.json").read_text())["videos"]}
    videos, skipped = [], {"no hand file": 0, "duration mismatch": 0, "too little tracking": 0}
    for tp in sorted((root / "transcript").rglob("*_transcript.txt")):
        vid = tp.name.replace("_transcript.txt", "")
        u = load_transcript(tp)
        if vid not in proto or len(u) < 5:
            continue
        best = None
        gaze = args.signal.startswith("gaze")
        folder = root / "gz" / "gaze" if gaze else root / "ht" / "hand_tracking"
        pattern = (f"{vid}_general_eye_gaze_*.csv" if gaze else f"{vid}_*.csv" if args.signal == "wrist"
                   else f"{vid}_hand_tracking_results_*.csv")
        for hp in sorted(folder.glob(pattern)):
            ts = pd.read_csv(hp, usecols=[0]).iloc[:, 0].to_numpy()
            if len(ts) > 100:
                d = abs((ts[-1] - ts[0]) / 1e6 - u[:, 1].max())
                if best is None or d < best[0]:
                    best = (d, hp)
        if best is None:
            skipped["no hand file"] += 1
            continue
        if best[0] > args.max_mismatch:
            skipped["duration mismatch"] += 1
            continue
        if gaze:
            sig = gaze_signal(best[1], args.signal)
            hs = to_states(sig) if sig is not None else None
        else:
            hs = hand_states(best[1], args.signal)
        if hs is None:
            skipped["too little tracking"] += 1
            continue
        state, recent = hs
        starts, unasked, speech = expert_turns(u, args.turn_gap, args.ask_window)
        long_ = speech >= args.long
        a = u[u[:, 2] == 0]
        tr = a[np.r_[True, a[1:, 0] - np.maximum.accumulate(a[:-1, 1]) >= args.turn_gap], 0]   # trainee turn starts
        ev = {"expert, unasked, long": starts[unasked & long_],
              "expert, unasked, short": starts[unasked & ~long_],
              "expert, after trainee speech, long": starts[~unasked & long_],
              "trainee turn": tr}
        ev = {k: np.minimum((t / DT).astype(int), len(state) - 1) for k, t in ev.items()}
        videos.append({"vid": vid, "protocol": proto[vid], "state": state, "recent": recent, "ev": ev})
    print(f"{len(videos)} videos used, skipped {skipped}", flush=True)

    def ratios(vs, key):
        """-> counts (4,), observed/expected ratio (4,), null 2.5% and 97.5% of the ratio (4,) each."""
        obs = np.zeros(4)
        null = np.zeros((args.shifts, 4))
        for v in vs:
            e, n = v["ev"][key], len(v["state"])
            if len(e) == 0:
                continue
            obs += np.bincount(v["state"][e], minlength=4)
            off = rng.integers(0, n, size=args.shifts)
            st = v["state"][(e[None, :] + off[:, None]) % n]            # (shifts, events)
            for k in range(4):
                null[:, k] += (st == k).sum(1)
        exp = null.mean(0)
        return obs, obs / np.maximum(exp, 1e-9), np.percentile(null, 2.5, 0) / exp, np.percentile(null, 97.5, 0) / exp

    def speed_profile(vs, key, offsets=(-4, -2, -1, 0, 1, 2, 4)):
        """Recent hand speed at onset + offset s, divided by the video's mean recent speed. Mean over onsets."""
        out = {}
        for o in offsets:
            vals = []
            for v in vs:
                r, e = v["recent"], v["ev"][key] + int(round(o / DT))
                e = e[(e >= 0) & (e < len(r))]
                vals.append(r[e] / np.nanmean(r))
            x = np.concatenate(vals)
            out[str(o)] = round(float(np.nanmean(x)), 3)
        return out

    keys = list(videos[0]["ev"])
    groups = {"all videos": videos,
              "protocol 1 (proactive)": [v for v in videos if v["protocol"] == 1],
              "protocol 2 (only when asked or necessary)": [v for v in videos if v["protocol"] == 2]}
    res = {"videos": len(videos), "skipped": skipped, "shifts": args.shifts,
           "hours": round(sum(len(v["state"]) for v in videos) * DT / 3600, 1),
           "state_share_of_time": {s: round(float(np.mean(np.concatenate([v["state"] for v in videos]) == i)), 3)
                                   for i, s in enumerate(STATES)},
           "groups": {}}
    for g, vs in groups.items():
        res["groups"][g] = {"videos": len(vs)}
        for k in keys:
            obs, r, lo, hi = ratios(vs, k)
            res["groups"][g][k] = {"n": int(obs.sum()),
                                   "ratio": {s: round(float(r[i]), 3) for i, s in enumerate(STATES)},
                                   "null95": {s: [round(float(lo[i]), 3), round(float(hi[i]), 3)] for i, s in enumerate(STATES)},
                                   "speed_profile": speed_profile(vs, k)}
    stem = "reports/egoextra_hands" + ("" if args.signal == "wrist" else "_" + args.signal)
    Path(stem + ".json").write_text(json.dumps(res, indent=2))

    L = ["# Ego-EXTRA: speech onsets against measured hand motion", "",
         f"Generated by `python -m scripts.egoextra_hands --signal {args.signal}`. "
         f"{res['videos']} videos, {res['hours']} h (left out: {skipped}). Hand state from the glasses' hand tracking "
         + {"wrist": "(wrist speed), ", "fingers": "(finger motion: change of hand shape), ",
            "gaze": "(NOT hands: gaze angular speed; still = eyes rest, moving = eyes jump), ",
            "gaze_depth": "(NOT hands: gaze distance; still = near third, moving = far third), "}[args.signal]
         +
         "see the script docstring. Share of time per state: "
         + ", ".join(f"{s} {v:.0%}" for s, v in res["state_share_of_time"].items()) + ".", "",
         "Each cell: onsets in this state, observed / expected by chance. "
         f"In brackets: the range that holds 95% of {args.shifts} random shifts (a ratio outside it is unlikely to be chance).", ""]
    for g, r in res["groups"].items():
        L += [f"## {g} ({r['videos']} videos)", "", "| onsets | n | " + " | ".join(STATES) + " |", "|---|---|" + "---|" * 4]
        for k in keys:
            c = r[k]
            L.append(f"| {k} | {c['n']} | " + " | ".join(
                f"{c['ratio'][s]:.2f} [{c['null95'][s][0]:.2f}, {c['null95'][s][1]:.2f}]" for s in STATES) + " |")
        L.append("")
    L += ["## Signal around the onset (all videos)", "",
          "Mean of the signal over the last 1 s, as a multiple of the video's average, at seconds relative to the onset.", "",
          "| onsets | " + " | ".join(f"{o} s" for o in res["groups"]["all videos"][keys[0]]["speed_profile"]) + " |",
          "|---|" + "---|" * 7]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in res["groups"]["all videos"][k]["speed_profile"].values()) + " |")
    Path(stem + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
