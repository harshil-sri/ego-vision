"""End-to-end REASONING-CHAIN evaluation on real validation videos.

Question answered: given this checkpoint's (weak) per-tick probabilities,
how far does the recipe state machine get through REAL annotated cooking
sequences — and how much does the recipe-aware prior add?

Protocol (perfect-matcher assumption: ingredient comes from the annotation,
so this isolates the verb chain: probs -> [prior] -> smoother -> edges ->
state machine):
  - each val video becomes a pseudo-recipe: its annotated (verb, noun)
    segments in temporal order = the step list;
  - the cached per-segment probability series is replayed in order; a fired
    verb edge feeds the registry state machine exactly like the live engine;
  - metrics: step advance rate (advances / steps), wrong_order + repeat
    events, completed recipes, and the index of the first wrong turn.

Usage:
    python3 -m scripts.eval_loop --cache runs/eval_cache/val_preds.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class _Step:
    verb: str
    ingredient: str
    note: str = ""


@dataclass
class _Recipe:
    name: str
    steps: list
    ingredients: list
    verbs: list


def _aligned_segments() -> pd.DataFrame:
    """Val rows in the SAME order the cache was built (deterministic sort),
    with the noun column merged back from the raw annotation CSVs."""
    import yaml
    from scripts.eval_smoothers import _load_val_segments
    seg = _load_val_segments(Path("runs/baseline"))
    ann = yaml.safe_load(Path("configs/data.yaml").read_text())["annotations_dir"]
    raw = pd.concat([pd.read_csv(f"{ann}/EPIC_100_train.csv"),
                     pd.read_csv(f"{ann}/EPIC_100_validation.csv")],
                    ignore_index=True)
    nouns = raw.set_index(["video_id", "narration_id"])["noun"]
    seg["noun"] = [nouns.loc[(v, n)] for v, n in
                   zip(seg.video_id, seg.narration_id)]
    return seg


def _run_video(machine_cls, steps, ticks, verbs, smoother_factory,
               use_prior: bool, leak: float, min_conf: float,
               oracle: bool = False, policy: str = "strict"):
    """Replay one video; return (advances, wrongs, repeats, completed, first_wrong).

    oracle=True fires the TRUE verb every tick (no model): an upper bound
    that validates the state-machine chain independently of perception."""
    from src.pipeline.prior import apply_recipe_prior, legal_verbs
    recipe = _Recipe(name="val", steps=steps,
                     ingredients=[s.ingredient for s in steps],
                     verbs=sorted({s.verb for s in steps}))
    machine = machine_cls(recipe, policy=policy)
    smoother = smoother_factory() if smoother_factory else None
    last_fired = None
    advances = wrongs = repeats = 0
    completed = False
    first_wrong = None
    for probs, (true_verb, noun) in ticks:
        if oracle:
            out, conf = true_verb, 1.0
        else:
            if use_prior:
                legal = legal_verbs(steps, machine.current_index)
                if legal:
                    probs = apply_recipe_prior(probs, verbs, legal, leak=leak)
            if smoother is None:
                i = int(np.argmax(probs)); conf = float(probs[i])
                out = verbs[i]
            else:
                out, conf = smoother.update(probs, 0)
            if conf < min_conf:
                continue
        if out is None or out == last_fired:
            continue
        last_fired = out
        ev = machine.update(out, noun, 0)
        if ev.kind == "advance":
            advances += 1
        elif ev.kind == "wrong_order":
            wrongs += 1
            if first_wrong is None:
                first_wrong = machine.current_index
        elif ev.kind == "repeat":
            repeats += 1
        elif ev.kind == "done":
            advances += 1
            completed = True
            break
    return advances, wrongs, repeats, completed, first_wrong


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="runs/eval_cache/val_preds.npz")
    ap.add_argument("--out", default="reports/loop_eval.md")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--leak", type=float, default=0.05)
    ap.add_argument("--policy", choices=["strict", "optimistic"], default="strict")
    ap.add_argument("--max-steps", type=int, default=15,
                    help="cap pseudo-recipes at demo-scale length")
    args = ap.parse_args()

    import src.registry as registry
    from src.pipeline.smoothers import make_smoother

    z = np.load(args.cache, allow_pickle=True)
    video = [str(v) for v in z["video"]]
    verb = [str(v) for v in z["verb"]]
    probs = z["probs"]
    verbs = [str(v) for v in z["verbs"]]

    seg = _aligned_segments()               # nouns aligned by position
    assert len(seg) == len(video), "cache/segments misaligned — rebuild cache"
    nouns = seg.noun.tolist()

    by_video: dict[str, list] = {}
    for v, lb, p, nn in zip(video, verb, probs, nouns):
        by_video.setdefault(v, []).append((p, (lb, str(nn))))

    machine_cls = registry.get_class("state_machine")
    smoothers = [
        ("raw", None),
        ("majority", lambda: make_smoother("majority", verbs, k=5,
                                           min_conf=args.min_conf)),
        ("ema", lambda: make_smoother("ema", verbs, alpha=0.3,
                                      min_conf=args.min_conf)),
    ]
    rows = []
    runs = [("oracle", None, False)] + [
        (sm_name, factory, use_prior)
        for sm_name, factory in smoothers
        for use_prior in (False, True)
    ]
    for sm_name, factory, use_prior in runs:
        tot_steps = tot_adv = tot_wrong = tot_rep = 0
        completed = 0
        first_wrongs = []
        for v, ticks_all in by_video.items():
            ticks = ticks_all[:args.max_steps]
            steps = [_Step(t[1][0], t[1][1]) for t in ticks]
            a, w, r, done, fw = _run_video(
                machine_cls, steps, ticks, verbs, factory,
                use_prior, args.leak, args.min_conf,
                oracle=(sm_name == "oracle"), policy=args.policy)
            tot_steps += len(steps); tot_adv += a
            tot_wrong += w; tot_rep += r
            completed += int(done)
            if fw is not None:
                first_wrongs.append(fw)
        rows.append({
            "smoother": sm_name, "prior": use_prior,
            "advance_rate": round(tot_adv / tot_steps, 4),
            "wrong_order": tot_wrong, "repeat": tot_rep,
            "completed": f"{completed}/{len(by_video)}",
            "median_first_wrong": (int(np.median(first_wrongs))
                                   if first_wrongs else None),
        })

    out = Path(args.out).with_name(Path(args.out).stem + ("" if args.policy == "strict" else "_" + args.policy) + Path(args.out).suffix)
    lines = [
        "# Reasoning-chain evaluation (real val videos, perfect-matcher assumption)",
        "",
        f"Cache: {len(video)} segments / {len(by_video)} videos from "
        "`runs/baseline`. Pseudo-recipes = each video's annotated (verb, noun) "
        f"sequence; leak={args.leak}, min_conf={args.min_conf}, policy={args.policy}.",
        "",
        "| smoother | recipe prior | advance rate | wrong_order | repeat | completed | median first-wrong step |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['smoother']} | {'ON' if r['prior'] else 'off'} "
            f"| {r['advance_rate']} | {r['wrong_order']} | {r['repeat']} "
            f"| {r['completed']} | {r['median_first_wrong']} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
