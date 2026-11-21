"""Build the public label release in release/ (see release/README.md for what each file holds).

    python -m scripts.make_release                 # after the human relabelling (needs reports/*_hand_labels_human.csv)
    python -m scripts.make_release --preliminary   # before it: AI labels only, marked as preliminary

The release holds labels and times only. It never holds frames, video, audio, transcripts, or the
names of labellers or raters.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.holoassist_need_controlled import FINE, FREEVERB
from scripts.holoassist_timing import LABELS
from src.paths import resolve_data_path

OUT = Path("release")
KIND = {"egoextra": {"expert_unasked": "expert speaks, unasked", "trainee_turn": "trainee speaks", "random": "random second"},
        "wtag": {"onset": "instructor starts a routine instruction", "user_question": "user asks a question",
                 "random": "random second"}}


def labels(ds: str, preliminary: bool) -> pd.DataFrame:
    human = Path(f"reports/{ds}_hand_labels_human.csv")
    if human.exists() and not preliminary:
        d = pd.read_csv(human, dtype={"recording": str})
        d = d.rename(columns={"ai": "label_ai_preliminary", "h1": "label_person_1", "h2": "label_person_2",
                              "agreed": "label_agreed"})
        cols = ["recording", "t_s", "kind", "label_person_1", "label_person_2", "label_agreed", "label_ai_preliminary"]
    else:
        if not preliminary:
            raise SystemExit(f"{human} is missing: run scripts.relabel_analyze first, or pass --preliminary")
        d = pd.read_csv(f"reports/{ds}_hand_labels.csv", dtype={"recording": str}).rename(
            columns={"label": "label_ai_preliminary"})
        cols = ["recording", "t_s", "kind", "label_ai_preliminary"]
    d = d.sort_values(["recording", "t_s"]).reset_index(drop=True)
    d["kind"] = d["kind"].map(KIND[ds]).fillna(d["kind"])
    d.insert(0, "dataset", {"egoextra": "Ego-EXTRA", "wtag": "WTaG"}[ds])
    return d[["dataset"] + cols]


def holoassist_groups() -> dict:
    data = json.loads(Path(resolve_data_path(LABELS)).read_text(encoding="utf-8"))
    counts = Counter(e["attributes"].get("Verb") for v in data for e in v["events"] if e["label"] == "Fine grained action")
    top = [v for v, _ in counts.most_common(20)]
    group = lambda v: "fine (screw, unscrew, insert)" if v in FINE else (
        "hold, place, inspect" if v in FREEVERB else "other action")
    return {"rule": "a second with no labelled action is 'idle'; otherwise the verb of the action decides. "
                    "Verbs outside the 20 most common count as 'other action'.",
            "verbs": {v: {"group": group(v), "actions": counts[v]} for v in top}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preliminary", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for ds in ("egoextra", "wtag"):
        d = labels(ds, args.preliminary)
        name = f"{ds}_hand_labels{'_preliminary' if args.preliminary else ''}.csv"
        d.to_csv(OUT / name, index=False)
        print(f"wrote {OUT / name}: {len(d)} moments")
    (OUT / "holoassist_hand_state_groups.json").write_text(json.dumps(holoassist_groups(), indent=2))
    print(f"wrote {OUT / 'holoassist_hand_state_groups.json'}")


if __name__ == "__main__":
    main()
