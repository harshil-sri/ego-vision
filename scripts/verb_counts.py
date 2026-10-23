"""Count verb instances in the EPIC-KITCHENS-100 annotation CSVs.

Writes a markdown report (train and validation counts per verb) with the
8 candidate demo verbs highlighted and any candidate under 100 training
instances flagged. Candidate verbs are a configuration, not a decision
made here; see DECISIONS_NEEDED.md.

Usage:
    python -m scripts.verb_counts \
        --annotations /mnt/ssd/epic-tools/epic-kitchens-100-annotations \
        --out reports/verb_counts.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CANDIDATE_VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]
MANY_SHOT_THRESHOLD = 100  # owner's rule: many-shot means >100 train instances


def count_verbs(csv_path: Path) -> pd.Series:
    """Return verb -> instance count for one annotation CSV.

    Input: CSV with a 'verb' column (string), one row per action segment.
    Output: pandas Series indexed by verb name, values are int counts.
    """
    df = pd.read_csv(csv_path, usecols=["verb"])
    return df["verb"].value_counts()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="path to epic-kitchens-100-annotations clone")
    parser.add_argument("--out", type=Path, default=Path("reports/verb_counts.md"))
    args = parser.parse_args()

    train_counts = count_verbs(args.annotations / "EPIC_100_train.csv")
    val_counts = count_verbs(args.annotations / "EPIC_100_validation.csv")

    lines = [
        "# Verb counts in EPIC-KITCHENS-100",
        "",
        f"Source: `{args.annotations}` (official CSVs).",
        "Every number below comes from `value_counts()` on the `verb` column;",
        "nothing is estimated.",
        "",
        f"Candidate demo verbs (from the build brief): {', '.join(CANDIDATE_VERBS)}",
        "",
        "| verb | train | validation | candidate | many-shot (>100 train) |",
        "|---|---:|---:|---|---|",
    ]
    all_verbs = sorted(set(train_counts.index) | set(val_counts.index))
    for verb in all_verbs:
        tr = int(train_counts.get(verb, 0))
        va = int(val_counts.get(verb, 0))
        cand = "**yes**" if verb in CANDIDATE_VERBS else ""
        shot = "yes" if tr > MANY_SHOT_THRESHOLD else "**NO**"
        lines.append(f"| {verb} | {tr} | {va} | {cand} | {shot} |")

    lines += ["", "## Candidate verbs summary", "",
              "| verb | train | validation | many-shot (>100 train) |",
              "|---|---:|---:|---|"]
    for verb in CANDIDATE_VERBS:
        tr = int(train_counts.get(verb, 0))
        va = int(val_counts.get(verb, 0))
        flag = "yes" if tr > MANY_SHOT_THRESHOLD else "**NO — under 100**"
        lines.append(f"| {verb} | {tr} | {va} | {flag} |")

    lines += ["", f"Total distinct verbs: train {len(train_counts)}, "
                  f"validation {len(val_counts)}.",
              f"Total train segments: {int(train_counts.sum())}, "
              f"validation {int(val_counts.sum())}."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    # Console echo of the decision-relevant part only.
    print("\ncandidate verbs (train / validation):")
    for verb in CANDIDATE_VERBS:
        print(f"  {verb:8s} {int(train_counts.get(verb, 0)):6d} / "
              f"{int(val_counts.get(verb, 0)):5d}")


if __name__ == "__main__":
    main()
