# Human relabelling: plan, fixed before any label was collected

Written on 2026-11-21. The git commit that adds this file, the pages and
`scripts/relabel_analyze.py` comes before any human label.

## Why

All 420 Ego-EXTRA and 417 WTaG moments in the paper were labelled S, B
or F by one labeller, the AI assistant that wrote the code. A first
human check of 60 furniture moments showed only fair agreement. Two
people now label every moment, independently, and the paper's numbers
are recomputed from their labels. The AI labels become a preliminary
set; the paper reports how well they agree with the humans.

## Protocol

- Moments: all 420 (Ego-EXTRA) and all 417 (WTaG), the same moments as
  in the paper. Each dataset is split at random (seed 2026) into 6
  batches of about 70.
- View: the same 4 frames the first labeller saw (the 2 s that end at the
  moment, 0.5 s apart).
- Blind: the page shows no first label, no kind of moment (speech onset
  or random second), no recording name and no original id. The moments
  of a batch appear in random order.
- Training: a guide with the three label definitions and 2 picture
  examples per label, then 6 practice moments with feedback. The guide
  and practice pictures are other seconds of the same recordings, at
  least 10 s from every moment of the sample (`study/label/examples.json`,
  chosen by looking at 120 random candidates, `scripts/relabel_examples.py`).
- Labellers: volunteers (friends of the author), who do not know the
  hypothesis. Each batch is labelled by two different people who work
  alone. One person may label several batches.

## Analysis (`scripts/relabel_analyze.py`)

- Per batch, the first two complete results from two different people
  are labeller h1 and h2. Agreed label = both chose the same label.
- Agreement: Cohen's kappa with a 95% interval (5,000 resamples of
  moments) for h1 vs h2, AI vs h1, AI vs h2, on three label sets: S/B/F,
  S vs not S, S or B vs F.
- The paper's onset analysis (share of S, and of S or B, at speech
  onsets vs random seconds; ratio with a bootstrap interval) is redone
  with each label set: AI, h1, h2, agreed.
- **Primary result: the agreed labels.** h1 and h2 are reported next to
  it, because keeping only agreed moments drops the unclear ones and may
  favour clear cases. A conclusion in the paper needs the same direction
  with h1 and with h2.
- Ego-EXTRA is reported for both scenarios together and for furniture
  assembly alone (as in the paper); WTaG as a whole.
- The paper replaces its AI-label numbers with these, whatever they show.
