# Pre-registration of the rating study

Written on 2026-11-12, before any rater took part. The date of the git
commit that adds this file is the evidence.

## Hypotheses

Rating: "The assistant spoke at a good moment", 1 (very bad moment) to
5 (very good moment).

- **H1 (routine).** For routine messages, the "held" version is rated
  higher than the "now" version.
- **H2 (warning).** For warnings, the "held" version is not rated higher
  than the "now" version. We expect it to be rated lower.
- **H3 (interaction).** The difference "held minus now" is larger for
  routine messages than for warnings.

H3 is the main test. It is the claim of the project: a message should
wait for the hands only if it is not urgent.

## Design

- 20 clips: 12 routine, 8 warning. One practice clip, not analysed.
- Every rater sees each clip once. Inside each message type, half of the
  clips are "now" and half are "held", at random for each rater. The
  order of the clips is random for each rater.
- Planned number of raters: 10 to 20. Data collection stops when no new
  rater has answered for 7 days or at 25 raters, whichever comes first.
  The data are not analysed before collection stops.

## Analysis (`scripts/rating_study_analyze.py`, written before the data)

- Unit: the rater. For each rater, the mean rating in each of the four
  cells; then the three differences above.
- Result: the mean of each difference over raters with a 95% interval
  from 10,000 resamples of raters. A hypothesis counts as supported when
  the interval excludes 0 in the predicted direction (H2: when the
  interval does not lie above 0).
- Left out: a rater who gave the same rating to every clip. Reported:
  how many.
- Reported next to the main result, not instead of it: the same analysis
  for the raters who heard the voice, and the mean rating of every clip.
- Nothing else is left out. No clip is removed after the data are seen.

## What would count against the project

- H1 fails: observers do not prefer a routine message after the
  fine-motor work.
- H3 fails with H1 true: observers prefer waiting for every message, so
  urgency does not matter to them.

## Amendment, 2026-11-13 (written before any rating was read)

- Collection is closed by Harshil on 2026-11-13 with 14 raters, one day
  after the invitation. This departs from the stopping rule above ("no
  new rater for 7 days or 25 raters"). The number is inside the planned
  10 to 20. A code that arrives after this date is not used.
- Nothing was analysed before this amendment. The codes were only
  checked for format: 13 decode as result codes with 20 answers each;
  one rater pasted the result as plain JSON instead of the code, and it
  was moved unchanged to `study/responses/xz294u3b.json`, a format the
  analysis script already reads.
- The git commit that adds this amendment and the responses comes
  before the commit that adds the results.
