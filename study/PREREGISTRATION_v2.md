# Pre-registration of the rating study, second round (confirmatory)

Written on 2026-11-21, before any rater of this round took part. The date
of the git commit that adds this file, together with the analysis script
`scripts/rating_study2_analyze.py`, is the evidence.

The first round (`PREREGISTRATION.md`, 14 raters, 2026-11-12 to 13) is
now the pilot. Its results were seen before this plan was written, and
this round tests them on new raters. The pilot is not pooled into the
confirmatory tests.

## What the pilot found

Rating: "The assistant spoke at a good moment", 1 (very bad moment) to 5
(very good moment). "now" = the message during fine-motor work; "held" =
the message when the rule releases it (the video head says the hands are
free).

| difference (held minus now) | mean | 95% interval over raters | SD over raters |
|---|---|---|---|
| routine messages | -0.24 | -0.60 to +0.14 | 0.73 |
| warnings | -0.82 | -1.14 to -0.46 | 0.69 |
| interaction (routine minus warning) | +0.58 | +0.02 to +1.13 | 1.10 |

The pilot's H1 (routine: held rated higher) was not supported: the point
estimate was in the other direction. So this round does not test whether
observers prefer waiting. It tests whether they do NOT prefer it.

## Hypotheses

- **H1 (warnings).** For warnings, "held" is rated lower than "now".
  Supported when the 95% interval of (held minus now) lies below 0.
- **H2 (interaction, main test).** The difference (held minus now) is
  larger for routine messages than for warnings. Supported when the 95%
  interval of (routine held minus now) minus (warning held minus now)
  lies above 0.
- **H3 (routine, no preference for waiting).** For routine messages,
  observers do not rate "held" meaningfully higher than "now". Supported
  when the upper end of the 95% interval of (held minus now) is below
  +0.25, a quarter of a scale point (the smallest difference we consider
  meaningful). If the whole interval lies below 0, we also report that
  observers rated "now" higher for routine messages.

## Design (same as the pilot, with four changes)

- The same 20 clips (12 routine, 8 warning), the same practice clip, the
  same random assignment: every rater sees each clip once; inside each
  message type half "now" and half "held", at random per rater; random
  order.
- Changes: (1) a consent screen (18 or older, agreement to take part);
  (2) a person who says they did a study like this before is thanked
  and stopped before any rating; (3) answers are sent to a collection
  sheet, with the copy-a-code route as a backup; (4) two optional
  questions at the end (age band, field of study) and a short
  explanation of the study after the last rating.
- Page: `study/rate/` (`https://harshil-sri.github.io/ego-vision/study/rate/`).
  No name, email or phone number is asked. The answers are anonymous.

## Raters and stopping rule

- Recruited by the author (Harshil) among friends and student groups.
  People who worked on the project and the 14 pilot raters do not take
  part.
- Target: at least 40 usable raters. Collection closes at the first of:
  60 complete raters, or 14 days after the first invitation. If fewer
  than 40 are usable at that point, collection is extended once by 7
  days and then closes whatever the number. The date of the first
  invitation and the date of closing are recorded in an amendment below.
- The data are not analysed before collection closes. Before that, only
  the number of results is checked.

## Why 40

With the pilot's spread (SD of the interaction 1.10 over raters) and 40
raters, the chance that H2's interval lies above 0 is about 0.92 if the
true interaction equals the pilot's +0.58, and about 0.39 if it is half
of that. H1 is near certain at the pilot's effect. For H3, the chance of
support is about 0.99 if the true routine difference equals the pilot's
-0.24, and about 0.58 if it is exactly 0. A failed H2 therefore does not
show that urgency has no effect, only that a large one was not found.

## Analysis (`scripts/rating_study2_analyze.py`, written before the data)

- Used: results of this round (version "v2") with consent recorded and
  all 20 clips answered. A rater id that appears twice is kept once (the
  first).
- Left out and counted: a rater who gave the same rating to every clip.
- Unit: the rater. For each rater, the mean rating in each of the four
  cells; then the three differences. Mean over raters with a 95%
  percentile interval from 10,000 resamples of raters (seed 0).
- Reported next to the tests, not instead of them: the cell means, the
  mean of every clip, the same analysis for raters who heard the voice
  (exploratory), the optional age and field answers, the comments.
- Nothing else is left out. No clip is removed after the data are seen.
- Exploratory analyses that may be added later (for example a mixed
  model with rater and clip effects, or the pilot and this round
  together) are labelled exploratory in the paper.

## How the result is used in the paper

- The paper reports the result whatever it is, next to the pilot.
- The paper does not claim that an assistant should wait. If H2 is
  supported, it says that observers judged a delay of urgent messages
  worse than a delay of routine ones. If H3 is supported, it says that
  observers did not prefer delayed routine messages.

## Amendments

(None yet. Dates of the first invitation and of closing go here.)
