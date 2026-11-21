# Messages for round 2 (raters) and for the labellers

## Raters (anonymous, about 7 minutes)

Send it personally where you can (a direct message works much better
than a group post), and post it in groups too. Do not change what it
says about the study: the raters must not know what is compared.

> Hi! I'm running a short study for my research paper on AI assistants
> in smart glasses. It takes about 7 minutes: you watch 20 short clips
> and rate each one from 1 to 5. It's anonymous, no name or number
> needed, and it works on a phone (hold it sideways, sound on).
>
> https://harshil-sri.github.io/ego-vision/study/rate/
>
> It would help me a lot. Everyone who takes part is thanked in the
> paper, and I'll share the paper with you when it's out. Please don't
> discuss the clips with others until they've done it. Thank you!

A reminder after 3 to 4 days to those who said yes but did not do it
yet is normal and polite. Ask everyone who finished to forward the link
to one or two friends (18 or older).

## Rules for you

- Do not take part yourself. Do not ask the 14 pilot raters or anyone
  who has seen the project. (The page also stops people who say they
  did it before.)
- Do not say what is compared or what you expect, also not after one
  person has finished.
- Write the date you send the first invitation into the amendment
  section of `PREREGISTRATION_v2.md`. Collection closes at 60 raters or
  14 days after that date (extended once by 7 days if under 40).
- At the end: download the `ratings` tab as CSV into
  `study/responses_v2/ratings.csv`, put any codes people sent you in
  `study/responses_v2/codes.txt`, then run
  `python -m scripts.rating_study2_analyze`.

## Labellers (about 12 minutes per part)

The links are in `runs/label_check/relabel/links.md` (private: never post
them in a public place). Each part needs TWO different people; note the
names in that file so nobody gets the same part twice. One person can
do several parts. Do the furniture parts (egoextra) first if you have few
people: those numbers matter most for the paper.

> Hi [name]! Could you help me with my research paper? I need people to
> label short picture sequences of hands at work: for each one you choose
> one of three buttons (what the hands are doing). There's a short guide
> and a practice round first, then 70 moments; about 12 minutes in total,
> and you can stop and continue later. Everyone who helps is thanked by
> full name in the paper's acknowledgements.
>
> [link]
>
> Please do it on your own and don't compare answers with anyone.
> Thank you!

If a labeller cannot send automatically, they download a file at the end
and send it to you. Put those files in `runs/label_check/relabel/answers/`.
