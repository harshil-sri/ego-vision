# Rating study: when should the assistant speak?

A 7-minute online study. Raters watch 20 short clips (plus one practice
clip) of a person in fine-motor work. An assistant says one sentence in
each clip. The rater answers one question: "The assistant spoke at a good
moment" (1 to 5).

## What is compared

Every clip exists in two versions. A rater sees each clip once, in one
version chosen at random (half and half inside each message type).

| | now | held |
|---|---|---|
| When the sentence is spoken | during the fine-motor work | when the app's rule releases it: the video head says the hands are free |
| Routine message (12 clips) | the real next instruction of the human instructor | same |
| Warning (8 clips) | "Stop. That is the wrong part." and similar | same |

The clips, the message moments and the release moments are chosen by
`scripts/rating_study_build.py` with fixed rules and a fixed seed. Nobody
picked clips by eye. The release moment comes from the video model, not
from the labels.

## Prediction (written before any rater took part)

See `PREREGISTRATION.md`.

## How to run it

1. Put the folder `study/` on any static web host. Two simple ways:
   - GitHub Pages: repository Settings, Pages, "Deploy from a branch",
     branch `main`, folder `/ (root)`. The study is then at
     `https://<user>.github.io/<repo>/study/`.
   - Local test: `python -m http.server 8765 --directory study`, then
     open `http://localhost:8765`.
2. Send the link to the raters. Ask them to use sound, and a laptop or a
   phone held sideways.
3. At the end each rater gets a code. They send it to you (chat or mail).
   Paste all codes into `study/responses/codes.txt`, one per line. A rater
   can also press "Download as file"; put those `.json` files in the same
   folder.
4. Run `python -m scripts.rating_study_analyze`. It writes
   `reports/rating_study.md`.

Optional: to collect answers without the copy step, set `ENDPOINT` at the
top of the script in `index.html` to a URL that accepts a POST (for
example a Google Apps Script web app that appends the body to a sheet).

## What to tell the raters

Only this: "It is a short study about when an AI assistant should speak.
It takes about 7 minutes. Please use sound." Do not tell them what is
compared or what we expect. A rater who knows the hypothesis is not
usable.

Do not take part yourself, and do not ask people who worked on the project.

## Limits

- The raters watch another person. They do not do the task. The study
  measures what observers find well timed, not what a user feels.
- 10 to 20 raters from one student society: young, technical, not a
  sample of all users.
- The voice is a synthetic Windows voice. The warnings are not tied to a
  real mistake in the clip.
- All clips are from HoloAssist (assembly and device setup).

Video: HoloAssist dataset, Microsoft, licence CDLA-Permissive-2.0.
