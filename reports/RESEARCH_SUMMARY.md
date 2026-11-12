# When should an assistant speak? One-page summary (2026-11-11)

Read this first. Details, numbers with intervals, and limits are in
`reports/research_gap_map.md`. Every number comes from a script in
`scripts/` (commands in `README.md`).

## The question

A camera on your head watches you do a task. An assistant can speak.
Papers in 2025 and 2026 decide the moment from what the task NEEDS
(a step ended, a mistake happened). The idea tested here: the moment
should also depend on what your HANDS are doing, because a remark in the
middle of fine work costs you something.

## The data

- **HoloAssist**: 166 h, a person does assembly tasks, a human instructor
  watches the camera and talks. Labels for every hand action and every
  utterance. 1,568 videos used with labels; 1,493 with video.
- **Ego-EXTRA**: 50 h, same setup with a remote expert, four scenarios
  (bike workshop, bakery, furniture assembly, kitchen). No action labels;
  transcripts, hand tracking, gaze, video.

## What was found

1. **People do not talk during fine-motor work.** While the performer
   screws, unscrews or inserts something, routine instructor speech
   starts at 0.34 times the chance rate. The performer's own questions:
   0.40. (HoloAssist, labels.)
2. **A video model can see this.** A small classifier on frozen VideoMAE
   features names the hand state. For predicting when the instructor speaks, the
   predicted hand state helps MORE than the human labels (+0.043 against
   +0.029 average precision). Routine speech starts in predicted
   fine-motor moments at 0.36 times chance. (HoloAssist, all 1,493
   videos.)
3. **At fixed need the pattern has the predicted shape.** When the next
   instruction is pending, it starts during fine-motor work at 0.17
   times the odds of idle hands. When a mistake is in progress, the
   correction is not delayed more by fine-motor work (0.99). A routine
   message waits; an urgent one does not.
4. **The instruction follows the end of an action by 1 to 2 seconds.**
5. **It did not transfer to the second dataset as stated.** On Ego-EXTRA,
   unasked expert speech shows no relation to measured hand motion, and
   no avoidance of fine-motor moments predicted by the HoloAssist model
   (1.11 times chance). The TRAINEE's speech does avoid them (0.90; 0.77
   in assembly).
   As a prediction task, video adds nothing to the dialogue history on
   Ego-EXTRA (who spoke how long ago predicts the next turn; the frozen
   video features do not improve it).
6. **Experts told to hold back still speak up.** In Ego-EXTRA's "speak
   only when asked or necessary" recordings, experts start about 0.8
   unasked turns per minute, the same as in the proactive recordings.

7. **"Wait and see" is a poor way to stay silent.** Performers fix 43% of
   their mistakes alone, in about 2 s. Instructors who correct also
   speak after about 2 s. So an assistant that waits before correcting
   is either late or speaks when it was not needed. A model that knows
   the kind of mistake and the history of the session does much better:
   at 20% failed corrections it speaks without need on 29% of
   self-corrected mistakes, waiting on 87%. (HoloAssist labels, "drop"
   mistakes removed.) From video: a model that sees 1 s of the mistake is too
   late if it must be as fast as the instructor (who has then corrected
   24% of the mistakes). If it may be up to 2 s later, history + video
   is as good as the labels (30% unneeded at 20% failed). Video of the
   2 s BEFORE the mistake helps a little and is not late (37% against
   41% for history alone).

8. **Finding 1 does not depend on our choices** (2026-11-12,
   `reports/holoassist_robustness.md`). Routine speech in fine-motor
   seconds, relative to chance:
   - 95% of 240 recording sessions are below 1 (median 0.35 to 0.38).
   - 18 of the 19 task types with fine-motor work are below 1. The
     exception is one coffee task where such work is 5% of the time.
   - A hand-state model that never saw the object of the test video
     still gives 0.49 (0.42 when it has seen the object, 0.40 with labels).
     A plain logistic regression gives 0.55.
   - Screw, unscrew and insert are the three most avoided of the 22 hand
     states (0.27 to 0.49; the next is 0.70). The grouping is a fair
     summary of the data, not a selection.
   - The effect is tied to the moment: 0.40 at the onset, 0.56 one second
     before, 0.72 two seconds before, 0.86 four seconds before.

9. **The same result on the dataset's official split**
   (`reports/holoassist_official_split.md`; train 1,306 videos, val 186).
   Average precision for "routine speech starts in the next second":
   task features 0.115; plus hand state from video 0.160 (+0.045,
   interval +0.034 to +0.057); plus the labelled hand state 0.147. Onset
   F1 with a tolerance of 1 s early to 2 s late: 0.449 to 0.481.

## What it means

- Supported: fine-motor moments are quiet moments, from both sides of the
  conversation; hand state from video carries timing information; the
  urgency of a message changes whether it waits.
- Not shown: that speaking during fine work HURTS the performer. All of
  this is observation. Mistakes are also rare during fine work, and a
  performer who stops may be asking for the next step, so "nothing needs
  saying" explains much of finding 1.
- Not new: using hand state as an input (HoloAssist's own benchmark,
  Sensible Agent). Possibly new: hand state as a cost that delays speech,
  checked against when real instructors speak and stay silent.

## What to do with it (my order)

1. A "should stay silent" evaluation: mistakes that performers fix alone
   are moments where a good assistant says nothing. Benchmarks do not
   score this. A first score exists (finding 7). Open: decide how much later than a human an
   assistant may be; the answer changes which gate wins.
2. Tighten the fixed-need test (finding 3), and repeat it from video on
   all videos.
3. The worker-side signal (when does the person want to talk): present
   on both datasets, but small on Ego-EXTRA.
4. A small controlled study with participants. Only this can show cost.
   A first step is built and waits for raters (`study/`): observers rate
   the timing of a routine message or a warning, spoken during fine-motor
   work or when the rule releases it. The hypotheses are fixed in
   `study/PREREGISTRATION.md`.

## In the app

The hand-state classifier runs on the vector the verb model already
computes (0.5 ms per tick) and appears in the engine state as `"hands"`.
It is experimental: trained on assembly video and only indirectly
checked on kitchen video.

One rule now acts on it (`src/pipeline/delivery.py`, only when the app is
started with `--hand-state-dir`): a routine step message waits while the
fine-motor probability is 0.5 or more, for at most 8 s; a wrong-order
warning never waits. The released message is a `"say"` event, and the
waiting message is `"held_message"` in the engine state. The screen
still shows every message at once: the rule is for a future voice output.
Not tested on a real kitchen recording.

The two numbers come from a replay of the rule on all 1,493 HoloAssist
videos (`reports/holoassist_delivery_sim.md`), with the hand state read
from video:

- Messages at random moments (a timer, a tip): 32% arrive during
  labelled fine-motor work without the rule, 21% with it (ratio 0.62 to
  0.67). 29% of the messages wait, 5 s on average; the mean delay over
  all messages is 1.4 s. With the labelled hand state instead of the
  video head the same rule reaches 10%.
- The time limit matters more than the threshold. Fine-motor work
  continues for a median of 6 s (10 s at the 75th percentile). With the
  first guess of 4 s, half of the held messages were released while the
  work still went on. 8 s is a judgement: how long a user accepts to
  wait is not measured.
- Messages at step boundaries (the app's "next step" message): only 1.5%
  fall in fine-motor work, because a step boundary is a moment between
  actions. The rule has almost nothing to prevent there. It holds 7% of
  these messages for about 1 s, mostly because the video head still
  reports the fine-motor work that has just ended (it looks at the last
  2 s). So the rule matters for messages that do not depend on the task
  timing, not for step messages.
- The app's hand-state head is now trained on all 1,493 videos (before:
  305). The indirect kitchen check improved from 0.60 to 0.63.
