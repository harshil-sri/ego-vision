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

## In the app

The hand-state classifier runs on the vector the verb model already
computes (0.5 ms per tick) and appears in the engine state as `"hands"`.
It is experimental: trained on assembly video and only indirectly
checked on kitchen video.

One rule now acts on it (`src/pipeline/delivery.py`, only when the app is
started with `--hand-state-dir`): a routine step message waits while the
fine-motor probability is 0.5 or more, for at most 4 s; a wrong-order
warning never waits. The released message is a `"say"` event, and the
waiting message is `"held_message"` in the engine state. The screen
still shows every message at once: the rule is for a future voice output.
The two numbers are first guesses. Not tested on a real kitchen recording.
