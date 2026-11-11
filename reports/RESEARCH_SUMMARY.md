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
  utterance. 1,568 videos used with labels; 305 with video so far.
- **Ego-EXTRA**: 50 h, same setup with a remote expert, four scenarios
  (bike workshop, bakery, furniture assembly, kitchen). No action labels;
  transcripts, hand tracking, gaze, video.

## What was found

1. **People do not talk during fine-motor work.** While the performer
   screws, unscrews or inserts something, routine instructor speech
   starts at 0.34 times the chance rate. The performer's own questions:
   0.40. (HoloAssist, labels.)
2. **A video model can see this.** A small classifier on frozen VideoMAE
   features names the hand state well enough to keep about 90% of the
   gain that human labels give for predicting when the instructor
   speaks. (HoloAssist, 305 videos.)
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
6. **Experts told to hold back still speak up.** In Ego-EXTRA's "speak
   only when asked or necessary" recordings, experts start about 0.8
   unasked turns per minute, the same as in the proactive recordings.

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
   score this. Running at full scale now (all HoloAssist videos).
2. Tighten the fixed-need test (finding 3), and repeat it from video on
   all videos.
3. The worker-side signal (when does the person want to talk) as a
   smaller, better supported claim.
4. A small controlled study with participants. Only this can show cost.

## In the app

The hand-state classifier runs on the vector the verb model already
computes (0.5 ms per tick) and appears in the engine state as `"hands"`.
It is experimental: trained on assembly video, only indirectly checked
on kitchen video, and nothing acts on it yet.
