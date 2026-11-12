# When should an assistant speak? One-page summary (2026-11-12)

**Paper framing (decided by Harshil, 2026-11-12): the HoloAssist finding
with honest limits.** The claim: in assembly tasks, routine speech from
both the instructor and the worker avoids fine-motor work, and a video
model can read the hand state well enough to use it. The limit, stated
in the paper itself: the effect was not found in cooking (WTaG, checked
with labels) and not with a predicted hand state on Ego-EXTRA. The
paper reports these as failed replications, not as support.

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
- **WTaG**: 9.9 h, three kitchen recipes, a human instructor. Labels for
  every utterance (routine instruction or correction) and every step;
  no hand labels; 417 moments labelled by eye for this project.

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
   - A second encoder with no motion input (DINOv2 on single frames)
     gives 0.52, and 0.68 for objects it never saw; 91% of sessions are
     below 1 (`reports/holoassist_robustness_dino.md`).
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

10. **A third dataset repeats the Ego-EXTRA pattern, not the HoloAssist
    one** (2026-11-12, `reports/wtag_from_video.md`). WTaG: 9.7 h, three
    kitchen recipes, human labels for routine instruction and
    correction, no hand labels. With the HoloAssist hand-state model
    applied unchanged, the instructor's routine instructions start in
    predicted fine-motor seconds at 0.91 times chance (inside the chance
    range), corrections at 0.96. The USER's speech avoids these seconds
    (0.79; questions 0.68), and the instructor's answers follow
    (0.67). A labelled sample (`reports/wtag_label_check.md`, 417
    moments labelled by eye, blind) changes the reading:
    - Precise hand work is seen at 38% of routine-instruction onsets
      and 40% of random seconds (ratio 0.94, interval 0.69 to 1.26).
      The instructor does not avoid it. This agrees with the model.
    - It is seen at 37% of the user's question onsets (ratio 0.93,
      0.67 to 1.23). With labels the USER does not avoid it either.
      The model's 0.68 is not confirmed.
    - The hand-state model is weak on kitchen video (12% of its "fine"
      moments are screw or insert; AUC 0.68).
    The labels are by the AI assistant, one pass; a person should check
    a part. A small effect (down to about 0.7) cannot be excluded.

11. **On HoloAssist the avoidance belongs to the kind of action, not to
    its length** (2026-11-12, `reports/holoassist_action_length.md`,
    labels only). Routine speech inside screw, unscrew or insert
    actions: 0.46 times chance when the action is shorter than 2 s,
    0.38 at 2 to 5 s, 0.35 at 5 to 10 s, 0.28 at 10 s and more. Inside
    actions of all other verbs: 0.91, 1.02, 0.89, 0.76. So even a short
    screw or insert is avoided, and a long action of another kind is
    hardly avoided. Corrections are not held back in short fine-motor
    actions (1.18, 1.03) and are rare in long ones (0.38, 0.20).
    What this means for WTaG: its "precise work" is mostly pouring,
    spreading and measuring, which are "other verbs" in HoloAssist terms
    and are not avoided on HoloAssist either. The WTaG moments with
    screw or insert are few (33 of 417), and for them the interval (0.52
    to 2.80 for instructions) is too wide to decide. So WTaG shows that
    the effect does not extend to precise work in general; it does not
    test the screw-and-insert effect well.

12. **The effect is seen again in furniture assembly on Ego-EXTRA, with
    labels by eye** (2026-11-12, `reports/egoextra_label_check.md`). 240
    moments from the bike workshop and furniture assembly scenarios: 80
    unasked expert turns, 80 trainee turns, 80 random seconds, labelled
    blind, no model.
    - Furniture assembly: screw or insert is in progress at 40% of
      random seconds, at 12% of unasked expert onsets (ratio 0.29,
      interval 0.09 to 0.61) and at 22% of trainee onsets (0.55, 0.24
      to 1.02). Both pooled: 0.41 (0.22 to 0.72).
    - Bike workshop: screw or insert is rare (1 of 30 random seconds),
      so it cannot be tested. Other precise work (fitting a tyre,
      bleeding a brake) fills 80% of random seconds and 56% of speech
      onsets (ratio 0.70, 0.53 to 0.93).
    - Both scenarios pooled, screw or insert: 0.50 (0.29 to 0.87).
    This is the first sign of the HoloAssist effect outside HoloAssist:
    another camera, other people, a remote expert. It is one labeller
    (the AI assistant), one pass, 92 onsets in furniture assembly. It
    needs a human check before it carries weight in the paper.

## What it means

- Supported, on HoloAssist (assembly, 166 h, human hand labels):
  fine-motor moments are quiet moments, from both sides of the
  conversation; hand state from video carries timing information; the
  urgency of a message changes whether it waits. Robust inside the
  dataset (finding 8) and on its official split (finding 9).
- Supported on one dataset only: that the INSTRUCTOR holds routine
  speech during fine-motor work. It holds on HoloAssist (assembly, hand
  labels) and not on Ego-EXTRA or WTaG, where the hand state is a
  prediction of a model trained on assembly video. On WTaG a labelled
  sample gives the same answer without the model (finding 10), so the
  effect looks specific to HoloAssist's assembly tasks, not hidden by a
  weak model. On Ego-EXTRA's furniture assembly a labelled sample does
  show it (finding 12). The pattern so far: the effect appears where
  screw and insert actions are common (HoloAssist, furniture assembly)
  and not where they are rare (cooking). The earlier Ego-EXTRA null
  result came from the model, which does not find these actions there.
- Not supported outside HoloAssist, after the label check: the worker's
  side. With the predicted hand state the worker speaks less during
  "fine-motor" seconds on all three datasets, but on WTaG the labels do
  not show it (finding 10). On Ego-EXTRA and WTaG the predicted state
  is therefore not evidence about hand work. The sentence that follows
  is true only for the predicted state: the
  person doing the task speaks less during (predicted) fine-motor work.
- Not shown: that speaking during fine work HURTS the performer. All of
  this is observation. Mistakes are also rare during fine work, and a
  performer who stops may be asking for the next step, so "nothing needs
  saying" explains much of finding 1.
- Not new: using hand state as an input (HoloAssist's own benchmark,
  Sensible Agent). Also not new: that a message in the middle of manual
  assembly work costs something (an ergonomics experiment of 2017 found
  such notifications are missed more often), and that people interrupt
  sooner when the matter is urgent (2021). Possibly new: the hand state
  read from the camera as the signal that delays routine speech, checked
  against when real instructors speak and stay silent. See the search of
  2026-11-12 in `reports/research_gap_map.md`.

## What the paper needs next (order follows the chosen framing)

1. **The rating study** (`study/`, hypotheses fixed in
   `study/PREREGISTRATION.md`). It is now the main evidence that the
   hold rule is worth something to a person, because the observational
   effect did not replicate. It waits for 10 to 20 raters.
2. **A person checks a part of the WTaG labels**
   (`reports/wtag_hand_labels.csv`, 417 rows; 60 rows are enough for an
   agreement number). Until then the failed replication rests on one
   labeller, the AI assistant.
3. **Say why assembly differs from cooking.** Tested (finding 11): the
   length of the action is not the reason; the kind of action is. The
   paper's claim must therefore name the actions (screw, unscrew,
   insert), not "fine-motor work" or "precise work" in general. Open:
   why these three. A data set with many such actions outside
   HoloAssist would be the real replication (Ego-EXTRA's bike workshop
   and furniture assembly are candidates, with labels by eye).
4. **Make the Ego-EXTRA furniture result solid** (finding 12; done as a
   first sample, 0.41 with interval 0.22 to 0.72). It needs: a person
   who labels 60 of the 240 rows blind (`reports/egoextra_hand_labels.csv`),
   and a larger sample from the 57 furniture videos only.
5. The "should stay silent" evaluation (finding 7) stays in the paper as
   a second section. It does not depend on the hand-state effect.
   Open: how much later than a human an assistant may be.
6. Baselines with released weights (ProAssist, EgoVoice if released).

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
