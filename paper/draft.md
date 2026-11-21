# Instructors stay quiet during screw and insert actions. Observers do not reward an assistant that does the same.

Draft 1, 2026-11-13. Author: Harshil. Status: first full draft, written
from the reports in `reports/`. Every number below comes from a script in
`scripts/`; the report file is named next to each result. Section 9 lists
what must be checked before submission.

## Abstract

An assistant that watches a task through a head-worn camera must decide
when to speak. Recent systems decide from what the task needs: a step
ended, or a mistake happened. We ask a second question: does the moment
also depend on what the hands are doing?

We measure when human instructors start to speak in HoloAssist (166 hours
of assembly and device tasks with labels for every hand action). While
the performer screws, unscrews or inserts a part, routine instructor
speech starts at 0.34 times the rate that chance would give. The
performer's own questions start at 0.40 times chance. The effect belongs
to these three actions, not to precise work in general and not to long
actions. Urgency changes it: when a mistake is in progress, the
correction is not delayed by such an action (odds ratio 0.99, 95%
interval 0.84 to 1.20). A pending instruction comes most often 1 to 2 s
after the hands become free. When an instructor does speak during such
an action, the hands stop sooner, and we find no robust rise in
mistakes. A small classifier on frozen video features reads
the hand state well enough to improve the prediction of speech onsets
(average precision 0.119 to 0.161). A published proactive assistant,
ProAssist, also speaks less during these actions in its released
outputs (0.61 times chance), less strongly than the instructors of the
same videos (0.49), and it speaks least while the hands are idle, where
the instructors speak most.

Outside HoloAssist the picture is mixed, and we report all of it. In
cooking video (WTaG) a blind labelled sample shows no clear avoidance of
precise hand work by the instructor (0.81, interval 0.59 to 1.08),
while the user's own questions do avoid it (0.66, interval 0.46 to
0.92). In furniture assembly
(Ego-EXTRA) a blind labelled sample shows the effect at a smaller size
(0.59, interval 0.40 to 0.86), with one labeller, two batches that
differ, and only fair agreement with a human checker (kappa 0.37 for
screw or insert).

Last, we test whether the observation gives a rule for an assistant. In a
pre-registered rating study, 14 observers rated the same message spoken
during the action or held until the hands were free. They did not rate
the held routine message higher (difference -0.24 on a 5-point scale,
interval -0.60 to +0.14). They rated a held warning clearly lower
(-0.82, interval -1.14 to -0.46). We therefore report a description of
human instructors and a negative result for the rule. We do not claim
that an assistant should wait.

## 1. Introduction

A person assembles a shelf. A camera on the person's head sees the task,
and an assistant can speak. The useful sentence is often clear: the next
step, or a warning. The open question is the moment.

Work from 2025 and 2026 answers from the side of the task. The assistant
speaks when a step ends, when an error is predicted, or when a learned
model of dialogue says so (ProAssist; Pro²Assist; TRAFA; EgoVoice;
EgoServe). Work in human-computer interaction answers from the side of
the person: an interruption costs less at a task breakpoint (Adamczyk and
Bailey 2004; Iqbal and Bailey 2008), and in manual assembly a
notification sent at a fine breakpoint, between small steps, is missed
more often than one sent between larger task units (Kolbeinsson et al.
2017). The second line of work mostly uses desk
sensors or pre-defined breakpoints, not a camera on the head.

This paper connects the two with data. Three datasets record a human
instructor who watches the camera of a performer and speaks freely. The
instructor's timing is a record of when an experienced person finds it
right to speak. We ask three questions.

1. **Description.** Does instructor speech avoid some hand states of the
   performer? Which states, and does urgency change the answer?
2. **Sensing.** Can a model read these states from the video, well enough
   to help predict when the instructor speaks?
3. **Rule.** If an assistant holds a routine message until the hands are
   free, do people who watch find its timing better?

The answers are: yes for a narrow set of actions (screw, unscrew,
insert); yes on the dataset where the model was trained, and no outside
it; and no. We give the three answers the same weight. The third answer
was fixed in a pre-registration before any rating was read, and it went
against our prediction.

**Contributions.**

- A measurement, on 1,568 HoloAssist videos, of how instructor and
  performer speech depend on the labelled hand state, with a test at
  fixed need that separates routine speech from corrections, the timing
  of a held instruction after the action ends, and what follows in the
  cases where the instructor speaks during the action (section 4).
- Evidence that the effect is specific to screw, unscrew and insert
  actions, and tests of the same question on two other datasets with
  blind labels: no clear instructor effect in cooking, a weaker one in
  furniture assembly, and a performer-side effect in both (sections
  4.4 and 6).
- A hand-state classifier on frozen video features that carries more
  timing information than the human hand labels on HoloAssist, and that
  does not transfer to the other two datasets (section 5).
- A pre-registered rating study that does not support the hold rule for
  routine messages and does support "a warning must not wait"
  (section 7).
- A separate evaluation of when an assistant should stay silent on a
  mistake (section 8).

## 2. Related work

**When proactive assistants speak.** ProAssist (arXiv 2506.05904) learns
the moment from dialogues that a language model generated. Pro²Assist
(2605.04227) uses optical flow in the hand regions to select key moments;
more hand motion makes help more likely there. TRAFA (2605.24526) uses
forecast hand motion to predict an error before it happens. In all three,
the hands are evidence of need. EgoVoice (2610.12248) trains a
listen-or-speak decision on HoloAssist instructor speech, the same onsets
that we analyse; it has no hand-state input. EgoServe, the benchmark of Vinci2 (2607.11523), scores
timing against annotated event windows. YETI (2501.09355) triggers on
scene change. The HoloAssist benchmark (2309.17024) already shows that
hand pose helps to predict the type of the next intervention. Using the
hands as an input is therefore not new.

**Interruption cost.** ProMemAssist (2507.21378) weighs the value of a
message against a modelled working-memory load and names hand interaction
as a cue for future work. Sensible Agent (2509.09255) uses gaze, ambient
noise and whether the hands are free to choose how a suggestion is
given (audio, visual, gesture or nothing); it does not choose when. In
its study (10 users) this lowered perceived effort against a voice-only
baseline. EgoIntrospect (2605.17262) records 180 hours from 60 wearers.
In one of its tasks the wearers marked afterwards the segments in which
they would not want a proactive suggestion, and a model must judge each
segment as interruptible or not. The labels are per segment and come
from the wearer; ours are per moment and come from an instructor's
behaviour. Kolbeinsson et al. (2017) sent notifications during a
simulated assembly task at one of two kinds of breakpoint. Participants
missed more notifications at fine breakpoints (between small steps)
than at coarse ones (between larger task units); the authors explain
this by an overlap of the current action with the anticipation of the
next one. Edwards et al. (2021, arXiv 2106.02077)
find that people interrupt a busy person (a Tetris player) sooner when
the matter is urgent, and several of them say they waited for a subtask
boundary. When2Talk (2609.12503) delays or withholds the speech of an
in-car agent by event priority and passenger activity; in a study with
41 passengers the delay lowered perceived interruption but also lowered
rated usefulness. A cost of waiting is therefore already reported, in a
different setting.

**What is different here.** We do not propose a new signal. We measure,
on recordings of real instructors, which hand states they avoid, we test
the measurement at fixed need and outside the first dataset, and we test
the rule that follows from it with human raters. We found no earlier work
that does these three things together. Our search was by keyword and is
not a systematic review.

## 3. Data

| Dataset | Hours | Task | Speaker | Hand labels |
|---|---|---|---|---|
| HoloAssist | 166 | assembly, device setup | instructor in the room | every action (verb, object) |
| Ego-EXTRA | 50 | bike workshop, bakery, furniture assembly, kitchen | remote expert | none; 420 moments labelled for this paper |
| WTaG | 9.9 | three kitchen recipes | instructor in the room | none; 417 moments labelled for this paper |

**HoloAssist.** We use 1,568 of 1,758 videos that have labels (113.8
hours inside task windows) and 1,493 with video. The labels give every
fine-grained action with a verb, and every utterance with a type. We call
an instructor utterance *routine* when it is a next-step instruction, a
follow-up or a confirmation, and a *correction* when it answers a
mistake. We group the 20 verbs into four hand states: *idle* (no
action, 21% of task time), *fine* (screw, unscrew, insert; 30%), *hold, place or inspect* (12%) and *other action*
(37%). Section 4.3 shows that the grouping follows the data.

**Ego-EXTRA and WTaG** have transcripts with times and no action labels.
For both we labelled moments by eye (section 6.1).

## 4. When do instructors speak? (HoloAssist, labels)

### 4.1 Method

For each video we count the speech onsets that fall inside each hand
state. The chance level comes from a circular shift: all onsets of a
video are moved by one random offset, which keeps the gaps between
utterances and breaks the link to the hands. We repeat this 1,000 to
2,000 times. A ratio is the observed count over the mean shifted count.

### 4.2 Result: routine speech avoids fine-motor actions

| Onsets | n | idle | fine | hold, place, inspect | other action |
|---|---|---|---|---|---|
| Instructor starts, routine | 18,649 | 1.48 | **0.34** | 1.31 | 0.81 |
| Instructor starts, correction | 2,999 | 0.65 | 0.59 | 1.16 | 1.43 |
| Performer asks a question | 851 | 1.19 | **0.40** | 1.36 | 0.94 |
| Instructor replies | 1,031 | 1.04 | 0.41 | 1.39 | 1.05 |

Each cell is observed over expected. For the fine column, 95% of the
shifts lie between 0.85 and 1.16 in every row
(`reports/holoassist_student_speech.md`). Both sides of the conversation
start to speak less during fine-motor actions.

### 4.3 The effect belongs to three actions

- Screw, unscrew and insert are the three most avoided of the 22 hand
  states (0.27 to 0.49). The next state is at 0.70
  (`reports/holoassist_robustness.md`).
- The length of the action does not explain it. Routine speech inside
  screw, unscrew or insert actions is at 0.46 times chance when the
  action is shorter than 2 s, and 0.38, 0.35 and 0.28 for longer ones.
  Inside actions of all other verbs the ratios are 0.91, 1.02, 0.89 and
  0.76 (`reports/holoassist_action_length.md`).
- The effect is tied to the moment: 0.40 at the onset, 0.56 one second
  before, 0.72 two seconds before, 0.86 four seconds before.
- It is general inside the dataset: 95% of 240 recording sessions and 18
  of the 19 task types with fine-motor work are below 1.

So the claim must name the actions. "Fine-motor work" in this paper means
screw, unscrew and insert.

### 4.4 At fixed need: routine speech waits, a correction does not

The ratio in 4.2 could come from need alone: nothing needs saying while a
screw goes in. We test this in two situations where the need is known
(`reports/holoassist_need_controlled.md`).

**A next-step instruction is pending** (3,928 episodes from a step
boundary to the instruction). A discrete-time model of the onset, at
equal time since the boundary, gives odds of 0.17 (0.13 to 0.23) for the
instruction to start during a fine-motor action, against idle hands.

**A mistake is in progress and will be corrected** (3,256 mistakes). The
term for "the wrong action is a fine-motor action and still continues"
has an odds ratio of 0.99 (0.84 to 1.20). The correction is not delayed
by the fine-motor action.

So at fixed need the pattern has the shape that an interruption-cost
account predicts: the routine message waits, the urgent one does not.
This is still an observation. It does not show that speaking during the
action would harm the performer.

### 4.5 Held, then released

If instructors only had nothing to say during a screw, the pending
instruction would come at any time after the screw ends. If they hold it,
it should come soon after the hands become free. We test this inside the
episodes of 4.4 (a next-step instruction is pending), on the seconds in
which the hands are idle, at equal time since the step boundary
(`reports/holoassist_idle_spell.md`). The reference is a hand pause that
is already 4 s or longer.

| Hands idle for | After a screw, unscrew or insert | After any other action |
|---|---|---|
| 0 to 0.5 s | 1.02 (0.48 to 1.99) | 1.36 (1.18 to 1.58) |
| 0.5 to 1 s | 2.19 (1.34 to 3.48) | 1.66 (1.39 to 1.91) |
| 1 to 2 s | **3.15** (1.80 to 6.44) | 1.79 (1.53 to 1.95) |
| 2 to 4 s | 2.03 (1.19 to 4.17) | 1.43 (1.24 to 1.67) |

Each cell is the odds that the instruction starts in that half second.
After every kind of action the instruction is most likely 1 to 2 s into
the pause. After a fine-motor action the peak is higher and comes after
a short delay (no rise in the first half second). The two columns
overlap, so we do not claim that they differ. The shape is what a held
message predicts: it waits for the action to end and for a short moment
after, and then it comes.

### 4.6 What follows when an instructor does speak during the action?

In 910 cases a routine instruction did start during a screw, unscrew or
insert action. We compare what followed with the same onsets moved by
the circular shift of 4.1, which lands them in actions in proportion to
their length (`reports/holoassist_speech_cost.md`).

| Outcome after a routine onset | During fine-motor work | During other actions | Fine over other |
|---|---|---|---|
| The action ends within 1.5 s | 1.48 (1.38 to 1.59) | 1.07 (1.05 to 1.09) | 1.39 (1.29 to 1.49) |
| The same verb and noun start again within 10 s | 0.92 (0.82 to 1.02) | 0.91 (0.87 to 0.95) | 1.01 (0.90 to 1.13) |
| A wrong action starts within 15 s | 1.19 (1.02 to 1.37) | 0.98 (0.94 to 1.02) | 1.22 (1.04 to 1.40) |

- **The hands stop.** A fine-motor action ends within 1.5 s of the
  speech more often than chance, far more than other actions do. The
  labels of the performer's hands around all instructor onsets agree:
  after a next-step instruction or a confirmation given while the hands
  are busy, the hands are idle for 0.5 to 0.6 s more in the next 5 s
  than chance gives; after a follow-up they are not
  (`reports/holoassist_aftermath.md`). This does not tell which came
  first: a performer who pauses may be the reason the instructor speaks.
- **The action is not broken off more often.** It is not resumed more
  often than chance, and not more often than other actions.
- **No robust rise in mistakes.** The raw ratio for mistakes is above 1,
  but it comes from follow-up instructions (1.42, 1.18 to 1.68), which
  often react to trouble that is already visible. For next-step
  instructions it is 0.88 (0.37 to 1.38) and for confirmations 1.00
  (0.76 to 1.25). Counting only mistakes that the performer corrected
  alone gives 1.04 (0.80 to 1.30). Leaving out onsets that fall inside a
  wrong action gives 1.16 (0.96 to 1.37) for fine over other.

So in these recordings speech during fine-motor work goes with a pause
of the hands, and we find no clear sign of a cost beyond that. This
matches the rating study of section 7, where observers did not reward a
held message. The data cannot show what would have happened without the
speech.

## 5. Can a model read the hand state from video?

### 5.1 Method

A frozen VideoMAE-B encoder (trained on Something-Something v2) gives one
vector for the last 2 s at each second. A small network predicts the hand
state (22 classes). An onset model (gradient-boosted trees) predicts
"routine instructor speech starts in the next second" from task features
(the task type, the position in the task, the time since the last
utterance, and similar), with or without the hand state. All splits are by recording
session, 5 folds, and the hand-state predictions are cross-fitted.

### 5.2 Results on HoloAssist

- Hand state, 22 classes: accuracy 0.386 (the most common state: 0.207).
  Recall is best for screw (0.61), unscrew (0.52) and insert (0.46).
- Onset prediction, average precision (base rate 0.043;
  `reports/holoassist_from_video.md`):

| Features | Average precision | Change against task |
|---|---|---|
| Task | 0.119 | |
| Task + labelled hand state | 0.147 | +0.029 (+0.025 to +0.032) |
| Task + predicted hand state | 0.161 | +0.043 (+0.038 to +0.047) |
| Task + raw video vector | 0.150 | +0.032 (+0.028 to +0.036) |

- The predicted hand state helps more than the human labels. A likely
  reason is that the prediction is a probability and carries more than
  the one label does.
- On the official split of the dataset the result is the same: 0.115 to
  0.160 (+0.045, interval +0.034 to +0.057)
  (`reports/holoassist_official_split.md`).
- Routine speech starts in predicted fine-motor seconds at 0.36 times
  chance. With a hand-state model that never saw the object of the test
  video the ratio is 0.49. With a second encoder that has no motion input
  (DINOv2 on single frames) it is 0.52.

### 5.3 The model does not transfer

Applied unchanged to Ego-EXTRA and WTaG, the HoloAssist model gives no
avoidance for the instructor: 1.11 times chance for unasked expert speech
on Ego-EXTRA and 0.91 for routine instructions on WTaG. Section 6 shows
that for Ego-EXTRA this null result came from the model. On kitchen video
only 16% of the moments that the model calls fine are screw or insert
actions (`reports/wtag_label_check.md`).

### 5.4 A published assistant: when does ProAssist speak?

ProAssist (2506.05904) watches the stream at 2 frames per second and
decides at each frame whether to speak. Its authors released their
evaluation outputs with the model (checkpoint L4096-I1, the not-talk
threshold of 0.3 that they chose). The outputs include HoloAssist videos,
so we can apply the test of section 4.1 to ProAssist's own speech times
without running the model. The stream time equals the video time: the
turns of ProAssist's reference dialogue sit closest to the labelled
action ends at zero lag. That reference dialogue was generated by a
language model from the HoloAssist annotations; it is ProAssist's
training target, not real speech. Each video has three samples; the
interval is a bootstrap over videos (`reports/proassist_baseline.md`).

| Onsets (test split, 86 videos) | n | idle | fine | hold, place, inspect | other action |
|---|---|---|---|---|---|
| ProAssist speaks | 5,416 | 0.68 (0.61 to 0.75) | **0.61** (0.56 to 0.67) | 1.03 | 1.40 |
| ProAssist reference dialogue | 11,903 | 1.37 | 0.58 (0.54 to 0.62) | 1.01 | 1.09 |
| Real instructor, routine | 988 | 1.47 (1.33 to 1.64) | **0.49** (0.38 to 0.60) | 1.19 | 0.76 |

- ProAssist already speaks less during fine-motor actions. Its training
  dialogue does too, because the generated turns are placed at action
  ends. The avoidance therefore comes with the training data; it is not
  something ProAssist lacks.
- It avoids them less than the real instructors of the same videos. The
  difference is +0.12 (interval -0.01 to +0.25) on the test split and
  +0.32 (+0.21 to +0.42) on the validation split, where the instructors
  are at 0.29.
- The clearest difference is idle time. Instructors start routine speech
  1.47 times as often as chance while the hands are idle (1.70 on
  validation); ProAssist does so 0.68 times (0.67). ProAssist speaks most
  during other actions (1.40), where instructors speak less (0.76).

This is a comparison of timing only. It does not say which timing users
prefer, and section 7 found no preference for the held message.

## 6. Outside HoloAssist: two samples with blind labels

### 6.1 Protocol

For each moment a sheet shows four frames, 0.5 s apart, from the 2 s that
end at the moment. Speech onsets and random seconds of the same
recordings are shuffled together. The labeller does not know which is
which; the key is stored apart. Three labels: **S** (screw, unscrew or
insert in progress), **B** (other precise work in progress, for example
pouring, spreading, hammering, fitting a tyre), **F** (everything else).
The ratio is the share of a label at onsets over its share at random
seconds, with a bootstrap interval.

**Limit that applies to all of section 6:** the labeller was the AI
assistant used in this project, in one pass. A human check of 60 moments
per dataset is in progress; the first result (furniture, one checker) is
in section 6.4 and shows only fair agreement.

### 6.2 Cooking (WTaG): no clear instructor effect, a user effect

417 moments (`reports/wtag_label_check.md`).

| Onsets | Hands | At onsets | At random seconds | Ratio |
|---|---|---|---|---|
| Instructor, routine instruction | S or B | 34% | 42% | 0.81 (0.59 to 1.08) |
| User, question | S or B | 28% | 42% | 0.66 (0.46 to 0.92) |
| Instructor, routine instruction | S | 8% | 8% | 0.89 (0.34 to 1.98) |
| User, question | S | 6% | 8% | 0.70 (0.21 to 1.50) |

The instructor's interval includes 1: there is no clear avoidance of
precise hand work, and a moderate one is not excluded. The user's
questions do start less often during precise hand work. Screw and insert
actions are rare in WTaG (31 of 417 moments), so WTaG does not test the
narrow effect; the intervals for S are too wide to decide.

A note on these labels: a first set was made from wrongly timed
pictures (a seek error in the frame grab). All 417 moments were labelled
again from correct frames; the numbers above are from the second set.

### 6.3 Furniture assembly (Ego-EXTRA): a weaker effect

420 moments in two batches (`reports/egoextra_label_check.md`). The bike
workshop scenario has too few screw or insert moments at random seconds
(1 of 30) for a ratio. Furniture assembly:

| Sample | Speech onsets with S | Random seconds with S | Ratio |
|---|---|---|---|
| First batch | 16% of 92 | 40% of 50 | 0.41 (0.21 to 0.70) |
| Second batch | 23% of 120 | 30% of 60 | 0.78 (0.48 to 1.35) |
| Both batches | 20% of 212 | 34% of 110 | 0.59 (0.40 to 0.86) |

With both batches, unasked expert turns are at 0.60 (0.37 to 0.93) and
trainee turns at 0.57 (0.33 to 0.89). The second batch was drawn to test
the first, and alone it does not show the effect. We use the pooled 0.59
and state that it is smaller than the 0.34 of HoloAssist and that the two
batches differ.

### 6.4 Agreement with a human checker (furniture, first checker)

A person who did not know the first labels, the kind of each moment or
the hypothesis labelled 60 furniture moments (40 speech onsets, 20 random
seconds, drawn at random from the 420) on the same four frames
(`reports/egoextra_label_agreement.md`).

| Labels | Agreement | Cohen's kappa |
|---|---|---|
| S, B, F | 0.68 (0.57 to 0.80) | 0.36 (0.15 to 0.56) |
| S against the rest | 0.83 (0.73 to 0.92) | 0.37 (0.05 to 0.66) |

This is fair agreement, not good agreement. The checker used S less
often than the first labeller (5 against 13 of 60 moments). Of the 13
moments that the first labeller called S, the checker called 4 S, 3 B
and 6 F. In the moments we inspected the disagreement is about holding
or aligning a part against inserting it, which four still frames do not
always show. So the furniture result of section 6.3 depends on a label
that two labellers do not apply the same way. With 60 moments the
checker's own labels cannot test the effect (one S at 20 random seconds).
A second checker will show whether two people agree better with each
other than with the first labeller; if they do not, the label itself is
the limit, not the labeller.

### 6.5 Reading

The instructor effect appears where screw and insert actions are common
(HoloAssist, furniture assembly) and is not testable where they are rare
(cooking, the bike workshop). For precise hand work in general it is not
clear outside HoloAssist. The performer's own speech avoids such work in
all three datasets (0.40 on HoloAssist, 0.57 in furniture assembly, 0.66
in cooking).

## 7. Does the observation give a rule? A pre-registered rating study

### 7.1 Design

The rule: a routine message waits while the video model reports
fine-motor work, for at most 8 s; a warning never waits. We built 20
clips from HoloAssist with fixed rules and a fixed seed
(`scripts/rating_study_build.py`). In each clip a person is in a screw,
unscrew or insert action that ends inside the clip. A synthetic voice
says one sentence in one of two versions: **now** (during the action) or
**held** (when the rule releases it, 2.5 to 7 s later). 12 clips carry a
routine message, which is the real next instruction of the human
instructor. 8 carry a warning ("Stop. That is the wrong part." and
similar). Each rater sees each clip once, in a version chosen at random,
and rates "The assistant spoke at a good moment" from 1 to 5.

The hypotheses, the analysis script and the exclusion rule were committed
before any rater took part (`study/PREREGISTRATION.md`). H1: held is
rated higher for routine messages. H2: held is not rated higher for
warnings. H3, the main test: the difference held minus now is larger for
routine messages than for warnings. The unit is the rater; intervals come
from 10,000 resamples of raters.

### 7.2 Result

14 raters; none excluded (`reports/rating_study.md`).

| Message | Now | Held | Held minus now |
|---|---|---|---|
| Routine | 3.74 | 3.50 | -0.24 (-0.60 to +0.14) |
| Warning | 3.70 | 2.88 | -0.82 (-1.14 to -0.46) |

- **H1 is not supported.** The direction is against the prediction: 10
  of 14 raters rated the held routine message lower.
- **H2 is supported.** A held warning is rated clearly lower.
- **H3** is +0.58 (+0.02 to +1.13). It meets the fixed criterion, but
  the interval nearly touches 0 and the difference comes from the
  warnings.

By the pre-registration, the failure of H1 counts against the rule.
Observers do not prefer a routine message after the fine-motor work.

### 7.3 Limits of the study

- The raters watched another person. They did not do the task. The study
  measures what observers find well timed, not the cost to a performer.
- 14 raters from one student society, recruited by the author.
- A held message is some seconds late, and it can then describe a step
  that the video has already passed. One rater wrote that the narration
  and the video did not match. The design cannot separate "late" from
  "after the hands are free".
- The warnings were not tied to a real mistake in the clip.
- One departure from the plan: collection was closed after one day at 14
  raters, not after 7 days without a new rater. The amendment was
  committed before any rating was read.

### 7.4 What the rule does in a replay

For completeness (`reports/holoassist_delivery_sim.md`): replayed on all
1,493 videos, the rule moves messages that arrive at random moments out
of labelled fine-motor work (32% without the rule, 21% with it), at a
mean delay of 1.4 s. Messages at step boundaries hardly ever fall in
fine-motor work (1.5%), so the rule has almost nothing to do for them.

## 8. A second question: when to stay silent on a mistake

This part does not depend on the hand-state effect
(`reports/holoassist_wait_policy.md`). Performers fix 43% of their
mistakes alone. A self-corrected wrong action lasts 1.95 s (median), and
the instructor corrects 1.99 s (median) after a wrong action starts. A
policy of "wait, then speak if the mistake continues" therefore cannot
separate the two cases: at 20% failed corrections (silent or later than
the instructor) it speaks without need on 87% of the self-corrected
mistakes. A model that knows the kind of mistake and the history of the
session reaches 29% (interval 25% to 33%). From video, a model that may
be up to 2 s later than the instructor reaches 30%. "Failed" here means
"later than a human", not harm: the instructor's correction ends the
wrong action, so nobody knows whether the performer would have fixed it.

## 9. Limits and open work

- **Observation, not cause.** Sections 4 to 6 describe when instructors
  speak. Mistakes are also rare during fine-motor work, and a performer
  who stops may be asking for the next step. The test at fixed need
  (4.4) reduces this problem and does not remove it.
- **One labeller outside HoloAssist.** The labels of section 6 are from
  an AI assistant, one pass. A person labels 60 moments of each sample
  blind (`scripts/label_check_kit.py`). First furniture checker: kappa
  0.36 for three labels, 0.37 for S against the rest (section 6.4).
  [TO DO: second furniture checker; two cooking checkers]
- **The furniture result is modest.** 212 onsets, two batches that
  differ, interval 0.40 to 0.86.
- **The model does not transfer.** The hand-state classifier is trained
  on HoloAssist and is weak on other cameras and tasks.
- **The rating study is small** and uses observers. A stronger test
  needs the performer and a held message that is still correct when it
  is spoken. That is a new pre-registration.
- **Baselines.** Section 5.4 compares the timing of ProAssist from its
  released outputs; we did not rerun the model, and its HoloAssist
  outputs cover 86 to 87 videos per split. EgoVoice announced code and
  weights for November 2026; they were not public on 2026-11-21.
  The released outputs also cover WTaG, but our WTaG labels exist only
  at sampled moments, so the test of section 4.1 cannot be applied
  there without new labels. [TO DO: EgoVoice once released]
- **Related papers read.** Edwards 2021 and When2Talk were read in full
  on 2026-11-19; Sensible Agent and EgoIntrospect on 2026-11-21.
  Kolbeinsson 2017 is not open access; section 2 now follows its full
  abstract (fine against coarse breakpoints), not the paper body.
  [TO DO: read the Kolbeinsson body through a library login]
  Done 2026-11-19: all 12 arXiv numbers in section 2 match the cited
  titles in arXiv's own records (EgoServe is the benchmark of the Vinci2
  paper, now cited that way). WTaG licence: the GitHub repository
  (sled-group/Watch-Talk-and-Guide) is MIT; the video-only download that
  we used has no separate data licence stated; only the audio-video
  version needs a signed agreement. Plan: release only our labels
  (recording, time, label), no frames. The WTaG contact was asked on
  2026-11-21. [TO DO: record the reply]

## 10. Conclusion

Human instructors on HoloAssist start routine speech during screw,
unscrew and insert actions about one third as often as chance would
give, and they do not hold back a correction. When they do speak during
such an action, the hands stop, but we find no clear sign of more
mistakes. The same pattern appears,
smaller, in a labelled sample of furniture assembly, and not clearly for
precise hand work in cooking, where only the performer's own questions
avoid it. A video model reads the hand state well enough to
improve onset prediction on HoloAssist. But 14 observers did not rate an
assistant better when it held a routine message until the hands were
free, and they rated it worse when it held a warning. The description of
instructor behaviour stands. The step from the description to a rule for
assistants is not supported by our data.
