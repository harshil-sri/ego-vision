# Labels and tools released with the paper

This folder holds what other researchers need to check our numbers and to
test their own assistant. It holds **labels and times only**: no frames,
video, audio or transcripts of any dataset, and no names of the people
who labelled or rated.

## Files

| File | What it holds |
|---|---|
| `egoextra_hand_labels.csv` | 420 moments of Ego-EXTRA, each labelled S, B or F by two people who did not know the kind of the moment |
| `wtag_hand_labels.csv` | 417 moments of WTaG, labelled the same way |
| `holoassist_hand_state_groups.json` | how the 20 most common HoloAssist verbs map to the four hand states of the paper |

The two label files are written by `python -m scripts.make_release` after
the human relabelling is finished. Until then this folder holds only the
mapping file.

### Columns of the label files

| Column | Meaning |
|---|---|
| `dataset` | Ego-EXTRA or WTaG |
| `recording` | recording id as in the dataset's own release |
| `t_s` | the moment, in seconds from the start of the recording; the labeller saw four frames at t−1.75, t−1.25, t−0.75 and t−0.25 s |
| `kind` | what happens at this second: a speech onset of one kind, or a random second of the same recording (the labeller did not see this) |
| `label_person_1`, `label_person_2` | the two independent labels |
| `label_agreed` | the label when both agree; empty when they differ |
| `label_ai_preliminary` | the earlier one-pass label of the AI assistant, kept for the agreement analysis of the paper |

Labels: **S** screw, unscrew or insert in progress; **B** other precise
hand work in progress (for example pouring, spreading, fitting a tyre);
**F** everything else. The full guide that the labellers saw is
`study/label/` and `scripts/relabel_build.py`.

## Test your own assistant: the timing audit

If your assistant runs on HoloAssist videos, write the times at which it
starts to speak to a CSV (`video,t[,end]`) and run

```
python -m scripts.timing_audit --speech my_assistant.csv --name "My assistant" --out reports/audit_mine
```

It reports, for each hand state, how often your assistant starts to speak
there against chance, next to the human instructors of the same videos,
and the difference with a 95% interval. `scripts/proassist_to_audit_csv.py`
is a worked example for the released outputs of ProAssist; its result is
`reports/timing_audit_proassist.md`. The tool needs the HoloAssist
annotation file and no video. `python -m scripts.timing_audit --self-test`
checks it on the real labels.

## Licences

- **Labels in this folder:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (full text: `LICENSE` in this folder). Use them freely; cite the paper.
- **Code** (`scripts/`, `src/`, `tests/`): MIT (`LICENSE` at the
  repository root).
- The underlying videos belong to their datasets and keep their own terms:
  HoloAssist (CDLA-Permissive-2.0), Ego-EXTRA, WTaG. Get them from their
  authors. Our labels give recording ids and times only.
