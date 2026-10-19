<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Measure what a candidate learned

Use this guide to grade a model or a candidate yourself, outside the release gate, and to
build an exam whose result you can trust. The gate's own checks, thresholds and report
fields are in the [evaluation reference](../reference/evaluation.md); the reasoning behind
them is in [release policy](../concepts/release-policy.md).

Three tools, in order of strength:

- `splinter eval` grades one model closed-book on a suite and names every task it got
  wrong, so two evaluations show what a candidate lost against the base.
- `splinter exam` puts a candidate and the model it continues to held-out tasks and has a
  calibrated judge compare their answers.
- `splinter exam-set create` freezes an exam before any training, from sources reserved
  apart from everything a candidate saw.

```bash
splinter eval policy:default --suite anchor
splinter eval 3f9a1c20 --suite retention
splinter exam 3f9a1c20 --prompt "the QX-7 console, boot, power and watchdog"
splinter exam-set create ./letters --families 50
splinter exam-set power --families 50 --tasks-per-family 8 --effect 0.1
```

## Grade a model on a suite


`splinter eval REF --suite SUITE` grades the model REF (a candidate id or a model reference)
closed-book on `held-out` (the default), `retention`, `anchor` or a file of tasks, and
prints each task it got wrong with the answer it gave. Run it on the base and on a candidate
and compare the two lists. With no `REF` it shows the anchor suite in force, and
`--freeze FILE` makes a file the next version of the suite. The suites and the file format are
in the [evaluation reference](../reference/evaluation.md#suites).

## The exam

The exam (`splinter exam CANDIDATE`, and the `exam` stage of `learn`) measures
what the release gate's code-only grading cannot: whether a candidate gives, in
its own words, what a held-out task's reference says. It puts the same held-out
tasks to what the candidate continues (its parent release, else the base) and
to the candidate, closed-book. With `--prompt GOAL` (the `exam` stage of `learn`
passes its goal) the base is asked once more under the goal as one added system
line, and the candidate is also compared with that: training is worth what it
adds beyond telling the base what the goal is. With `--retrieve SOURCE-ID...` the candidate is
also asked with the passages of those sources that bear on each task shown
before it (as `ask --retrieve` does), compared with the candidate alone, and the
report says for how many tasks a retrieved passage overlaps the evidence the
task was written from: what retrieval found, apart from what the model made
of it. The sources hold the held-out letters too, so this measures access to
what the weights never saw, not recall of it. A judge compares each answer
with the reference, and code separately checks whether the answer states a
number, name or quotation the task's source does not hold. Before the judge
grades an arm it is calibrated on controls made from the tasks' own references
- each task's reference answering it, and another's answering it - which no
model wrote; a judge that cannot tell them apart grades nothing and the report
says it makes no claim. A task an arm gave no answer to counts as not done and
is reported as unanswered. The result is a paired sign test of the judged
results, in which tasks about one family of source text (two prints of a
letter, several questions about one document) count once, as the release gate's
improvement check counts them: a model that knows the letter gets all of its
questions right, so they are one piece of evidence; beside it, a bootstrap
interval of the candidate's gain per family (won 1, lost -1, tied 0), which is
what a result on a few families has to be read by. It never reaches a
training set. Each control the judge did not
judge as labelled is reported with the answer and the judge's reason, so a judge
that is not trusted can be seen failing.

## The powered exam

An exam of a few families cannot show a gain: the paired
tests above rest on as many units of evidence as families, which is a handful.
`learn --exam-families N` (fifty by default for a run with a persona, none
otherwise; 0 reserves none; `--exam-tasks-per-family`, eight; and `--dev-families` and
`--dev-tasks-per-family`, fifty of four, a dev suite on its own families for choosing a
checkpoint) reserves the exam's families before anything is
generated or built, after the plan. A family is a group of text parts that
print the same text (the groups a split uses, named by the least content digest
in them); a stable hash of the name chooses N of the examinable ones - long
enough to write tasks from, not so long that reserving them costs the run its
corpus, in a directory source, and not text the policy's own lineage was trained
on - and the run is refused, saying why, when the sources cannot spare that many
or reserving them would leave less to learn from than is examined. The text of a
reserved family in every edition is left out of the sources the run reads from
then on (the source records the parts as skipped, reserved), so no task, dataset
or passage index can hold it, and the exam is written from a source of the
reserved parts alone, with `--exam-tasks-per-family` tasks for each family, and
frozen under `<state>/exams/<id>/` with a ledger that refuses a changed manifest;
the manifest carries the tasks and the text they were written from, so loading it in
another state root installs the same exam there.
`splinter exam-set create` makes such an exam from sources apart from any
`learn`, clear of what the candidates named by `--not-trained-by` (and the
releases they continue) were trained on; `splinter exam CANDIDATE --exam-set`
puts a candidate to it. Four arms answer each task - the base, the base under the
prompt the candidate was trained under (the goal when it has none), the
candidate under the default prompt and the candidate under its own prompt - the
first answer greedy and the rest sampled (`--resamples`, three by default); a
family the candidate or a release it continues was trained on is left out. A
judge calibrated on controls from the tasks' references, and on harder wrong
answers (each task answered with the most similar reference of another family),
decodes greedily and judges each answer alone, shown only the task, the reference
and the answer: it cannot know which arm gave it or what was judged before. The
report keeps every answer and verdict of every arm so the analysis can be redone,
and for each pair of arms gives the per-task counts, the exact sign test over
tasks and over families, the difference in the share right and in the pass rate
over the samples with a family-clustered bootstrap interval, the same test over
answers of like length, the answers' lengths, the invented specifics, the number
of discordant tasks and the smallest difference the exam's own spread over
families could have shown at 80% power. The voice of each arm is also scored with
no judge, by the likelihood it gives the writer's own text of the reserved
families (brain's held-out scoring path). The `exam` stage of a `learn` that
reserved its exam runs this exam.

`splinter exam-set power` sizes an exam before it is paid for: it simulates the primary
comparison's test (the family-clustered interval of the difference above zero) under an assumed
effect, discordance and intraclass correlation, or under the discordance and clustering the
report of a pilot (`exam --pilot-families N`) estimated, and says how often the test finds the
effect and how often it finds one that is not there.

The judge is measured against a person, who has to supply the labels:
`exam-report labels-export` writes a sample of the exam's answers (400 by default)
stratified by arm, by what the judge said and by length, shuffled and blind to the arm, each
with the task and its reference passage and an empty `label`, and a key that keeps the arm and
the judge's verdict; the person fills each `label` with `right` or `wrong`, and `labels-import`
reports the agreement, Cohen's kappa, the share of what the judge called right that the person
called wrong and the reverse, agreement on short and on long answers and the length bias
(the judge grades each answer alone, so there is no position to favour). Nothing here makes up
a label. `exam-report memorisation` measures how much of each arm's answers is the training
text again - the share of 8, 13 and 20 word runs found in what the candidate was trained to
produce, the longest run, the answers quoting twenty words or more - so voice can be told from
memory; nearest-passage similarity and an authorship embedding are not measured.

## Choose how to improve

`splinter improve EXPERIMENT.toml...` measures each intervention an experiment names (more
samples, another episode protocol, a candidate that `practice` or `train` kept, no change)
against the incumbent on the same target tasks and chooses the one with the greatest net value:
the lower confidence bound of the paired gain per family, times what a success is worth and how
often it is used, less its one-time and per-use cost, or no change when none pays. Nothing is
applied. An intervention whose mechanism is not there yet is reported as not measured.
`--compare-fixed` shows what applying one intervention to every experiment would have come to.

`make qualify/list` shows the qualification matrix and the mandatory ablations
`scripts/qualify/qualify.py` runs as one recorded command.
