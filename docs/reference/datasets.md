<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Datasets reference

A dataset is training data projected from experience sets by one view, stored with a manifest
that names where every record came from. `dataset build` makes one, `dataset export` copies
it with its manifest into a directory, and `train` reads it. The guide to using them is
[Build datasets and train a candidate](../guides/training.md).

## Views


`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference`,
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`, `voice`
(the writer's own text as answers, from every text part of the sources the
experience sets cite; see [Learn how a person thinks](../guides/personas.md)), `rehearsal` (a model's own answers as it
gave them, every experience with a final answer that no verifier decided
wrong, under the default prompt), `policy-gradient` (every action of every graded attempt,
weighted by the attempt's advantage; see below). `--limit N` keeps at most N records of any
view, evenly spread over the projection, the rest counted as `over_limit`. The
default `--min-strength` is `consistency`; `--strip` defaults to `all`
(the student sees only the instruction - also for a teacher's solve,
whatever it was shown). Every conversation starts with the system turn
every solve, `ask` and probe runs under: one short system prompt of
Splinter's own, in place of sven's coding-agent prompt, so the policy is
trained under the prompt it answers under. The exception is the
`sft-step` view of a repository solve: its records open with the prompt the
solve's own profile records (the coding-agent prompt, `coding-agent-1`, for
a solve made under it), the dataset's manifest records that prompt, and a
model trained on it is asked under it. A solve whose recorded prompt this
build holds no text of is left out; runs under two different prompts do not
make one dataset until `--system-prompt` writes them all under one. Chat views are written as
brain's `generic-messages-v2`, `preference` as brain's
`generic-preference-v1` (one pair per line: the student's prompt, the
chosen and the rejected answer), each checked by brain's own parser.
Objectives brain cannot train (contrastive, reward, raw text) need
`--export-only`.

### Policy gradient

The `policy-gradient` view is reinforcement learning from graded attempts. Where `sft-step`
imitates the actions of verified work, this view keeps the failures too and says how each
attempt did against the others at the same task:

- An attempt is a benchmark trial; its conversations (the roles of a harness) are one
  rollout. The verdict `ingest-bench` records names the trial, the task id and version, the
  model that ran it and the metrics the verifier measured; an experience whose verdict names
  no trial or task is left out as `no_rollout`.
- The reward of a pass is `1.0`. A failure earns `--fail-ceiling` (default `0.5`, always
  below one, so any pass outranks any failure) times the weighted sum of the metrics named
  by repeated `--reward-metric NAME=WEIGHT` (metrics in [0, 1], weights non-negative and
  summing to one); with no `--reward-metric` a failure earns nothing. A failed attempt that
  recorded no such metric, or one outside [0, 1], stops the build naming the trial and the
  metric; it is never read as zero. Splinter's core knows no metric: the names are yours.
- A group is the attempts at one task version made by one policy (the model recorded on the
  verdict); each trial counts once however many conversations it had. The advantage of an
  attempt is `(reward - mean) / (std + 1e-6)` over its group (population standard deviation;
  GRPO's), or with `--advantage mean-centred` the reward less the mean. A group of one
  attempt, or whose attempts all earned the same, teaches nothing and its experiences are
  counted as `no_group_signal`.
- Each action of each conversation of an attempt with an advantage is one record, rendered
  exactly as `sft-step` renders it (the requests its model was sent, `--images`, `--roles`,
  `--only-actions`) with the attempt's advantage in the record's top-level `advantage` field,
  the key brain's policy-gradient trainer reads. Step labels judge nothing here.
- The build report and the manifest list every group (task, version, policy) with each
  trial's reward and advantage, including groups that yielded no record. The exam seal is
  unchanged: a held-out trial is refused, or left out with `--drop-leaking`. The same
  experiences always give the same bytes.

`train` trains such a dataset by brain's clipped policy gradient; see
[Build datasets and train a candidate](../guides/training.md).

The reasoning an agent step recorded (ATIF `reasoning_content`) is the `reasoning_content`
of the assistant message it becomes, trimmed. Brain's chat template renders it as the
turn's think block, in either thinking mode; a step that recorded none carries none, and
reasoning that holds think markup leaves the trajectory out as `reasoning_in_text`.

A workspace trajectory is cleaned of the machine that ran it: the harness's
scratch directory is named relative to itself, the owner and group of a long
listing and the name of a home directory become `user`. An action that repeats
an earlier call of its trajectory with the same result is context and not an
action to learn (counted as `repeated_action`), so the tail of a loop is
shown to what follows and never trained.

A trajectory is projected only when its source survives the projection whole.
Results are paired to calls by id within their step; a step whose calls share
an id is left out and counted under `duplicate_call_id` in the manifest's
`excluded` map. Ids repeat across steps (a model numbers each response's calls
from zero), so a record names its calls `s<step>-<id>`, unique in the
conversation. The projection must keep every call and result of the steps it
renders, else the trajectory is left out as `not_conserved`. Brain's chat
parser then checks every record, with the same pairing rule, before a dataset
is stored, so a dataset with a repeated call id, a call answered twice or a
result before its call cannot be stored.

## Benchmark trials

An experience of a benchmark trial (`experiences ingest-bench`, see
[Benchmark trials as training data](../guides/bench-trials.md)) is training data only
when its task is in the benchmark's `train` split. A build refuses an experience of an
`eval` task, or of a task the register holds as `eval`; `--drop-leaking` leaves it out
and lists it instead.

## Manifest

Every dataset is stored under the state root by content address, beside a manifest that
names where each record came from: the experience and task it was projected from, and the
source content under them. The manifest also keeps the seeds the build drew (for example
`--mix-seed`) and the build's controls. A record of the writer's own text (the `voice` view)
names the part it prints and no experience, so a dataset shows what is the writer's and what a
model wrote. The manifest is what lets `splinter lineage` walk from a release back to the
bytes it learned from.
