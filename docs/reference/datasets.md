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
weighted by the attempt's advantage; see below), `self-distill` (the student's own conversation
of every attempt with a teacher context built from what the verifier knows; see below). `--limit N` keeps at most N records of any
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
The `cpt` view (continued pretraining on the text of sources) is written as
`generic-messages-v2` too: each text is one assistant turn with nothing before
it, supervised whole, so training minimises the next-token loss over all of its
tokens. Objectives brain cannot train (contrastive, reward) need
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
- `--overlong-max-tokens L` adds DAPO's soft over-length penalty to every reward, from the
  completion tokens the attempt recorded (the trial's `tokens.completion` on its verdict, else
  its score's usage, else sven's trajectory total, else the sum over its agent steps): nothing
  up to `L - B` tokens, `(L - B - n) / B` from there to `L`, and `-1` past `L`, with the buffer
  `B` from `--overlong-buffer` (default `L / 4`). An attempt that recorded no count stops the
  build naming it. The manifest records both thresholds as `overlong: {max_tokens, buffer}`.
- A group is the attempts at one task version made by one policy (the model recorded on the
  verdict); each trial counts once however many conversations it had. The records carry
  rewards, and brain computes each attempt's advantage from its group's rewards (brain's
  training-record contract, version 2). `--advantage` names how, and the dataset's manifest
  records it for `train` to ask brain for: `standardised` (the default, GRPO's: the reward
  less the group mean over the group's standard deviation), `mean-centred` (the reward less
  the mean), `leave-one-out` (the reward less the mean of the other attempts) or `positive`
  (only the attempts whose reward is above their group's mean are written, standardised among
  themselves; nothing is pushed down, since a wrong answer shares most of its tokens with a
  right one and pushing it down damages those). A group of one attempt, or whose attempts all
  earned the same, teaches nothing and its experiences are counted as `no_group_signal`; so
  are the attempts `positive` leaves out, and a group with fewer than two kept attempts that
  differ (pass/fail rewards leave `positive` one attempt above the mean, or several alike).
- Each action of each conversation of a written attempt is one record, rendered exactly as
  `sft-step` renders it (the requests its model was sent, `--images`, `--roles`,
  `--only-actions`) with the attempt's reward, its group's key (a digest of the task, its
  version and the policy) and the attempt's name in the record's top-level `reward`, `group`
  and `rollout` fields, the keys brain's policy-gradient trainer reads. Step labels judge
  nothing here.
- The build report and the manifest list every group (task, version, policy) with each
  trial's reward and whether it was written, including groups that yielded no record. The exam seal is
  unchanged: a held-out trial is refused, or left out with `--drop-leaking`. The same
  experiences always give the same bytes.

`train` trains such a dataset by brain's clipped policy gradient; see
[Build datasets and train a candidate](../guides/training.md).

#### Self-distillation

Where every attempt at a task failed the advantage is zero. Two ways to use those groups:
`--view policy-gradient --cliff-distill` distils the groups whose attempts all failed alike (and
keeps the reward of every group with a spread of rewards, so one build holds both kinds of
records), and `--view self-distill` distils every attempt, passed or failed, with no reward. A group is `graded` (its rewards differ), `all-fail` or `all-pass`; the report and
manifest give each group's kind and whether it was distilled.

A distilled record is the same chat record (the attempt's own conversation, nothing privileged in
it) with a top-level `teacher_context`: `{"at": <index>, "messages": [{"role": "user", "content":
...}]}`, the privileged turns inserted before message `at` (right after the first user turn) for
the teacher only. Brain reads it, renders the teacher's conversation with the model's own chat
template and refuses a record whose assistant tokens differ between the two renderings. The text
is built from the experience alone: the task's reference change (privileged kind `Reference`) and
hidden check commands, the commands the deciding verdict records as failed (phase, exit status,
failing and missing tests; the verifier keeps digests of its raw output, never the output), and the
verifier's `verification.reason` that `ingest-bench` copies into the verdict's evidence. An
attempt with none of these is left out as `no_privileged_context`. The exam seal is unchanged.

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

The sft-step view does not teach reading the repository's history. An action whose tool call
walks or shows commits (`git log`, `rev-list`, `reflog`, `show`, `blame`, `cat-file`, `fsck`, a
commit before `HEAD` such as `HEAD~1`) or reads git's storage (`.git/objects`, `.git/refs`,
`.git/logs`, `.git/HEAD`) is, by default, not supervised and stays in the context of what follows
(`repository_history`). `--repository-history truncate` ends the demonstration before it (each
action after it counted as `after_repository_history`); `keep` learns it. The working tree and its
state against `HEAD` (`git status`, `git diff`, `git diff HEAD`) stay readable. The manifest and
the report record the policy as `curation: {"policy": "no-repository-history", "mode": ...}`.
A student trained on a teacher's `git log --oneline -20` learns to run `git log --all` and
`git reflog`, which a benchmark's leak audit fails.

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

## Held-out and monitoring split

`train` holds a tenth of a dataset's records out for scoring and carves its monitoring set from
the rest; the release gate draws the same held-out set from the same records. A dataset built by
`dataset build` decides both by seed and stratum: every record carries `metadata.split_draw`,
the seed of the dataset's split and its stratum - the domain its task exercises, the first part
of a benchmark task's `dimension` (`support` of `support.missing-info-defect-photo`), else the
task's kind. Within a stratum the groups (a benchmark task, a family of source text) are taken in
the order of a hash of the seed and the group, and the strata are interleaved in proportion to
their records, so held-out and monitoring sets hold every stratum at its share wherever the
records sit in the file. Without `--split-seed N` the seed is derived from the experiences the
records come from, so building again from them splits the same; another seed draws another
split. The manifest and the build report record it as `split`: `{"rule": "seeded-stratified",
"seed": N, "strata": {"<stratum>": records, ...}}`. Records written before datasets carried a
draw are held out from the newest end of the file, as they were when they were trained on.

## Manifest

Every dataset is stored under the state root by content address, beside a manifest that
names where each record came from: the experience and task it was projected from, and the
source content under them. The manifest also keeps the seeds the build drew (for example
`--mix-seed`) and the build's controls. A record of the writer's own text (the `voice` view)
names the part it prints and no experience, so a dataset shows what is the writer's and what a
model wrote. The manifest is what lets `splinter lineage` walk from a release back to the
bytes it learned from.

## Checking a stored dataset

`splinter dataset check DATASET-ID --from REF [--max-block TOKENS]` has the trainer read a stored dataset before any
device is claimed: brain's `check_dataset` over every record, rendered for the base `--from` names (the policy's
without it) and in the mode a supervised run reads it (reasoning when the records carry it, one example per record
when the dataset was built `--pack`). It reports the records and examples, the tokens and the supervised tokens per
example (total, min, mean, p50, p90, p99, max), the examples that supervise nothing, the examples longer than
`--max-block`, the longest records by line and every record the trainer refuses with why; it exits 1 when it refuses
any. It asks a brain service started with `--training --artifact-store DIR`, with that directory given as
`--brain-artifact-store`.
