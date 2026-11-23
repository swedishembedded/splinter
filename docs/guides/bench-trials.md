<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Benchmark trials as training data

> **Recorded material, not benchmark runs.** Splinter never drives `bench run` on a benchmark's collections for
> training data: a benchmark is the sealed measurement authority, asked for verdicts on finished candidates. Running
> rollouts of a recipe over Bench's collections, and hinted repair of failed attempts, are the learning factory's
> (`docs/guides/learning-factory.md`). What this page covers is what Splinter reads of material that already exists:
> an export of a run (`experiences ingest-bench`), the practice material a run recorded (`practice.json`), the
> diagnosis of failed trials and the competence report. See also [Rollouts over a served
> policy](../concepts/learning-loop.md#rollouts-over-a-served-policy).

A benchmark that runs an agent on tasks it grades itself can feed Splinter's
experience, dataset, train and select pipeline. The benchmark exports each run as a
directory of immutable trial artifacts; Splinter admits them as experiences, keeps the
benchmark's own grade as the verdict, and refuses to train on any task the benchmark
holds out for evaluation.

```sh
splinter experiences ingest-bench export-a export-b        # trials as experiences, one set per split
splinter experiences ingest-bench --assume-set train old-export
splinter dataset build --view sft-step TRAIN-SET-ID        # passing train trials as tool-use records
```

An export directory holds `index.jsonl`, one row per trial, and one directory per trial
with `manifest.json`, `trajectory.atif.json` and the files the manifest lists. A trial is
admitted only when the manifest is the one the index names by sha256, agrees with its
index row, and every file in the directory is one the manifest lists with the digest it
lists. A tampered, partial or unlisted file refuses that trial with the reason, the rest
of the export is still admitted, and the command exits 1.

The manifest's score record (`score`, schema `bench.score/1`: how the trial ended, whether it
is counted, the score) must name the manifest's task and digest and, for a counted trial,
the grade's verdict and reward; a record that disagrees refuses the trial. A trial the record
does not count (the sandbox, the model server, the agent or the verifier failed, a budget ran
out) has no verdict, never a fail, so an infrastructure failure never becomes a reward of 0;
the record is kept in the verdict's evidence. A trial marked `confidential` (it ran a released
harness package, whose method is not training material) is refused, and so is a whole export
with `sealed-export.json` beside it (exam material exported under the benchmark's audited
override).

The experience is the trial's ATIF trajectory under a task whose environment records the
benchmark task (id, version, digest), the repository snapshot, the verifier and the split
the task is on. The verdict is the benchmark's grade (`pass` or `fail`), an executable
one. The agent build, the model, the image, the reward and metrics and the artifact's
digests are the verdict's evidence. Ingesting an export again changes nothing; a trial
name that returns with other content is refused.

## Practice material: the examiner's recorded train side

Practice material is a directory the examiner wrote earlier: the staged task folders, `job.yaml` (the Harbor job),
`practice.json` (schema `bench.practice/2`: the collection's manifest and content digests, each staged task's digest,
its capability ids, its naive floor and the draw) and `bench-run.json`. Only the train side is ever written;
`practice.json` of any other side is refused. Splinter reads it and runs nothing from it: `experiences capabilities`
takes a task's capability ids from it, and `dataset build --practice DIR` takes each task's naive floor and lineage
from it for the `shaped-partial` reward. Running rollouts over such material is a recipe's job in the learning
factory.

A run's export (`bench export <job> --out EXPORT`) is ingested as above, and a trial's outcome is read from the
examiner's score record: `passed`, `failed`, `exposed` (the agent reached for the hidden answers), a verifier
failure, `ungraded` (a time limit on a task that does not grade timeouts, a spent budget, an unsupported capability,
a defective task) or an infrastructure failure. Only the first three have a reward; an infrastructure failure is never
a reward of 0 and leaves no experience.

From the trials to a candidate:

```sh
splinter experiences ingest-bench ~/practice/export-support    # a teacher's export
splinter dataset build --view sft-step --pack --max-block 32768 --from local:Qwen/Qwen3-4B TRAIN-SET-ID
splinter dataset check DATASET-ID --from local:Qwen/Qwen3-4B --max-block 32768
splinter --brain-artifact-store /srv/brain-artifacts train DATASET-ID --from local:Qwen/Qwen3-4B --max-block 32768
```

The whole recipe, from a teacher's practice run or a published bundle of its trajectories to an examined student,
is [Distil a teacher's verified trajectories](distillation.md).

`sft-step --pack` is one record per passing trajectory, the system prompt, the instruction and every tool result as
context and every assistant turn (its reasoning, its tool calls, its answer) supervised. `sft-final` does not fit an
agent's trial: the trial's work is its tool calls, and it records no final answer (the build counts every trial as
`NoFinalOutput`). `dataset check` is brain's own reading of the stored dataset (`check_dataset`, every record
listed) for the base `--from` names and in the mode a run would read it (reasoning when the records carry it, one
example per record when the dataset was built packed): the tokens and supervised tokens per example, the examples
that supervise nothing, the longest records and every record the trainer refuses (the command exits 1 when it
refuses any). It needs a brain service started with `--training --artifact-store DIR` and that directory as
`--brain-artifact-store`, as `train` does.

## Why a trial failed

```sh
splinter experiences diagnose ~/practice/export-support         # an export, or a practice run directory
splinter --json experiences diagnose STATE/practice/splinter-0d53...  # report.json and export/ of a practice run
```

Every failed trial gets one failure class from the evidence it left, by deterministic rules in order, strongest
evidence first: the examiner's score status (`bench.score/1`) and its cause, the verifier's own reason
(`verification.json`), whether a worktree change was exported, and the ATIF trajectory. A status the infrastructure
decided is never a model failure: `environment_error` and `agent_error` are `environment`, a failed model request is
`gateway` (HTTP 502 or 504, or the endpoint unreachable) or `model_server` (any other 5xx), `verifier_error` is
`verification`, `invalid_task` is `unclear_task` and `unsupported` is `unsupported`. The examiner's security veto is
`history_probing`, decided at the first call that reads the repository's history. Of the rest, an answer the
verifier read and found wrong is `wrong_content` and one it could not read is `malformed_artifact` (both decided at
the last step that wrote a file); without an answer, a call repeated unchanged three times is `repetition`, a run
whose last calls failed is `tool`, a run a time, turn or token limit ended is `resource`, and a run that stopped is
`no_answer`. Each result names its rule, the decisive step when there is one, and the evidence spans it read (the
step, or the record field, and the excerpt). The report counts the classes per domain and per attribution (`model`,
`infrastructure`, `task`). Only a trial of an export is read, and an export of sealed material is refused.

## Competence per capability

```sh
splinter experiences capabilities STATE/practice/splinter-cb58... STATE/practice/splinter-9e57...
splinter experiences capabilities RUN... --labels ~/practice-relabelled/compiled   # material staged before bench named capabilities
```

Bench labels every task with the capability ids of its taxonomy (`config/capabilities.toml`): the staging of practice material
writes them into `practice.json`, `bench export` into every manifest, index row and `skipped.jsonl` line. Splinter reads
them as data, never a list of its own. A trial's ids come from its export; else from the practice run's own material
(the `practice.json` its `report.json` names, held to the sha256 the run recorded), by the staged task's name; else from
a `--labels` practice directory, only for a task whose `source_digest` is the same (the examiner staging the same draw
again names the capabilities of the same bytes). A trial labelled by none counts as `unlabelled`.

Every trial is diagnosed (above). A pass is a graded pass; a failure the diagnosis blames on the infrastructure or the
task is ungraded and never a `0`; every other failure is a graded fail with its class. Per policy (a practice run's
report names it: `<model>+<adapter>`; `--policy` names it for a bare export) and capability the report gives the tasks,
graded attempts and passes, the pass rate with its 95% Wilson interval, the ungraded attempts, the tasks in each band
(frontier, never solved, retired, unmeasured: the bands of the practice curriculum over these trials, retired at 0.9 on 8 graded
attempts), the failures by class, and the curriculum's weight (`[rollout] curriculum = "capability"`,
`docs/guides/learning-factory.md`). The report is kept in the state's store, by the digest of everything it counted and the
`index.jsonl` and `practice.json` digests it came from, and printed with that address. A sealed export is refused whole.

Each policy's graded attempts are also measured for what they can teach a group-relative policy gradient per token
(`signal`): each task's attempts are one group, rewarded twice, by the strict pass and by `shaped-partial` (a failure
earns `0.5 * max(0, (f - f_naive) / (1 - f_naive))` of its `functional_correctness` above its task's naive floor from the
material, `0` without one; a failure that recorded no `functional_correctness` is left out and counted). For each the
report gives the learning signal per task and per family (the lineage the staged `task.toml` names), the share of
groups whose rewards differ, and `sum Var(r)` per million completion tokens, uniform and as the variance-per-cost draw
(`[rollout] curriculum = "signal"`) would weigh the groups; see the learning signal in
[Datasets](../reference/datasets.md). It is a zero-GPU estimate of what a reward or a selection buys on rollouts that
already exist.

## Repairing a failed attempt

Continuing a failed attempt from its last good step with a teacher, and retrying a failed task with the reference
solution as a hint, are rollout passes of the learning factory (a pass with `hint:`), run over a recipe with
`splinter run`; see `docs/guides/learning-factory.md`. Splinter no longer has the `experiences repair` and
`experiences rationalise` commands that drove Bench for them. The attempts those commands already recorded stay as
experience, and `experiences diagnose` and `experiences capabilities` still read their run directories.

## Moving experience between state roots

```sh
splinter experiences import --from ~/old-state blake3:c2ff3061    # sets by id or unique prefix in that root
```

`experiences import` reads experience sets from another state root and writes them into this one through the
store's own API, whichever layout (loose or packed) either database has: every member experience checked against
its content address and written under the same one, every annotation in the order written, every experience a
relation names (with its annotations), the examiner's trial register for what was imported (so examination material
stays out of every dataset here), and the set under its own id. A write that lands under another address is
refused. Importing again writes nothing. The other root is only read; import from a copy of a root another process
writes.

## Harness trials: one experience per conversation

A benchmark run through a sven harness is several conversations, each answered by a
model role (a surveying `visual` role, a `specifier`, a `designer`, a `reviewer`, ...).
Its top-level `trajectory.atif.json` holds no steps; the record is the
`conversations/NNNN-<name>.json` files, which the manifest lists with their digests like
every other file. A conversation file that is altered, missing, or not listed, or a
conversation the top-level trajectory names that the manifest does not hold, refuses the
trial exactly as any other tampered file does.

`ingest-bench` admits such a trial as **one experience per conversation**. All of them
share the trial's task, split and verdict: the benchmark graded the trial, so its grade
applies to every conversation that contributed. Each experience's trajectory records the
conversation in `extra.splinter_harness`:

- `conversation`: its name;
- `role` and `role_source`: the model role that answered it. `export` when the export
  names it (`extra.harness.role`); `request_log` when an older export does not but
  `model-requests.jsonl` attributes it. That attribution is made only when the requests
  divide exactly among the conversations (one request per agent step, in order) and one
  role answered all of a conversation's; otherwise the role is `unknown` and is never
  guessed;
- `shown_images`: whether its model was shown images, in a user turn or in a tool's
  result (read from the request log; the trajectory need not carry them). When the log
  cannot be divided among the conversations, any of them may have been shown the images
  the log holds, so all are marked.

When the log divides among the conversations and holds a response to every request, the
experience also records the requests its model was sent (`extra.splinter_harness_calls`):
each request exactly as sent, with the response it got. A request repeats the one before
and adds the turns since, so they are stored as the difference to the request before,
and a picture shown once is stored once. A trial ingested before requests were kept
records none (ingest it again into a fresh store to get them).

The command reports the conversations admitted by role. The register, the idempotency
rule and the leakage protection work per conversation: a trial is admitted once, an
`eval` trial's conversations never enter a dataset, and an interrupted ingestion is
completed by the next one. A trial of a single agent is unchanged.

```sh
splinter experiences ingest-bench export-dir
splinter dataset build --view sft-step --roles specifier,designer TRAIN-SET-ID
```

`--roles` (sft-step view) keeps the conversations answered by the roles named, each
role's model learning from its own conversations; every other experience, including a
conversation of unknown role, is counted as `RoleNotWanted`. The build reports both counts
with the other exclusions (and fails, listing them, when nothing is left).

### Training the roles that read images

A conversation whose model was shown images (a `visual` role surveying a drawing, a
`reviewer` reading previews, any role that attached a file) is left out and counted as
`ShownImages` unless the build is told the training base reads them:

```sh
splinter dataset build --view sft-step --images \
    --roles visual,specifier,designer,reviewer TRAIN-SET-ID
```

`--images` says the base is a vision-language one (brain trains Qwen3.8 on records whose
turns carry pictures). Without it a record would teach answers about pictures the student
never saw, so such conversations are never trained on as text alone. With it they are
built from the requests, as below, and `splinter train` validates the dataset against the
base it trains: a base without a vision tower refuses it, naming the record.

**A conversation that recorded its requests is built from them**, whether or not it was
shown images; one that did not is built from its trajectory as before. Each action
(an agent step's tool calls, or the final answer) is one record: the request its model was
sent, as sent - its system prompt, the instruction with the pictures attached to it, the
earlier turns, the tool results with the pictures they returned, the tools it was offered -
and the model's answer as the only supervised turn (its reasoning as `reasoning_content`,
its tool calls in `generic-messages-v2` pairing, as the trajectory-based records have
them). Pictures are `image_url` data-URL parts of the user and tool turns, in place, and
never in a supervised turn. Where the request log is the record, the trajectory's gaps do
not matter: a survey conversation's trajectory holds no user turn, the request holds it.

This replaces the trajectory-based projection for these conversations instead of being a
second mode, because it is the record of what the model was asked and offered. For a
conversation of text, the records are the ones the trajectory gives except in two places,
both corrections: the user turn is the one the model was sent (the trajectory path put the
task's description, `Conversation <name> of benchmark task ...`, where the instruction
stands), and the tools are the ones the request offered (the trajectory path listed every
tool of the harness's agent record, including some the role was never offered).

What is counted instead of dropped:

- `RequestsNotRecorded`: the conversation was shown images but recorded no requests (it
  was ingested before they were kept, or the log did not divide among the conversations).
- `TooManyImages`: an action whose request holds more than 8 images, the most brain reads in
  one record. Later actions of a conversation that accumulates pictures are counted too.
- `Compacted`: the conversation's requests do not extend one another (its context was
  rewritten between two of them), so no single list of messages holds what the model saw.

`--pack` and `--max-block` apply as for text: `--max-block` counts each picture as the
rows the base's vision configuration gives it and not as text, exactly as `train` counts it, so
name the base you train with `--from` (as `train --from` does; without it the policy is
counted, whose tokenizer and image handling may be another model's, and the report says which
base counted). Where
the base has no readable chat tokenizer, the build says in its report that it measured by
a floor instead, and then counts a picture as one row per 32 x 32 pixels (at most 4096). Brain
refuses an image of more than 36 megapixels, a longer record than `max_block`, or a
held-out record with pictures when it trains; the dataset check at build time reads the
pictures only as inline `data:` URLs.

A trial artifact carries the verifier's statement of why it decided as it did
(`verification.json`'s `reason`); ingestion copies it into the verdict's evidence, where
the `self-distill` view reads it as privileged context for a teacher. The artifact holds
no reference change and no test output: a task's reference change reaches a dataset only
through the task record (a privileged `Reference` item), so attempts of a benchmark task
whose export names none are distilled from the verifier's reason alone.

## Splits

The command writes two experience sets, `bench train trials` and `bench eval trials`,
and prints both. Only the first can be built into a dataset:

- A task on the `eval` side is examination material. Every dataset build, with or
  without `--drop-leaking`, refuses an experience of such a task (`--drop-leaking` leaves
  it out and lists it), and so does the last step that stores a dataset.
- The split is part of the task, hence of the experience's address, and the register of
  admitted trials also refuses a task that was admitted on both sides: a task registered
  as `eval` is never training data, whatever a later experience of it claims.
- An export that states no split (made before the benchmark recorded them) is refused.
  `--assume-set train|eval` states it for such trials; the register records that the
  operator did, and an export that does state its split cannot be overruled.

## The dataset

`sft-step` fits a multi-turn agent trajectory: one record per action (a step's tool
calls, then the final answer) with the conversation up to it as context and only the
action supervised, or one record per trajectory with `--pack`. A trial the benchmark
failed yields nothing. The view writes tool calls and their results in
`generic-messages-v2` pairing. The reasoning a step recorded rides on its assistant
message as `reasoning_content`, the field brain's chat template renders as the turn's
think block (the one serving sends back), so a trained turn shows its reasoning where the
model produces it; a step that recorded none has none. A benchmark trajectory of this agent records neither its system prompt nor its tool
schemas, but the trial's first model request (`model-requests.jsonl`, digest-checked like
every file) holds both. Ingestion adds them to the experience's trajectory, the system
prompt as a leading system step and the schemas as `agent.tool_definitions`, and records
the derivation in the trajectory's `extra.splinter_enrichment` (source file digest, the
digest of the trajectory as exported, what was added); the exported artifact stays the
source of truth. A trial without that file is refused. The records then open with the
system prompt the trial's agent was served (it differs between trials only in the
repository commit it names, so a dataset holds each record under its own) and carry the
tools it offered. A tool the agent loaded later in a session is not in the first
request's list. When the log holds one answered request per agent step and none of them
showed a picture, ingestion also files the requests themselves with the experience (as a
harness conversation's are), and every view reads the conversation from them: the
instruction as the harness sent it (its time line included) and the history as the model
was given it, so a policy-gradient record's prompt ids are those of its context. Requests
that do not read as one conversation (the agent rewrote its context between two of them)
leave the trial the record its trajectory makes, which a policy-gradient build counts as
`context_reconstructed`. An export that declares its log (`requests.schema` in the manifest)
in a schema other than `sven.request-log/1` is refused.

## Reinforcement learning from the trials

Several trials of one task by one policy are a group, and the `policy-gradient` view
turns the group into training records whose weight is how much better each attempt did than
the others - failures included, which `sft-step` discards:

```
splinter dataset build --view policy-gradient \
    --reward-metric iou=0.5 --reward-metric feature_match=0.25 --reward-metric deviation_score=0.25 \
    --fail-ceiling 0.5 --advantage standardised --roles designer --images \
    --max-block 8192 --drop-leaking TRAIN-SET-ID
splinter train DATASET-ID --from local:/models/qwen3.8-27b+round-3.safetensors \
    --clip-eps 0.2 --kl-beta 0.02 --reference start --base-precision bf16
```

A pass earns 1; a failure earns at most `--fail-ceiling` by the metrics the verifier measured,
so a close miss ranks above a far one and any pass above both. `--advantage` defaults to
`standardised` (GRPO's) and may be `mean-centred`, `leave-one-out` or `positive`; brain computes
the advantages from the rewards the records carry. The report lists each group's trials with
their reward and whether each was written; a group of one trial, or whose trials all earned the same, yields
no record (`no_group_signal`). `--from` names the adapter whose policy ran the trials, so the
rollouts are close to on-policy. Add `--cliff-distill` to give the groups whose attempts all
failed (where the advantage is zero) dense per-token credit from a teacher that is shown what
the verifier knows, see [Self-distillation on the cliff](training.md#self-distillation-on-the-cliff).
The exam seal is unchanged: `eval` trials never reach the
dataset. See [Datasets reference](../reference/datasets.md#policy-gradient) and
[Build datasets and train a candidate](training.md#training-on-rollouts).
