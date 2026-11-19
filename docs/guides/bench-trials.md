<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Benchmark trials as training data

> **Practice material, not benchmark runs.** A learning campaign never drives `bench run` on a benchmark's
> collections for training data: a benchmark is the sealed measurement authority, asked for verdicts on finished
> candidates. What the examiner publishes *for* a learner is its practice material, the train side of a collection
> staged as a Harbor job (`bench task practice`); [Practice material](#practice-material-the-examiners-staged-train-side)
> below runs it under a policy and admits the graded trials through the same intake. The other source of training
> episodes is sven's training executor (see [A continuous campaign, and where its rollouts come
> from](../concepts/learning-loop.md#a-continuous-campaign-and-where-its-rollouts-come-from)). Ingesting a benchmark
> run's export stays for the recorded arms and for exports that cannot be regenerated.

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

## Practice material: the examiner's staged train side

`bench task practice <collection> [--count N --seed S] --environment userns --profile P --out DIR` writes
`DIR/<collection>/`: the staged task folders, `job.yaml` (the Harbor job), `practice.json` (schema
`bench.practice/2`: the collection's manifest and content digests, each staged task's digest, and the draw) and
`bench-run.json`. Only the train side is ever written; `practice.json` of any other side is refused.

A teacher's run of it, exported by the examiner (`bench export <job> --out EXPORT`), is ingested as above. To run
the material under a policy brain serves:

```sh
splinter --brain-endpoint http://127.0.0.1:8811/v1 --brain-key-file brain.key \
    tasks harbor practice ~/practice/support --policy local:Qwen/Qwen3-4B \
    --bench ~/code/bench --sven-build ~/sven-build/sven --attempts 2 --jobs 16
```

The examiner owns everything that runs inside a trial, so the policy is named to it the way it documents: a
profile. The command derives one from the profile the material was staged under (`bench-run.json` names it and its
sha256; `--profile` names another), changing only the main role's upstream to the endpoint and the id brain lists
the policy under, and starts the examiner's gateway for it (`bench gateway serve`, with the endpoint's key in the
gateway's environment only). It then stages the same material again under that profile into the run's own directory
under the state root (`bench task practice`, which writes the job config, registers the job with the gateway and
records the run), and refuses to go on unless every staged task's digest, the draw and the collection's digests are
the ones the given `practice.json` names. Harbor runs the job (`harbor run -c`, the checkout's own Harbor unless
`--harbor` names another) with the attempts, concurrency and tasks (`--task`) of the command, the gateway is
stopped, the examiner exports the graded trials and they are admitted exactly as `experiences ingest-bench` admits
them. The material is never edited.

Every trial is reported with a typed outcome read from the examiner's score record: `passed`, `failed`, `exposed`
(the agent reached for the hidden answers), a verifier failure, `ungraded` (a time limit on a task that does not
grade timeouts, a spent budget, an unsupported capability, a defective task) or an infrastructure failure (the
sandbox, the model server or the agent failed). Only the first three have a reward; an infrastructure failure is
never a reward of 0 and leaves no experience. The report keeps the material's provenance (the `practice.json`
sha256, the collection digests, the draw, the examiner's commit, the profile and its sha256), the policy (its
identity and adapter digest, and the id it was served under) and the seed; it is also written as `report.json` in
the run's directory.

A run is keyed by the material, the policy, the profile, the attempts, the tasks and the seed. Running it again
after it finished runs nothing, exports again (the export is idempotent) and admits nothing twice. A run that did
not finish is staged and run again whole: the examiner's gateway forgets the jobs registered with it when it stops,
and only a staging registers one. The seed names the run. How the policy samples is the profile's: its `[sven.sampling]` (the report records it as
`sampling`, each trial seeded with its own seed derived from the profile's run seed by the examiner), or, without one, the
policy samples as its server does. Material staged for docker sandboxes is refused: the gateway is served on the
loopback address, which userns sandboxes reach.

From the trials to a candidate:

```sh
splinter experiences ingest-bench ~/practice/export-support    # a teacher's export (the command above admits its own)
splinter dataset build --view sft-step --pack --max-block 32768 --from local:Qwen/Qwen3-4B TRAIN-SET-ID
splinter dataset check DATASET-ID --from local:Qwen/Qwen3-4B --max-block 32768
splinter --brain-artifact-store /srv/brain-artifacts train DATASET-ID --from local:Qwen/Qwen3-4B --max-block 32768
```

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

Bench labels every task with the capability ids of its taxonomy (`config/capabilities.toml`): `bench task practice`
writes them into `practice.json`, `bench export` into every manifest, index row and `skipped.jsonl` line. Splinter reads
them as data, never a list of its own. A trial's ids come from its export; else from the practice run's own material
(the `practice.json` its `report.json` names, held to the sha256 the run recorded), by the staged task's name; else from
a `--labels` practice directory, only for a task whose `source_digest` is the same (the examiner staging the same draw
again names the capabilities of the same bytes). A trial labelled by none counts as `unlabelled`.

Every trial is diagnosed (above). A pass is a graded pass; a failure the diagnosis blames on the infrastructure or the
task is ungraded and never a `0`; every other failure is a graded fail with its class. Per policy (a practice run's
report names it: `<model>+<adapter>`; `--policy` names it for a bare export) and capability the report gives the tasks,
graded attempts and passes, the pass rate with its 95% Wilson interval, the ungraded attempts, the tasks in each band
(frontier, never solved, retired, unmeasured: the campaign's own bands over these trials, retired at 0.9 on 8 graded
attempts), the failures by class, and the curriculum's weight (`[rollout] curriculum = "capability"`,
`docs/guides/campaigns.md`). The report is kept in the state's store, by the digest of everything it counted and the
`index.jsonl` and `practice.json` digests it came from, and printed with that address. A sealed export is refused whole.

## Repairing a failed attempt

```sh
splinter experiences repair STATE/practice/splinter-0d53... --teacher local:Qwen/Qwen3.8-27B-FP8 --bench ~/bench
```

Every failed trial of the practice runs is diagnosed (above) and routed by its class: a model failure whose decisive
step the diagnosis placed is forked after its last good step; an infrastructure or task failure, a history probe
and a failure with no decisive step are not repaired, and the report says why. The fork opens the examiner's sandbox
session for the task (`bench task session`: the pristine workspace bound where a trial's agent works, the trial's
account and image, the task's service when it has one), keeps the student's steps through the last good one, has
sven's fork replay them (every replayed observation held to the recorded one; a fork whose replay differs is refused
and counted) and the teacher continue the same tool loop under the profile the student ran; every shell command runs
in the session (`bench task session-exec`, sven's shell launcher), so it sees what the student's saw. The session
then grades the state the commands left with the task's own verifier, outside any agent. Only a pass is kept: as an experience of the student's task whose first steps are
the student's stored steps (marked as copied context, never supervised), with the verifier's pass and a
`revision_of` relation to the student's attempt, and its lineage in `extra.correction` (the
student, the diagnosis, the kept step, the shared prefix, the teacher, the grade). The accepted repairs form one
experience set, which `dataset build --view sft-step` trains on the teacher's suffix alone. A fail is counted and
recorded, never stored. A repair is keyed by the student's attempt, the teacher and the intervention: a rerun finds
the record and does nothing. `--jobs` repairs run at once (the teacher's decode slots), `--max` bounds a run and
`--class` picks the classes. `--per-task N` spreads a run across task families: the tasks are taken in turn, in an
order `--seed` fixes (drawn and printed when not given), at most N repairs of a task, the ones an earlier run
already graded counted. `--samples K` has the teacher continue each selected failure K times, each a repair of
its own (keyed by its sample index), since one sample of a hard task is often wrong.

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
request's list.

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
