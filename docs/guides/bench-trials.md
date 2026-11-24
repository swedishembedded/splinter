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
experience store. The benchmark exports each run as a directory of immutable trial
artifacts; Splinter admits them as experiences, keeps the benchmark's own grade as the
verdict, and refuses any task the benchmark holds out for evaluation.

```sh
splinter experiences ingest-bench export-a export-b        # trials as experiences, one set per split
splinter experiences ingest-bench --assume-set train old-export
```

Training data is made by a recipe of the [learning factory](learning-factory.md): a
`dataset.project` stage over attempts, which a Harbor job of the benchmark joins through
`attempt.import`.

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
takes a task's capability ids from it, and its naive floor and lineage feed the `shaped-partial` reward the
competence report below measures. Running rollouts over such material is a recipe's job in the learning
factory.

A run's export (`bench export <job> --out EXPORT`) is ingested as above, and a trial's outcome is read from the
examiner's score record: `passed`, `failed`, `exposed` (the agent reached for the hidden answers), a verifier
failure, `ungraded` (a time limit on a task that does not grade timeouts, a spent budget, an unsupported capability,
a defective task) or an infrastructure failure. Only the first three have a reward; an infrastructure failure is never
a reward of 0 and leaves no experience.

From the trials to a candidate, a recipe stage projects the attempts to a dataset (`dataset.project`, with
`projection: native-sft`) and trains it (`model.train`); the whole recipe, from a teacher's practice run or a
published bundle of its trajectories to an examined student, is [Distil a teacher's verified
trajectories](distillation.md). A projection makes one record per verified success: the system prompt, the
instruction and every tool result as context, and every assistant turn (its reasoning, its tool calls, its answer)
supervised. Final-answer records do not fit an agent's trial: the trial's work is its tool calls, and it records no
final answer (the build counts every trial as `NoFinalOutput`). Once a run has made a dataset, `dataset check` is
brain's own reading of it (`check_dataset`, every record listed) for a base and in the mode a run would read it:

```sh
splinter --brain-artifact-store /srv/brain-artifacts dataset check DATASET-ID --from local:Qwen/Qwen3-4B --max-block 32768
```

It reports the tokens and supervised tokens per example, the examples that supervise nothing, the longest records
and every record the trainer refuses (the command exits 1 when it refuses any). It needs a brain service started with
`--training --artifact-store DIR` and that directory as `--brain-artifact-store`.

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
```

The conversations an admitted trial recorded, with the requests their models were sent, stay with their
experiences. Projecting them to training records is a recipe stage of the [learning
factory](learning-factory.md), not a command of its own.

A trial artifact carries the verifier's statement of why it decided as it did
(`verification.json`'s `reason`); ingestion copies it into the verdict's evidence, where
an `objective: distill` projection reads it as privileged context for a teacher. The artifact holds
no reference change and no test output: a task's reference change reaches a dataset only
through the task record (a privileged `Reference` item), so attempts of a benchmark task
whose export names none are distilled from the verifier's reason alone.

## Splits

The command writes two experience sets, `bench train trials` and `bench eval trials`,
and prints both. Only the first can be built into a dataset:

- A task on the `eval` side is examination material. A dataset never holds an experience
  of such a task.
- The split is part of the task, hence of the experience's address, and the register of
  admitted trials also refuses a task that was admitted on both sides: a task registered
  as `eval` is never training data, whatever a later experience of it claims.
- An export that states no split (made before the benchmark recorded them) is refused.
  `--assume-set train|eval` states it for such trials; the register records that the
  operator did, and an export that does state its split cannot be overruled.

## What ingestion adds to a trial

A benchmark trajectory of this agent records neither its system prompt nor its tool schemas, but the trial's first
model request (`model-requests.jsonl`, digest-checked like every file) holds both. Ingestion adds them to the
experience's trajectory, the system prompt as a leading system step and the schemas as `agent.tool_definitions`, and
records the derivation in the trajectory's `extra.splinter_enrichment` (source file digest, the digest of the
trajectory as exported, what was added); the exported artifact stays the source of truth. A trial without that file is
refused. A tool the agent loaded later in a session is not in the first request's list. When the log holds one
answered request per agent step and none of them showed a picture, ingestion also files the requests themselves with
the experience (as a harness conversation's are), so the conversation can be read as the model was given it. An
export that declares its log (`requests.schema` in the manifest) in a schema other than `sven.request-log/1` is
refused.

## Datasets and reinforcement learning

Records for supervised fine-tuning, preference pairs and policy-gradient groups are projected from attempts by a
`dataset.project` stage, and a `model.train` stage trains them; the stages, their parameters and what each projection
leaves out are in the [learning factory](learning-factory.md#what-each-stage-guarantees). See also the [datasets
reference](../reference/datasets.md#policy-gradient).
