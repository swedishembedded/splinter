<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Benchmark trials as training data

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

The experience is the trial's ATIF trajectory under a task whose environment records the
benchmark task (id, version, digest), the repository snapshot, the verifier and the split
the task is on. The verdict is the benchmark's grade (`pass` or `fail`), an executable
one. The agent build, the model, the image, the reward and metrics and the artifact's
digests are the verdict's evidence. Ingesting an export again changes nothing; a trial
name that returns with other content is refused.

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
- `shown_images`: whether its model was shown images (read from the request log; the
  trajectory need not carry them). When the log cannot be divided among the
  conversations, any of them may have been shown the images the log holds, so all are
  marked.

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
conversation of unknown role, is counted as `RoleNotWanted`. A conversation whose model
was shown images is left out and counted as `ShownImages`, whatever the roles asked for:
brain's LoRA path trains text only, and a record of such a conversation would teach
answers about pictures the student never saw. The build reports both counts with the
other exclusions (and fails, listing them, when nothing is left).

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
