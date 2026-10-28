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
