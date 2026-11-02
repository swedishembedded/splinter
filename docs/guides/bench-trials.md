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
rows the base's vision tower gives it and not as text, exactly as `train` counts it. Where
the base has no readable chat tokenizer, the build says in its report that it measured by
a floor instead, and then counts a picture as one row per 32 x 32 pixels (at most 4096). Brain
refuses an image of more than 36 megapixels, a longer record than `max_block`, or a
held-out record with pictures when it trains; the dataset check at build time reads the
pictures only as inline `data:` URLs.

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
`standardised` (GRPO's) and may be `mean-centred`. The report lists each group's trials with
their reward and advantage; a group of one trial, or whose trials all earned the same, yields
no record (`no_group_signal`). `--from` names the adapter whose policy ran the trials, so the
rollouts are close to on-policy. The exam seal is unchanged: `eval` trials never reach the
dataset. See [Datasets reference](../reference/datasets.md#policy-gradient) and
[Build datasets and train a candidate](training.md#training-on-rollouts).
