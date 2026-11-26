<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Datasets reference

A dataset is training data projected from attempts, stored with a manifest that names where every
record came from. A recipe makes datasets with the `dataset.*` operations below, a `model.train`
stage reads them, and `dataset export` copies one with its manifest into a directory. The guide to
using them is [Build datasets and train a candidate](../guides/training.md).

## Operations

| Operation | Makes |
|---|---|
| `dataset.project` | Training records projected from the `attempts` of an attempt stage, for one objective. |
| `dataset.text` | A continued-pretraining dataset from the documents under a directory input. |
| `dataset.import` | A dataset from training records made elsewhere. |
| `dataset.extend` | A dataset that admits new records of an existing one only as new solutions of their task. |
| `dataset.rlvr` | The tasks a reinforcement learner can learn from, selected by their measured pass rate. It makes a task collection, not a dataset. |

### dataset.project

`dataset.project` takes the `attempts` of an `attempt.collect` or `answer.collect` stage (and, optionally,
`answers`) and writes a dataset. An `sft` dataset may also take `tasks`, a `tasks.pin` stage's tasks: only
trials of those tasks are admitted, every other trial is excluded as `not_in_tasks` and the waterfall gains an
`in_tasks` step. With a `tasks.pin` `holdout` (see [the tasks reference](tasks.md)) it keeps every trial of the
held-out tasks out of a dataset built from attempts that cover all of them. Its parameters:

- `target` (required): the model role the records are for.
- `seed` (required): the seed of the validation split and of any draw the projection makes.
- `objective`: `sft` (the default), `preference`, `feedback`, `policy-gradient`, `distill` or `retrieval`.
  The objective decides the record format and the training regime a `model.train` stage applies to it.
- `admission` and `projection`: an `sft` dataset's only. `admission: verified-success` (the only value)
  admits a trial that was counted and passed at full credit with no veto; every other trial is excluded as
  `not_a_success:<status>`. `projection: native-sft` (the default) writes the conversation exactly as the
  agent's model was sent it, from the requests it recorded; `portable-sft` rebuilds it from the trajectory
  for an agent whose model calls were not recorded, and is required for attempts that run a harness. Another
  objective is projected by its own view, and the stage is refused if it names another projection or admission.
- `preference.pairs_per_task`: for `preference` only, the most pairs one task of one policy gives, drawn by
  the seed when it has more; every pair when unset.
- `max_block`: the longest record in the target's tokens; a longer one is excluded and listed.
- `thinking`: train the target to reason before it answers; off means no-think mode.
- `validation`: `{group_by, percent}` holds out whole groups for validation, so that no group is in both files.
  `group_by` is `task` (the default) or `task_family`; `percent` is required and has no default, and 100 is
  refused. Without `validation` nothing is held out.
- `supervise`: an `sft` dataset's only: which actions of a record are supervised. `all` (the default)
  supervises every action of the agent. `productive` keeps an action whose tool calls were all answered with
  an error in the record as context (`train: false`), so the target is not taught the failed attempt but still
  sees it; the action that recovers after an error, any action with a call that succeeded, and the final answer
  stay supervised. A record left supervising nothing is excluded as `nothing_supervised`. Whether a result was
  an error is read from the requests the model was sent (the agent marks an error result) against the tool's
  own output in the trajectory, so `productive` needs `projection: native-sft`. The report counts the actions
  kept as context.
- `per_task` and `prefer`: an `sft` dataset's only. `per_task: N` keeps at most N admitted trials of each task
  (after admission, projection and `max_block`); the others are excluded as `per_task_limit`, each with how many
  actions it took. `prefer: fewest-actions` (the default) keeps the trials that took the fewest actions, ties drawn
  by the seed; `prefer: random` draws them by the seed. A trial's draw depends only on the seed and the trial, so
  the same seed keeps the same trials. `prefer` without `per_task`, and `per_task: 0`, are refused.
- `control: shuffled-requests`: an `sft` dataset's only, a negative control. Each record's task request (its
  first user message: the task's instruction with its starting files) is replaced by the request of another
  admitted task, by a derangement of the tasks drawn by the seed: no record keeps its own task's request, every
  record of one task gets the same other task's, and everything else (the system prompt, every action and result,
  the supervision, the task the record is grouped and split by) is the record's own. It needs at least two tasks.
  The manifest says `control: shuffled-requests` and `supervision.jsonl` names the task each record's request is
  from (`request_of`). A dataset made this way keeps the tool-use format and the actions but not the mapping from
  a request to the actions that answer it, so a model that gains as much from it as from the real records gained
  from the format, not from the content.

A stage that projects a non-`sft` dataset is refused when a source attempt stage is hinted, continued or runs
a harness: such a dataset is projected only from attempts made as the target will be asked.

The product holds `train.jsonl` and `validation.jsonl`, `accepted.jsonl` and `excluded.jsonl` (one line per
trial, with the reason for each exclusion), and the manifest. An `sft` dataset also holds `supervision.jsonl` (each record's
trial, split and supervised messages) and `composition.json`, which the manifest names (`composition`) and
`report.md` shows: the records (train and validation), the distinct trials and tasks they are of, the distinct
action sequences (each record's supervised tool calls in order, by tool name and arguments with sorted keys and no
spacing), the supervised and the masked (`train: false`) assistant messages, the characters of the supervised
messages (text, reasoning and each call's name and arguments; tokens are the trainer's to count), the actions
`supervise: productive` kept as context, and the `supervise`, `per_task` and `control` it was built with. Two
arms built from one set of attempts are compared by it before either is trained. Chat records are written as `generic-messages-v2`.
Every record's `metadata.trial` names the pass, slot, agent and model that made its trial, and what the target was
produced with that the record leaves out (`withheld`), so that a hint is never supervised silently.

By objective:

- `sft`: one chat record for each admitted trial, every action of the agent supervised.
- `preference`: pairs of a verified pass preferred over a verified fail of one task, as `generic-preference-v1`
  (the shared prompt, the chosen and the rejected continuation).
- `feedback`: one judged completion for each verified attempt, `desirable` when it passed, as
  `generic-feedback-v1`; the manifest records the class weights a `model.train` stage sends.
- `policy-gradient`: every action of every graded attempt, each record carrying the attempt's `reward`, its
  `group` (the attempts at one task by one policy) and its `rollout`. The trainer computes the advantage from
  the group's rewards.
- `distill`: the attempt's own conversation plus a `teacher_context`, turns that only the teacher sees, built
  from what the verifier recorded. An attempt with nothing of that to show is excluded as
  `no_privileged_context`. See [Build datasets and train a candidate](../guides/training.md).
- `retrieval`: records for a retrieval model.

### dataset.text

`dataset.text` captures every file under its directory input (hidden entries are left out) and makes one
record of each file's text: a single assistant turn with nothing before it, supervised whole, so training
minimises the next-token loss over all of its tokens. A text that two documents share is trained on once
(`duplicate`); a file that is not text or is empty is left out and counted. The dataset's objective is `cpt`.
Parameters: `target` (required), `seed` (required) and `validation: {percent}`, which holds whole documents out.

### dataset.import and dataset.extend

`dataset.import` takes training records made elsewhere (`target`, `max_block`, `thinking`); a longer record is
excluded and listed, and a record that carries reasoning is refused unless `thinking` is on.

`dataset.extend` admits the records of a new dataset into a base dataset only as new solutions of their task.
Parameters: `target` and `holdout_seed` (required), `per_task` (the most records a task has in the result,
default 4), `ignore_call_arguments` (argument names left out when two solutions' tool calls are compared),
`task_prefixes` (prefixes taken off a record's task id so that datasets written in different ways compare
tasks by one key) and `holdout_percent` (the share of new tasks held out as whole tasks, default 10).

### dataset.rlvr

`dataset.rlvr` keeps the tasks a reinforcement learner can learn from, since a task that the policy always
solves or never solves gives each of its groups the same reward and so no gradient. A task is kept when its
counted attempts number at least `min_attempts` (default 2) and its pass rate is above 0 and below 1, or within
`[min_pass_rate, max_pass_rate]` when either is given. Its product holds each kept task's folder and `rlvr.json`,
every attempted task's counts, pass rate and whether it was kept.

## The `dataset` command

`splinter dataset` has four subcommands:

- `dataset conversations GRADED.JSONL --system-prompt TEXT [--advantage KIND]` builds a policy-gradient
  dataset from graded conversations with a counterpart (a client, a student). Each input line names the
  `case`, the `policy` that played the learner, the `trial`, the `reward` the case's grade earned and the
  conversation `record`; only conversations the admission rules accept are used, and the learner's turns are
  the actions. `--system-prompt` is the role prompt the learner played under, which opens every record.
  `--advantage` names how brain reads an attempt's advantage from the rewards of its case, and the manifest
  records it for the training stage.
- `dataset decision-fixtures RUNS-DIR --out FILE` writes labelled decision fixtures from the run records of an
  ablation benchmark output directory, split by task family and seed so that no task is in two splits. The
  output is a JSON-lines file, not a stored dataset.
- `dataset check DATASET-ID`, below.
- `dataset export DATASET-ID --out DIR` copies a dataset and its manifest into a directory.

Run `splinter dataset SUBCOMMAND --help` for every flag.

## Loading a dataset

One interface reads every dataset a run uses, as the Hugging Face `datasets` library does
(`splinter_data::load`):

```rust
let dict = load_dataset::<ChatRecord>("swedishembedded/vim-headless-mastery-sft", None, &options)?;
let train = dict.get("train").unwrap();
let loader = DataLoader::new(train, BatchSampler::new(RandomSampler::new(17), 32, false));
for batch in loader.epoch(0) { /* a Vec<&ChatRecord> */ }
```

`load_dataset(reference, config, options)` gives a `DatasetDict`: the splits of the dataset by name, each a
`Dataset` of typed records with random access (`len`, `get`, `iter`, `select`, `shuffle`). The reference is

- a directory holding `SPLIT.jsonl` files at its root or under `data/` (a configuration is a subdirectory of the
  same places);
- a dataset product a factory run wrote (a directory with `dataset.json`);
- a Hub repository, `owner/name` or `hf:owner/name@revision`, resolved to a commit, downloaded into the cache
  the options name and checked against the sizes and hashes the Hub lists. Every split says the commit it was read
  at, also when a tag or branch was asked for, so a run never trains on a moving name.

`eval`, `val` and `dev` name the `validation` split. The record type is the caller's (`Record`: one JSON line in,
the same line back); `ChatRecord` is the supervised chat record of the trainers, and `dataset.import` and
`dataset.extend` read their records through it. `DataLoader`, `BatchSampler`, `RandomSampler` and
`SequentialSampler` give batches in an order that is a function of the seed and the epoch only, so two processes
that name one seed walk a dataset the same way. A seed is always named.

Brain reads token streams for its trainers (`TokenDataset`); this loader reads the records before they are
tokenized and is the one place splinter reads a dataset.

## Validation split

A `dataset.project` or `dataset.text` stage holds records out only when it names `validation`. Whole groups
(tasks, task families or documents) are held out, never single records, and the groups are drawn by the stage's
`seed`, so the same seed on the same records holds out the same groups. `train.jsonl` and `validation.jsonl`
never share a group. A `model.train` stage evaluates the validation records every `eval_every` steps (see
[the training guide](../guides/training.md)); a dataset with no validation records is scored on its training
records.

## Manifest

Every dataset is stored under the state root by content address, beside a manifest that
names where each record came from: the experience and task it was projected from, and the
source content under them. The manifest also keeps the seeds the build drew and the build's controls. The manifest is what lets `splinter lineage` walk from a release back to the
bytes it learned from.

## Checking a stored dataset

`splinter dataset check DATASET-ID --from REF [--max-block TOKENS]` has the trainer read a stored dataset before any
device is claimed: brain's `check_dataset` over every record, read as chat records, preference pairs or judged
completions as its format is, rendered for the base `--from` names (the policy's
without it) and in the mode a supervised run reads it (reasoning when the records carry it, one example per record
when the dataset was packed). It reports the records and examples, the tokens and the supervised tokens per
example (total, min, mean, p50, p90, p99, max), the examples that supervise nothing, the examples longer than
`--max-block`, the longest records by line and every record the trainer refuses with why; for a preference or feedback dataset also the
pairing policy or class weights its manifest records and the tokenizers its records name. It exits 1 when the trainer
refuses any record or the records name more than one tokenizer. It asks a brain service started with `--training --artifact-store DIR`, with that directory given as
`--brain-artifact-store`.

## Checking recorded speech against its transcripts

`splinter dataset verify-speech METADATA` checks an audio dataset in the Hugging Face audio-folder layout (a
`metadata.jsonl` whose rows name their recording in `file_name`, relative to the file, and its transcript in `text`,
or the column `--text-column` names) before it trains anything. Each recording is sent as it is to the brain
server's transcription route, which reads WAV at any rate and channel count, Ogg/Opus and WebM/Opus, and what the
recogniser (`--recognizer`, default `brain/qwen3asr`, which hears Swedish and English) heard is compared with the
transcript on letters and digits alone: spacing, punctuation, case and bracketed marks such as `[laughter]` do not
count, so neither does a recogniser joining or splitting compounds. Two numbers judge a take:

- the similarity, one minus the character edit distance over the transcript's length: at least `--pass-at` (0.90)
  passes, below `--fail-below` (0.60) fails, and between the two is for review;
- the heard length over the transcript's, which must lie within `--min-length-ratio` and `--max-length-ratio`
  (0.80 to 1.25) whatever the similarity: a take read twice is near 2, one cut short below 0.8.

The verdict is written into each row with its evidence: `asr_model`, `asr_text`, `asr_similarity`,
`asr_length_ratio`, `asr_audio_sha256`, `validation` (`pass`, `review` or `fail`) and `validation_reason`; every other
column keeps its value and its place. A recording whose file and recogniser are those of its last hearing is not heard
again, only judged again against the transcript as it now is (`--rehear` hears every one). A file that is missing or
that the server cannot decode fails with the reason. The command exits 1 unless every recording passes.
