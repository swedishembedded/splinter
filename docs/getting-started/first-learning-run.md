<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Your first learning run

Teach a small model a document it has never seen and train a candidate on what it answered
verifiably; releasing the candidate is a later, optional step. The material is
`examples/first-run/manual.md`, a manual for a
device that does not exist (the Quillon QX-7), so no model can know its console speed, boot jumper,
power limits or watchdog from training. Splinter reads the manual, writes grounded questions about it,
has the model attempt them, and trains an adapter on the verified answers. The recipe ends at the
training stage: it has no evaluation or promotion stage, so nothing is released unless you add them
(step 7). That is the `knowledge-learning` recipe of the
[learning factory](../guides/learning-factory.md), run here. You can then ask the model what the
manual says and trace the answer back to the manual's bytes.

Run every command from the repository root. Nothing here uses the network except fetching the model in step 1.
Steps 1 to 3 load no model; the run in step 4 loads the student and wants a GPU.

Add `--state /tmp/first-run` to every command to keep the walk-through out of your real state.

## What this page does and does not show

Steps 1 and 2 are checked by the test suite against the real program on every build: the output quoted there is
what the command prints. The page ships no expected accuracy numbers and claims no improvement: on this single page
of material a small student and teacher may admit few tasks, and a run reaches a promoted candidate only when a
recipe with evaluation and promotion stages says so. Steps 4 to 8 show what a run writes and how to read it.

## 1. Build and check that a run can start

```bash
make build
export PATH="$PWD/target/release:$PATH"
brain pull Qwen/Qwen3-0.6B
splinter doctor
```

`doctor` loads no model, uses no network and creates nothing under the state root. It prints one line per check:
`state-writable`, `free-space`, `gpu` and one `role:<name>` for each of the roles (`policy`, `teacher`, `generator`,
`planner`, `judge`, `critic`, `router`). On a machine whose model store is empty it fails like this, with the fix under
each failure, and exits 1:

```text
ok   state-writable: ... accepts writes
ok   free-space: ... GiB free under ...
FAIL role:policy: policy:default needs .../Qwen/Qwen3-0.6B, which does not exist
     fix: fetch the base checkpoint, or set the model store (see `splinter status`)
...
not ready: fix the failed checks above
```

Failure modes:

- `FAIL role:...`: the model the role would get is not on disk. Fetch the checkpoint with `brain pull`, or name the
  store with `--models-dir` or the checkpoint with `--policy-base` (see [configuration](configuration.md)).
- `warn role:...`: the policy would play a role that helps it learn (teacher, generator, planner, judge, router).
  The run can start, and rarely trains anything: pull a larger chat model (`brain pull Qwen/Qwen3-8B`), which
  Splinter then picks as the assistant, or name one with `--assistant-model`.
- `warn gpu`: no card has room for the policy's weights right now, or none was found. Brain would run the model on the CPU
  backend without a word, which is too slow for a real run.
- `FAIL free-space`: less than 1 GiB is free under the state root (a warning under 10 GiB).

## 2. Freeze the anchor suite (once)

```bash
splinter eval --freeze examples/first-run/anchor.jsonl
```

`anchor.jsonl` holds twenty general-knowledge questions, held out of every training set: a gate that asks them before
and after training refuses a candidate that got worse at them. This command only freezes the suite; it measures nothing
and loads no model. It prints:

```text
anchor suite version 1 (blake3:7c42b37ca7977684b60a61b466bf3065861735b777b848b47b4f6aa7462e788b), 20 task(s)
run: run-<timestamp>-<id>
```

Freezing the same tasks again is the same version; different tasks are the next version. `splinter eval`, with nothing
after it, shows the suite in force.

## 3. Load the source material

```bash
splinter source add examples/first-run/manual.md --usage-policy research_only
splinter source list
```

`source add` captures the file, splits it into addressable parts and sections, and stores it under a content address. The
same bytes are the same source, so capturing it again stores nothing new. `--usage-policy` records the terms the content
is under; without it the terms are unstated and a release stays restricted to local use.

## 4. Run it as a recipe

```bash
splinter init --template knowledge-learning first-run-recipe
cp examples/first-run/manual.md first-run-recipe/documents/
cd first-run-recipe
splinter validate recipes/knowledge-learning.yaml
splinter plan recipes/knowledge-learning.yaml
splinter run recipes/knowledge-learning.yaml --out runs/knowledge
```

`validate` checks the recipe and everything it names, reporting every problem at once, and runs nothing; `plan` shows the
stages and the work each does. The run's stages: `task.generate` has the teacher propose closed-book tasks from each
document and admits only those whose evidence quotes the document verbatim; `answer.collect` has the student answer each
task several times, every answer graded by the task kind's verifiers; `dataset.project` makes the training records of the
passed answers; `model.train` trains a LoRA on them. Each stage's product is committed under the run directory, so a run
interrupted or rerun repeats no finished stage. The [learning factory](../guides/learning-factory.md) says what a run
needs on the machine (the execution worker, the model connections) and what each operation does.

What to expect of a first run on one page: a weak teacher proposes tasks the admission rules refuse (one that never
names its subject, one that names something the manual does not contain), and nothing that fails them is trained on. Name
a stronger teacher in the recipe's `models:`; a model that brain has to place on the CPU backend (`brain doctor --model
<it>` says so) is too slow to be usable.

## 5. Inspect what the run made

```bash
splinter status runs/knowledge
splinter report runs/knowledge
splinter inspect runs/knowledge --stage generated
splinter verify runs/knowledge
```

`status` shows the run's stages and their slots; `report` prints what each stage reported (the tasks admitted and refused
by reason, the answers graded, the records made, the candidate trained); `inspect` shows a stage's product; `verify` checks
every product against its blobs and its view. [Experience](../concepts/experience.md) and
[Verification](../concepts/verification.md) explain what the verdicts mean.

## 6. The dataset and the trained candidate

The dataset stage's product holds the records and their manifest: chat conversations whose supervised answer was verified,
the student's prompt the instruction alone, never the passage a teacher was shown; the manifest names where each record came
from. See the [datasets reference](../reference/datasets.md). The training stage's product is the candidate: its adapter,
its training curve, the step it carries and any warnings. A trained candidate is not released.

## 7. Evaluate and promote

To release a candidate, add an `evaluation` stage (the candidate against the baseline on held-out tasks, under an
evaluation plan) and a `promotion` stage to the recipe, as the `agent-sft` template does (`splinter init --template
agent-sft`), and run it again: the finished stages are reused. The promotion stage promotes only when every required
purpose's evaluation passed (`target`, `transfer`, `retention`; retention must not regress), and records the evidence in
the release. `splinter eval runs/knowledge --plan PLAN --out DIR` evaluates the run's candidates under another plan,
reusing them.

## 8. Releases

These commands apply once a recipe with a `promotion` stage has released a candidate; the
`knowledge-learning` recipe as shipped makes no release.

```bash
splinter release list
splinter rollback default
```

A release is immutable and carries every number its gate measured; `release list` shows each release with its parent and
the aliases that point at it. The manifest `brain serve` follows is written to `<state>/serving/default.json`, and
`rollback` points the alias back at the release it was trained from. See [Release, serve and roll back](../guides/releases.md)
and [release policy](../concepts/release-policy.md).

## 9. Ask, and explain the evidence

This step assumes a release exists (step 7); the `release` and `lineage` parts need one.

```bash
splinter ask "At what baud rate does the QX-7 console run?"
splinter ask "Which header starts the QX-7 in recovery mode?"
splinter ask "At what baud rate does the QX-7 console run?" --policy policy:default
```

`ask` is closed-book: the answer comes from the model, not from the manual. Compare it with `manual.md`, and with the answer before the release, to see what
changed. Every answer is recorded with the model that gave it and the release the alias resolved to; the report carries the answer's `id`. To see what the
answer rests on:

```bash
splinter lineage $ANSWER --up
splinter lineage $RELEASE --up --depth 3
```

`lineage` walks from the answer to the release, the candidate, the datasets, the experiences and tasks, down to the source part and byte range each task is
grounded in, with those bytes quoted. See [provenance](../concepts/provenance.md). A release can also be served without Splinter, on plain `brain serve`.

## When the steps do not go as described

| You see | Meaning | Do |
|---|---|---|
| `FAIL role:...` from `doctor` | The role's model is not on disk | `brain pull`, or name `--models-dir` or `--policy-base` |
| `warn role:...` from `doctor` | The policy would teach, judge or write tasks for itself | `brain pull` a larger chat model, or name `--assistant-model` |
| Every proposed task rejected | The teacher is too weak for the admission rules | Name a stronger teacher in the recipe's `models:`; see [configuration](configuration.md) |
| The dataset stage made no record | No answer passed its verifiers | More material, or a stronger student or teacher |
| Not promoted | A candidate was trained and its evaluation did not pass | Read the evaluation's report; the policy is unchanged |
| Exit status 2 | Refused before anything ran: usage, an unknown id, a remote model without the opt-in | Read the message |
| Exit status 3 | A sentence was asked back | Name the command, or answer the question |

See [limitations](../concepts/limitations.md) for what this walk-through cannot tell you.
