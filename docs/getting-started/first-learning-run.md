<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Your first learning run

Teach a small model a document it has never seen, measure that it learned it, and release the
result only if the evidence says so. The material is `examples/first-run/manual.md`, a manual for a
device that does not exist (the Quillon QX-7), so no model can know its console speed, boot jumper,
power limits or watchdog from training. Splinter reads the manual, writes grounded questions about it,
has the model attempt them, trains an adapter on the verified answers, and releases it only if the
release gate passes. You can then ask the model what the manual says and trace the answer back to the
manual's bytes.

Run every command from the repository root. Nothing here uses the network except fetching the model in step 1.
Steps 2, 4 and 5 load no model; steps 3 and 6 onward load the policy model and want a GPU.

Add `--state /tmp/first-run` to every command to keep the walk-through out of your real state.

## What this page does and does not show

Steps 1, 2 and 5 are checked by the test suite against the real program on every build: the output quoted there is
what the command prints. The observations in step 6 were made on a GPU, one run per option, so they are a pilot and
not a measurement. On this single page of material no run reached a trained candidate. The page ships no expected
accuracy numbers and claims no improvement. Steps 3 and 7 to 11 show the program's report formats and what to do when
a run does produce a candidate (for example from several documents and a stronger teacher); those shapes were not
recorded from a run on this material.

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
  store with `BRAIN_MODELS_DIR` or the checkpoint with `BRAIN_QWEN_WEIGHTS` (see [configuration](configuration.md)).
- `warn gpu`: no card has room for the policy's weights right now, or none was found. Brain would run the model on the CPU
  backend without a word, which is too slow for a real run.
- `FAIL free-space`: less than 1 GiB is free under the state root (a warning under 10 GiB).

## 2. Freeze the anchor suite (once)

```bash
splinter eval --suite anchor --freeze examples/first-run/anchor.jsonl
```

`anchor.jsonl` holds twenty general-knowledge questions. The release gate asks them before and after training and refuses a
candidate that got worse at them. This command only freezes the suite; it measures nothing and loads no model. It prints:

```text
anchor suite version 1 (blake3:7c42b37ca7977684b60a61b466bf3065861735b777b848b47b4f6aa7462e788b), 20 task(s)
run: run-<timestamp>-<id>
```

Freezing the same tasks again is the same version; different tasks are the next version. `splinter eval --suite anchor`, with no
model, shows the suite in force.

## 3. Measure the baseline

```bash
splinter eval policy:default --suite anchor
```

This asks the policy model the twenty anchor questions closed-book and grades each against its reference by code. It loads the
model, so it needs the checkpoint from step 1. The report names the model, then the suite with how many tasks were graded and the
accuracy, then each task it missed with the answer it gave:

```text
<model> (<reference>):
  anchor: <graded> of 20 task(s) graded, accuracy <share>
    missed (<kind>): <instruction> -> <answer>
```

Keep this output: it is what the base model does before it has learned anything, and the later comparison is against it. Answers a code
check cannot decide are reported as undecided and kept in the report.

## 4. Load the source material

```bash
splinter source add examples/first-run/manual.md --usage-policy research_only
splinter source list
```

`source add` captures the file, splits it into addressable parts and sections, and stores it under a content address. The same bytes are
the same source, so capturing it again stores nothing new, and `learn` in step 6 reuses it. `--usage-policy` records the terms the content
is under; without it the terms are unstated and a release stays restricted to local use.

## 5. See the plan without running it

```bash
splinter learn examples/first-run/manual.md --goal "the QX-7 console, boot, power and watchdog" --kinds denoise --budget 40m --dry-run
```

It prints the sources, the models that play each role and the stages, and writes nothing under the state directory. The stage line is
the pipeline a `learn` runs:

```text
  stages:  policy -> sources -> tasks -> solve -> verify -> teach -> frontier -> variants -> critique -> select -> dataset -> train -> exam -> release
```

## 6. Learn

```bash
splinter learn examples/first-run/manual.md --goal "the QX-7 console, boot, power and watchdog" --kinds denoise --budget 40m
```

This is a real run on a GPU. It prints each stage as it finishes and opens its report with an `outcome` block that keeps five things apart: whether
the run finished, whether a candidate was trained, what the release gate decided, whether a release was written, and whether `policy:default`
moved. For a run that trained nothing it reads:

```text
outcome  run:      ran every stage it was asked to
         training: no candidate was trained
         gate:     not run
         release:  none
         policy:   policy:default unchanged
```

A run can train a candidate that the gate then blocks; that is a result, not a failure of the tool, and the policy stays as it was
(`gate: blocked on improvement, ...`). The exit status is 0 only when a release was written.

What the options did, as observed in one run each:

- **Without `--kinds`** a planner model chooses the task kinds. On a one-page manual and a 0.6B policy it chose kinds the manual cannot support,
  the plan failed its own checks and the run stopped with `no plan was made`. Name the kinds instead.
- **`--kinds recall`** has a model write questions from the manual. The default setup uses the 0.6B policy as that writer, and both questions it proposed were
  rejected by the admission rules (one never named its subject, one named something the manual does not contain), so no task was admitted and the
  run stopped. The rules did their job: nothing that fails them is trained on. Name a stronger writer and teacher with `--generator` and `--teacher`.
  A model that brain has to place on the CPU backend (`brain doctor --model <it>` says so) is too slow to be usable: its requests timed out.
- **`--kinds denoise`** builds fill-in-the-blank tasks from the manual's own text in code, with exact answers, so it needs no writer model. One short
  document gave one task. The 0.6B policy answered it wrong on every attempt and so did the 0.6B teacher, so no task was worth training on; `learn`
  stopped with `no candidate was trained`, the gate did not run and the policy was unchanged.

To reach a trained candidate you need more material than one page (several documents give several tasks) and a teacher that can solve what the policy cannot.

## 7. Inspect what the run made

Whatever `learn` reported, these show what was stored and what happened:

```bash
splinter status
splinter runs list
splinter runs show $RUN
splinter tasks list
splinter tasks show $TASKSET
splinter experiences list
splinter experiences show $EXPERIENCE
```

`status` shows the policy in use, whether any release exists, the counts of what the stores hold (sources, task sets, tasks, experiences,
datasets, candidates), the recent runs and the concepts the policy has mastered least. `tasks show` prints a task set's tasks, or one task with
the source section its evidence falls in. `experiences show` prints an experience with its annotations: the verdicts, with the verifier that
gave each. Ids are the full `blake3:<hex>`, the hex, or a unique prefix of at least four hex digits. [Experience](../concepts/experience.md) and
[Verification](../concepts/verification.md) explain what the verdicts mean.

## 8. The dataset and the trained candidate

When a candidate was trained, the `learn` report names it (`candidate <id> trained`) and the dataset it was trained on. Export the dataset to read its
records and manifest:

```bash
splinter dataset export $DATASET --out /tmp/first-run-dataset
```

The records are chat conversations whose supervised answer was verified; the student's prompt is the instruction alone, never the passage a teacher
was shown. The manifest names where each record came from. See the [datasets reference](../reference/datasets.md). The candidate's training curve,
the step it carries and any warnings are in the report; the gate repeats the warnings beside its verdicts. To stop at the candidate and decide later, run
`learn` with `--no-release` and release it yourself in step 10.

## 9. Compare the candidate with the baseline

```bash
splinter eval $CANDIDATE --suite anchor
splinter eval $CANDIDATE --suite held-out
splinter eval matrix
```

Compare the anchor report with the one from step 3: it lists the tasks each model missed, so you see what the candidate lost as well as what
it gained. `--suite held-out` grades the candidate on the records its training held out. A gain on the manual's facts that costs anchor tasks is
exactly what the gate refuses. The suites are described in the [evaluation reference](../reference/evaluation.md#suites).

## 10. The release decision

```bash
splinter release $CANDIDATE
splinter release list
```

The gate compares the candidate with the champion (the base, before any release) on improvement, retention, anchor and serve. Each check is printed with
its numbers; a check that could not be measured fails:

```text
  improvement  <PASSED|FAILED: reason>
               <n> held-out task(s), <n> paired: candidate <share> vs champion <share>; wins <n>-<n>, ties <n>, unpaired <n>; sign test p = <p> (alpha <a>)
  retention    ...
  anchor       ...
  serve        ...
released <id>; default points at it
```

or `not released: the gate blocked it`. A release is immutable and carries every number the gate measured; `release list` shows each release with its parent
and the aliases that point at it. The manifest `brain serve` follows is written to `<state>/serving/default.json`, and `splinter rollback default` points the
alias back at the release it was trained from. See [Release, serve and roll back](../guides/releases.md) and [release policy](../concepts/release-policy.md).

## 11. Ask, and explain the evidence

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
| `FAIL role:...` from `doctor` | The role's model is not on disk | `brain pull`, or set `BRAIN_MODELS_DIR` or `BRAIN_QWEN_WEIGHTS` |
| `no plan was made` | The planner chose kinds the sources cannot support | Name `--kinds` |
| Every proposed task rejected | The writer model is too weak for the admission rules | Name `--generator` and `--teacher`; see [configuration](configuration.md) |
| `no candidate was trained` | No task was both failed sometimes and solved with a verified answer | More material, or a teacher that can solve what the policy cannot |
| `gate: blocked on ...` | A candidate was trained and the gate refused it | Read the failed check's numbers; the policy is unchanged |
| Exit status 2 | Refused before anything ran: usage, an unknown id, a remote model without the opt-in | Read the message |
| Exit status 3 | A sentence was asked back | Name the command, or answer the question |

See [limitations](../concepts/limitations.md) for what this walk-through cannot tell you.
