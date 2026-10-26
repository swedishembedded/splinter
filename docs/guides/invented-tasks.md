<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Tasks invented by agents

`learn` can have agents write the tasks instead of asking the generator model once per
window of the sources. The flywheel harnesses (`harnesses/flywheel/`) are run by `sven`
against a model server that is already running; Splinter lays out each run, admits what
it leaves and decides what is trained on.

```sh
splinter learn docs/ \
  --task-harness harnesses/flywheel/invent \
  --solve-harness harnesses/flywheel/solve \
  --validate-harness harnesses/flywheel/validate \
  --sven ~/bin/sven \
  --task-harness-endpoint http://127.0.0.1:8798/v1 \
  --task-harness-model brain/served-name \
  --task-harness-key-file keys.json
```

A brain server is addressed with the `brain/` provider and a key file of the form
`{"brain": "key"}`: the `openai/` driver never sends the thinking switch. The
`PATH` of every run holds the task runtime's directories (`workspace.runtime_dirs`), the
directory of `sven` and the default search path (`/usr/local/bin:/usr/bin:/bin`), where
`git`, `python3` and `bash`, which the harness's gates need, must be; a run is refused,
naming the tool, when one is missing.

## From source to tasks

The tasks stage runs invent once per window of each text part, on one workspace that
lasts for the stage, so a later window sees the tasks the earlier ones made. A run takes
tens of minutes; `--task-harness-timeout` (seconds, default 7200) bounds it and the
stage's budget cuts it short. A run that is stopped keeps the tasks it had committed.

`tasks/` in that workspace is a git repository and every admitted task is one commit.
Splinter reads only commits: the task directories added since the run began, as
committed. Each directory is

```text
instruction.md      what the solver is asked
environment/        the files the solver starts with
solution/solve.sh   a reference solution, run in a copy of environment/
tests/test.sh       the verdict, run in the finished workspace; exit 0 means solved
harness.yaml        the process that solves this task (optional)
```

and becomes a workspace task: the solver's tree is `environment/` with the instruction
as `task.md`; `tests/` is the hidden check, run with `TESTS_DIR` set. The task is held to
the bar of every workspace task - its tests fail on that tree on every run and the
change `solve.sh` makes passes them - and a directory that is not laid out as above is
refused with the missing piece named. The report of the tasks stage carries what invent
logged (admitted, rejected, duplicate, failed) beside what Splinter admitted and refused,
with the reason for each directory it refused.

## Solving and validating

A task is solved by its own `harness.yaml`, run in process on the task's tree; a task
without one is solved by `--solve-harness`. The task's own tests decide the attempt.

With `--validate-harness` the validate stage follows verify. Every passing attempt is
judged under each focus (outcome, process, teaching) by the validate harness, and kept
only when every decision is keep. A validator can remove a passing attempt and never
admit a failing one. A validator run that ends without a decision is not a discard: the
attempt is left out, and the run is reported apart with its focus and why.
