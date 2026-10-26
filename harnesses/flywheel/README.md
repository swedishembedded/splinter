<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Flywheel harnesses

Three sven harness documents (api_version 2) that turn raw data into a diverse set of
verified agent trajectories. sven runs them, one `sven harness run` process per run.
Splinter lays out each run's workspace, starts runs in parallel, reads back what they
leave, runs the checks the agents never see, and decides what is trained on. The design
and the prior work it draws on are in `.agents/research/flywheel-harnesses.md`.

| Harness | Workspace the host lays out | Inputs | Leaves |
|---|---|---|---|
| [invent](invent/harness.yaml) | `data/` (the raw material), `brief.md` (audience and skills), `tasks/` (earlier tasks, may be empty) | `count` (angles, so new tasks, per run) | new task directories in `tasks/`, an updated `tasks/INDEX.md` |
| [solve](solve/harness.yaml) | a copy of a task's `environment/`, its `instruction.md` as `TASK.md` | none | the finished workspace; the conversation (`--conversations`) is the trajectory |
| [validate](validate/harness.yaml) | `TASK.md`, `workspace/` (the finished work), `tests.md` (hidden tests and what they printed), `trajectory.md` | `focus`: `outcome`, `process` or `teaching` | output `decision`: `{decision: keep or discard, evidence, rejection_reasons}` |

## A task directory

```text
tasks/<slug>/instruction.md      what the solver is asked
tasks/<slug>/environment/        the files the solver starts with
tasks/<slug>/solution/solve.sh   a reference solution, run in a copy of environment/
tasks/<slug>/tests/test.sh       hidden verification, run in the finished workspace; exit 0 = solved
tasks/<slug>/task.json           the idea, the novelty and review decisions, and the gate's runs
```

This is the shape a benchmark harness runs. A task is solved end to end by copying
`environment/`, running the solve harness on it, and running `tests/test.sh`. The same
tasks therefore serve as training material and as an exam.

## The loop the host runs

1. **invent**, repeatedly on the same `tasks/`. Each run explores the data, picks angles
   the index does not cover yet, and for each angle runs four agents. A proposer suggests
   one real problem. A novelty decision compares it with the index. An author builds the
   directory. A reviewer takes the admission decision over everything: files, the gate's
   transcripts, the material and the index. The gate admits a task only when its tests
   fail on the untouched environment and pass after the reference solution.
2. **solve**, k runs per task. The task's tests decide each attempt. The pass rate per
   task gives the frontier: tasks always or never solved are not trained on.
3. **validate**, one run per passing attempt and focus, in parallel. An attempt is kept
   only when every decision is `keep`. A validator can remove a passing attempt but never
   admit a failing one.
4. The kept conversations are the training records: ATIF with every tool call and result.

## What decides what

- **Code decides only what execution can decide.** Every file exists. The tests fail
  before and pass after the reference solution. The admitted task still holds up when the
  reviewer accepts it. The hidden tests pass on the solver's work.
- **Agents take every judgement as a typed decision with its reasons and evidence:** novel
  or duplicate, accept or reject, keep or discard. The decision travels with the task in
  `task.json` or with the trajectory in the dataset. A decision is a choice over options,
  so a decision model can take the place of the agent later without the documents changing
  shape.

## Running one by hand

```sh
sven harness run harnesses/flywheel/invent --input-file inputs.json --workspace WS \
    --scratchpad SCRATCH --model openai/<name> --model-endpoint http://127.0.0.1:<port>/v1 \
    --api-key-file keys.json --approvals auto --json --trajectory traj.json --conversations CONV
```

The tools the tasks need (e.g. `lua`) must be on the PATH of the `sven` process.
`--scratchpad` must lie outside the workspace; the invent gate installs its checker there,
out of the agents' reach.
