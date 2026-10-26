<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# The flywheel as sven harnesses, and the prior work it draws on

Status: design note, 2026-10-26. The harnesses are `harnesses/flywheel/{invent,solve,validate}`.
Works are cited by title and year from memory, not re-read for this note; check a claim
against the paper before a paper states it.

## Split of responsibility

- **sven runs the agents.** Each stage is a v2 harness document. The host starts
  `sven harness run <dir> --workspace <tree> --model ... --json --trajectory ... --conversations ...`
  as a process, the way a benchmark harness runs an agent on a task directory. The agent's
  workspace holds only what the stage may see.
- **Splinter orchestrates and judges.** It lays out each workspace, fans runs out in parallel,
  reads back the files and ATIF trajectories sven leaves, grades with checks the agents never saw,
  and decides what becomes training data. Nothing an agent says is evidence.
- **brain serves and trains.** One `brain serve` per stage carries all of that stage's concurrent
  harness runs. The incumbent model plays every role.

## Stages

| Stage | Agents (each in a fresh conversation) | Decided by code | Decided by agents, as typed choices |
|---|---|---|---|
| invent | explorer (maps the data, chooses angles) -> per angle: proposer -> novelty -> author -> reviewer | the task directory is complete; tests fail on the untouched environment and pass after the reference solution; still so when admitted | novelty (`novel`/`duplicate`, nearest task, reason); admission (`accept`/`reject`, reasons, unfixed problems) |
| solve | one engineer per attempt in a copy of the task's environment | the task's hidden `tests/test.sh` on the finished workspace | none |
| validate | one judge per focus (outcome, process, teaching), in parallel, per passing attempt | nothing more | `keep`/`discard`, with repeatable evidence and rejection reasons |

The output of invent is a directory of runnable tasks in the shape a benchmark harness
runs (instruction, environment, reference solution, hidden tests), so the same tasks feed
training and evaluation, and every task can be run end to end again later, with another
model or another harness.

## Why judgements are decisions, not string checks

A first version checked task quality in code: quotes compared with the source word for
word, outputs recorded from runs, a minimum number of tests. It refused sound tasks over
markup and accepted weak ones that met the counts. Grounding, fairness, novelty and
worth are judgements over rich context, so they are taken by an agent that reads the
files, the transcripts and the material, runs what it doubts, and answers with one of a
fixed set of options plus its reasons. Two things follow:

- the decision is recorded with the task or trajectory it admitted, so a rejected item
  can be audited and a judge's precision measured against a sample a person grades;
- the shape is the one a decision model answers (a distribution over options given a
  state). sven's decision runtime (`crates/decision`, served by brain, e.g.
  `convaiinnovations/laya`) can take a step's place once the harness language has a
  `decide` step backed by the run's decision provider - open work in sven.

Code keeps what execution decides, because execution cannot be argued with.

## What is taken from where

- **Grounding in raw content, transformed first.** Instruction-synthesis pipelines that start from
  raw documents use content-transformation agents to turn text into an intermediate form before any
  instruction is written (AgentInstruct, 2024). Here the surveyor turns manual sections into a list
  of testable behaviours. A writer sees one behaviour and the source, never a blank page.
  Seeding from real artefacts instead of from the model's own memory is the core of OSS-Instruct
  (Magicoder, 2023).
- **Task = instruction + reference + verification + failure cases.** Code-as-task formulations
  make a task carry its own verifier and an example solution, and check that the verifier rejects
  known-bad solutions (Self-Challenging Language Model Agents, 2025). Here: tests must print
  something different without the reference, the reviewer runs a deliberately wrong solution, and
  splinter's mutation validation repeats it on admission. Code tasks shipped with a solution and
  tests that the solution passes are also the shape of KodCode (2025).
- **Layered verification: format, then execution, then semantics.** Function-calling data
  pipelines filter in that order, cheapest first (APIGen, 2024). Here: the gate (format, quotes,
  execution), splinter's admission, the hidden tests on solve attempts, then model judges.
- **Refinement by a critic.** Suggester/editor pairs that rewrite a seed to be harder or cleaner
  (AgentInstruct's refinement flows, 2024) and evolution of instructions toward depth and breadth
  (Evol-Instruct, 2023). Here the reviewer may rewrite or drop. Making a task harder on purpose is
  left to the frontier rule below, not to a prompt.
- **Learnability, not difficulty.** Self-play setups reward a proposer for tasks the solver
  sometimes solves (Absolute Zero, 2025; R-Zero, 2025). Splinter's frontier rule does the same:
  attempts with pass rate strictly between 0 and 1 are kept, and always-solved tasks are not
  trained on.
- **Train on what passed: rejection sampling on trajectories.** Self-taught reasoning and its
  successors fine-tune on the model's own successful attempts (STaR, 2022; ReST, 2023). For agents,
  the attempt is a tool-use trajectory filtered by hidden tests (SWE-Gym, 2024; SWE-smith, 2025).
  The solve stage's ATIF conversation is that trajectory.
- **A jury, not a single judge.** Several smaller judges with different instructions disagree less
  with people than one large judge does (Replacing Judges with Juries, 2024). Here three judges with
  distinct foci run in parallel, and all must keep a trajectory.

- **Generation, execution, verification and filtering as separate stages, deterministic first.**
  The Nemotron 3 agentic data pipelines (technical reports, NeMo Data Designer, NeMo Gym)
  generate tasks, execute them with agents across several harnesses, judge per turn and per
  trajectory, filter structurally and by difficulty, and keep independently verified
  successes. Their published coding-trajectory rejection criteria - no submission, invalid
  git operations, edit-test thrashing, lost exploration, malformed tool calls, debug
  artefacts left behind, untested changes - are the process judge's instructions here.
  Unlike their single-call synthetic conversations, every tool result here is a real
  execution.
- **Run the same tasks under more than one harness.** Varying the harness keeps a model from
  learning one harness's habits; a task directory can be run by any agent runner.

## Where this departs from those pipelines, and why

- **Who invents.** Measured on the Lua manual (2026-10-26): Qwen3-4B as inventor made no tool
  call in 30 writer turns, invented citations and wrote stub references, so no task passed. The
  local 27B wrote grounded, well-tested tasks through the same harness. Inventing and judging
  need a stronger model than the 4B policy, as in the large-teacher pipelines; the policy solves.
  Whether the policy can take over inventing as it improves is an open measurement.
- **One idea per conversation.** A failed idea is dropped (`on_item_failure`) without failing the
  run, and no conversation grows past one task.
- **Judges only remove.** Correctness is decided by the hidden tests. A validator can take a passing
  trajectory away but cannot let a failing one in, so a judge's mistakes cost yield, never
  correctness. A 4B judge marked a correct, general solution "hard-coded" while answering keep; a
  typed decision with evidence makes such contradictions visible.
- **Execution is the authority.** Every prompt says to run things rather than trust memory, and
  every check that execution can decide is code.

## Open questions

- Does the explorer's single pass cover the material, or should it run per part with the novelty
  decision across parts?
- Does training on solve trajectories with thinking enabled beat training without it at 4B?
  Thinking is off for now; sven ends a looping turn with the stop status `watchdog`.
- How should the judges' decisions be combined? Unanimity is a guess; measure each focus's
  precision against a sample a person grades.
- A `decide` step in the harness language, answered by a decision model, for the novelty and
  admission decisions.

## Next workstream: a Linux kernel expert dataset

The same three stages, with the kernel's own history and test tooling as the source of truth.
Steps, cheapest verification first:

1. Miner (deterministic): commits with a `Fixes:` tag whose changed files a KUnit suite covers;
   parent, fix and kunitconfig recorded.
2. Ground-truth gate (deterministic, in an isolated copy): the parent builds, the suite fails
   on it, the original fix makes it pass, and reverting the fix makes it fail again. The golden
   patch is kept for analysis, never as an exact-match answer.
3. Agents: a task designer writes the symptom-level issue without the fix; a feasibility
   decision (ACCEPT, REJECT, NEEDS_REPAIR, with evidence); the novelty decision against the index.
4. Solve: a kernel-engineer harness in a worktree at the parent; afterwards the verifier applies
   the agent's diff to a clean baseline and runs the private suite.
5. Datasets: passing attempts for SFT, pass/fail pairs on one task for preference training,
   still-hard tasks for RL, a frozen held-out set as the exam.

Constraint of the current nodes: they are aarch64 (GH200), where KUnit's user-mode Linux path
is not available, so tier 1 is KUnit under QEMU (KVM if the node allows it). Measure a minimal
KUnit build and run before fixing the design.
