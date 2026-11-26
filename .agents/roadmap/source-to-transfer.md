<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Source to transfer: do trajectories on tasks authored from a manual teach skills that transfer

Scope: the controlled experiment, registered 2026-11-26 before any of its results existed, that tests whether tasks
Splinter authors from source knowledge alone, solved through sven, yield trajectories that teach the next model skills
it applies to unseen tasks, and which property of the training material carries the effect.

## Why this experiment, and why SFT first

The factory makes verifiable tasks; it has not shown that it makes transferable learning experiences. Earlier evidence:
the vim manual generation 1 (582 reviewed tasks, about 960 verified solutions) moved the 4B student from 0 to 4 of 100
exams solved within five attempts (p 0.0625, inconclusive); on-policy RL (R19) and rejection-sampling SFT on practice
tasks gave no held-out gain (`experience-to-competence.md`). Without a negative control, a gain could also be mere
exposure to the tool's format; without arms that differ only in the material, nothing says which property helps. The
experiment is SFT only: if correct, curated trajectories do not transfer under SFT, an RL result would not be
interpretable either.

## Material (one run, shared by every arm)

- Source: the Vim 9.1 user manual (`usr_*.txt` of the runtime the task image carries), chapter and section headings
  added by bench `environments/vim/manual_headings.py`: 222 sections, split to 263 items of at most about 1500 words.
- Tasks: bench `recipes/vim-manual/transfer.yaml` (the range pipeline: three material variants of one to six steps per
  section, Vim runs every reference, refusals repaired once, a reviewer checks each result against its request, fixes
  re-reviewed, the oracle solves every task kept). The author and teacher is Qwen3.8-27B-FP8, the student Qwen3-4B,
  thinking off, sven's narrate-before-each-call prompt, budget 60 turns and 16384 completion tokens.
- Attempts: the student 8 per task (temperature 0.7, top-p 0.8, top-k 20), the teacher 4.
- T1 hold-out: 10% of the authored tasks by task id (`tasks.pin` `holdout`, seed fixed in the arms recipe); no arm
  trains on any trial of them.

## Arms (from the same base Qwen3-4B, LoRA rank 16, alpha 32, lr 2e-4, 800 steps of 4 records, max block 16384,
## checkpoint chosen by validation loss every 40 steps, patience 4; seeds 1, 2, 3)

| Arm | Training records |
|---|---|
| A | none (the base model) |
| B | every verified successful trial of teacher and student, every action supervised (the existing pipeline) |
| C | B with at most one trial per task, the one with the fewest actions (diversity per token) |
| D | B with only productive actions supervised: an action whose tool calls all returned an error stays in context, unsupervised |
| F | negative control: B with each record's task request replaced by another task's (derangement); same tool-use format, no request-to-action mapping |

## Measurements

- T2/T4 (unseen material, unseen structure, compositions of several ideas): the 100 held-out course exams
  (`tasks/vim` eval side, hand-written, never authored from the manual or read by any stage), 4 attempts per exam per
  candidate, same sampling and budget as practice, concurrency within the servers' slots.
- T1 (unseen instances of the authored distribution): the held-out authored tasks, 4 attempts each.
- Infrastructure failures are never a fail (bench's classification); a run with more than 2% infrastructure failures in
  any arm is repeated after its cause is fixed.

## Decision rule (fixed now)

- Unit: an exam task. For a pair of arms X, Y: d_t = (passes of X over its 3 seeds x 4 attempts) - (passes of Y over
  12 attempts; for A, 12 attempts of the base). One-sided exact sign test over tasks with d_t != 0.
- Primary family (Holm, alpha 0.05): B > A, D > A, C > A on the exams.
- The goal's claim ("trajectories teach transferable skills") holds when at least one of B, C, D beats A in the
  primary family AND the same arm beats F on the exams (one-sided sign test, alpha 0.05). Beating A alone could be
  format exposure; beating F says the content of the trajectories matters.
- Secondary, reported with task-bootstrap 95% intervals and never as claims when not significant: D vs B, C vs B,
  every arm on T1, per-seed rates, failure classes.
- No exam result selects a checkpoint, a recipe, a seed or a data mix; the exam trials are never trained on.

## Status

- 2026-11-26 14:50: material run started on 16 GH200 (4 nodes, one brain per card behind one router).
- Arms options (curation, per-task limit, shuffled control, holdout, composition report) being built in splinter.
