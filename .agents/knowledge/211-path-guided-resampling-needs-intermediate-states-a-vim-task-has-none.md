<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 211. Path-guided resampling needs intermediate states, and a vim task has almost none

**Lesson:** forking a student at a verified path's states yields training material only where some state lies
between "nothing done" and "work done". A vim task of the headless-mastery course is one decisive command: a teacher
pass has a median of 1 state-changing action. A fork before it is the from-scratch problem the student already fails;
a fork after it leaves the student nothing to do but stop. Count the state-changing actions per path before spending
GPU time on path-guided resampling (E4), and run it on multi-step domains.

Three generic rules came out of the run, all specs in `crates/pipelines/tests/it/path_guided.rs`:
- A fork point is an *open* state: the task's verifier does not pass the path's state there yet. Each state is
  graded once by forking there with a model that ends at once. The forks run earliest first; samples beyond the open
  states try them again under their own seeds.
- A continuation that calls no tool is refused: it is the path's work, not the student's.
- A passing continuation from an open state therefore changed the workspace itself; the kept suffix trains only the
  student's own steps (the copied path is context).

## Evidence (2026-11-21, lane PLAY, E4 phase 1; data in `lane-play/e4`)

- 73 train tasks the 4B never solved in `train-k1`/`train-k4` (5 attempts each). The 27B teacher passed 69 of them
  (214 passes). Replay through the tasks' own tools (bench `task replay`): 149 passes use `shell`, which the task's
  Sven configuration does not offer the 4B; 26 hit the vim tool's random scratch path (bench knowledge 076); 39 passes
  on 29 tasks replay. Per task: 29 routed, 27 every pass uses shell, 11 shell or scratch path, 2 scratch path only,
  4 no teacher pass.
- State-changing actions per teacher pass (vim write commands, shell writes): median 1; 123 passes have 1, 65 have 2,
  26 have 3 or more.
- Fork after every action but the last, latest first, 3 samples: 117 of 145 failures "repaired" on 24 tasks, but
  in 101 of the 106 dataset records the only target was a final reply after the teacher had done the work.
- With continuations without a tool call refused (4 samples): 12 failures on 5 tasks.
- With samples revisiting states (8 samples): 19 failures on 7 of 29 routed tasks, 24% (7 of 73, 9.6%); 421 of 1,048
  attempts failed the verifier, 559 were refused as the path's own work. The pre-registered gate (30%) fails: E4 on
  vim is recorded `falsified` (experiment `play-e4-gate`) in the canonical campaign. The open-state rule landed after
  this run and was not run on vim: with a median of one state-changing action it can only turn the refusals into
  verifier failures (the state before the decisive command is the from-scratch problem), not reach 22 of 73 tasks.
