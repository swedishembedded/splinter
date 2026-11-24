<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 217. A token-bounded producer deadlocks on tokens its consumer never sees, or on a consumer that refuses below a step

**Lesson:** the online controller admits a new group only while the tokens it produced and the learner has not
applied stay below `max(max_staleness, 1) * tokens_per_step`. Two ways that bound closes for good: (1) the controller
counts tokens the learner never receives (inexact or unscored rollouts of a sent group), so the allowance is full
while the learner holds less than one step and waits for more; (2) the learner refuses groups (a full queue) while
it holds less than its step budget, so it can neither train nor take more. A queued group therefore counts only its
exact scored rollouts (`Controller::hold_trainable`), and a learner must accept while it holds less than a step:
brain's streaming job may answer 429 only once it holds a step's worth. Spec: splinter
`crates/factory/tests/it/online/simulation.rs` (mutant `online-inexact-tokens-queued`).

## Evidence (2026-11-24, lane agrpo-ctl)

The first run of the discrete-event simulation stopped at seed 8646803706839047604 with no progress until the
simulated horizon; with both causes fixed, three runs of at least 100k finished rollouts each (118 to 150 simulated
worlds) finish. Removing the first fix makes the simulation deadlock again (seed 10870117228981610903).
