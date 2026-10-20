<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of the acquisition experiments ACQ1, INT1 and GEN1

Written on 2026-10-19, before any run of these experiments exists. They extend the
coding-task registration (`coding-tasks.md`) to what comes after a first headline: a
capability the policy lacks acquired from documentation (ACQ1), the choice of how to improve
(INT1), and gains that persist over generations (GEN1). Anything in this file changed after the
first run of the experiment concerned is a dated amendment below the line "Amendments", and a
result computed under an amended rule is labelled post hoc. Thresholds are never changed after a
result is seen; a sample too small to decide is labelled a pilot, not rescued by a looser rule.
The mechanisms exist and are specified on scripted agents (the paper's section "After the first generation"); none of
these experiments has been run on a trained model.

## ACQ1: a documented domain acquired with no manual edit

- **Domain and objective.** The Lua 5.1 reference manual (the page the Lua project publishes for
  5.1) and the objective "Become an exceptional Lua 5.1 programmer", with the interpreter pinned
  to Lua 5.1.5 built from its release tarball.
- **Arms.** The 8B policy with the 27B model as teacher, generator and planner; the policy as it
  begins, and the policy as it ends, measured on the same final probes.
- **Unit.** A skill. A skill is assessed only if it has at least three dev and three final probes.
- **Primary criterion.** The run is accepted if every assessed skill reaches a pass rate of 0.7
  on its final probes, and at least half of the assessed skills have a final pass rate above the
  starting policy's. A run that does not meet it is reported as not accepted with the skills that
  fell short.
- **Controls and checks.** No final probe is ever attempted by a teacher or trained on, and no
  training task repeats a probe (both are checked and counted by the run); the competence graph
  and the skill graph of every revision are kept.
- **Pilot rule.** With fewer than eight assessed skills the result is labelled a pilot.

## INT1: choosing how to improve

- **Clusters.** Three families of target tasks on which the incumbent fails for different
  reasons: a harness that cuts work short, missing domain knowledge, and tasks that more samples
  solve. The clusters are named before the arms are measured, and an arm is applied to all
  of them.
- **Arms.** More samples, a longer episode protocol, a candidate trained from a teacher's
  demonstrations, and no change.
- **Primary criterion.** The per-cluster choice has a net value, at the pricing stated in each
  experiment file, at least as great as that of every fixed strategy over the clusters the
  strategy was measured in. An arm whose mechanism is not available is reported as not measured
  and is in no strategy.

## GEN1: three generations of a coding policy

- **Design.** The three generations of the E3 plan, each trained on the plan's families for it
  and held to the exams of every earlier family; the same plan under each replay arm (none,
  uniform, balanced over capabilities, weighted by the champion's failures, rebuilt from the
  base).
- **Primary criterion.** At least three generations are accepted, and the backward transfer of
  the full retention matrix is not below -0.05. A generation is accepted only if no protected
  family falls by more than 0.2, the mean over protected families falls by no more than 0.05,
  it gains on its new exam, and the release gate passes.
- **Replay comparison.** The average final performance of the full matrix under each replay arm,
  with the family-cluster bootstrap interval; reported whichever arm is ahead.
- **Curriculum arm.** Amendment 17 of `coding-tasks.md` adds, before any generation runs, an ablation of
  how the solve and teach budget is spent: the frontier (admission) order against a learning-progress order and
  a gain-ranked order on the same E3 plan, seeds and budget.
- **Learning-progress replay arm.** Amendment 18 of `coding-tasks.md` adds, before any generation runs, `--replay
  learning-progress` beside the uniform and failure-targeted replay arms: the replay weighted by how fast each earlier
  exam's pass rate has moved across releases, on the same E3 plan, seeds and budget.
