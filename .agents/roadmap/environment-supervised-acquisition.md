<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Environment-supervised acquisition roadmap

Scope: what follows the flywheel paper's S4 result: acquiring transferable skills faster, keeping them longer and improving
future acquisition under a fixed budget, using the environment (an executable oracle) as the source of supervision. S4 stays
the baseline; three consecutive improving rounds are not the destination.

**Status: planned. Nothing below has been started.** Update this file in the same commit that starts, finishes or changes an
item. The compact-reasoning line (`compact-reasoning.md`) is one branch of this programme (reasoning traces and execution
states as supervision).

## What S4 established and what it did not (the starting point)

- Established: an adaptive acquisition loop (the champion proposes Lua programs, the interpreter labels them, a failure filter
  keeps the ones the champion gets wrong at least once in four samples, a fresh adapter is trained from the base on the growing
  pool) produced better adapters on its registered exam (4 of 4 lineages) and, on a fresh unselected exam, roughly doubled
  direct-prediction accuracy by generation 3 (generation 1 alone did not move). The interpreter supplies correct targets on
  problems the model never solves: environment supervision, not reinforcement of the model's own successes.
- Not established: that recursion helps (recursive structure exists; a benefit of it over suitable controls and improving
  improvement efficiency are unshown), which of S4's simultaneous changes matter, retention (generation 3 loses 14-30% of its
  predecessor's right items), reasoning (thinking off everywhere; the base with thinking on far exceeds generation 3), breadth
  beyond one task family.

## Experiments, in order (each registered before its run, judged by the separate examiner on sealed exams)

1. **Proposer x selector (2x2).** Proposer {frozen base, updated champion} x frontier selector {frozen base, updated champion};
   datasets trained from the same base at matched training-token budgets, several training seeds. Separates the evolving
   proposer, adaptive selection and their interaction (the matched-data control M confounds the two roles).
2. **Data- and compute-matched baselines.** S4 spends 75 + 225 + 525 = 825 training steps over three rounds against 525 for one
   final-sized fit (57% more, before proposal and prediction cost): static-data baselines matched on tokens and on total compute
   (GPU-hours incl. proposing and predicting), reported as capability per budget.
3. **Reset vs continuation** on controlled data schedules (identical data, adapter continued vs retrained from the base).
4. **Richer supervision from the same oracle** (multiple verifiable task types): predict output (S4), predict intermediate state
   (instrumented runtime trace), find an input producing an outcome (execute the proposal), infer a program from examples (hidden
   tests), repair a defect (behavioural + regression checks), produce a counterexample (execution witness), compose learned
   operations. Output-only vs execution traces vs the mix, on held-out families.
5. **Curriculum by learnability, not difficulty.** The four-sample failure filter admits with probability 1 - p^4 (a 90%-right
   program 34% of the time); classify how a problem can teach (sometimes solved: outcome-based RL or verified self-imitation;
   never solved with an oracle answer: supervised acquisition; never solved with execution states: trace supervision; trained
   repeatedly without transfer: prerequisites or relevance) and allocate by measured transfer per cost: objective
   (gain on independent dev tasks - lambda * retention loss) / acquisition and update cost; start with category-level bandit
   allocation (as in Self-Evolving Curriculum) and occasional controlled update probes.
6. **Mixed acquisition sources:** model proposals, procedural generation, mutations, real failures, prerequisite exercises;
   coverage and collapse measured (the proposer's distribution vs the intended domain).
7. **Retention and continual learning:** longer mixed-domain sequences with bounded replay and fixed storage/training budgets;
   a skill-by-generation matrix (rows: capabilities acquired at some point, columns: the model after each later update) measured
   with fresh independent variants; whether earlier learning speeds later acquisition.
8. **Learning to improve the solver:** the improvement mechanism (exercise design, failure diagnosis, update choice) as its own
   optimisation target, rewarded by downstream gain after training; compared with a fixed mechanism at equal cost.
9. **Other domains** (cross-domain plan): Python output prediction (near free), the compiled document-rule environment, then
   stateful and agentic domains once reasoning and policy-gradient numerics are qualified.

## Evaluation streams (kept separate)

- Development feedback: available to the learning controller.
- Retention probes: fresh variants of previously acquired capabilities.
- Final transfer tests: sealed, never available to curriculum or method selection; the examiner keeps item-level results for
  later analysis, the learner never receives them.

## Known gaps to close on the way

- The serving build behind the flywheel paper's published verdicts is in no published source history; examiner runs must use
  builds whose source is reachable.
- Learner-side engineering: a run fingerprint checked on resume, crash-recoverable journals (torn last line), the frontier
  journal keyed by sampling settings, no file-per-call proposer state.
- Appendix-level mechanisms that sound stronger than they are: lexical grounding is provenance, not entailment; verifier kinds
  check different properties (no universal ranking); aggregate gates can hide a rare skill's loss.
