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

10. **Learning to learn across task families (the decisive test).** Train sequentially on a sequence of distinct domains
    sharing one evidence format, then start a domain chosen only after that learning is complete, with identical tool access,
    memory and budget: does the trained system reach a fixed competence threshold with fewer verified experiences and less
    compute than an otherwise equivalent baseline, under matched compute, on unseen instances, with irrelevant recent context
    removed, and with earlier domains retained? Primary metrics: time-to-competence, experiences to threshold, learning-curve
    area. Remembering A-E is learning those domains; transferring a procedure from A to F is transfer; acquiring structurally
    different domains faster is learning to learn.
    - **Invented systems, so nothing is already known.** The cheapest decisive environments are generated: a counterfactual Lua
      dialect (changed indexing base, operator meaning, scoping or library rules) with a generated manual and a modified
      reference interpreter as its verifier; a new virtual microcontroller in vemu; an invented API or workflow with a reference
      implementation. The model starts uncertain, the verifier is exact, and a family of dialects gives as many related but
      distinct domains as needed. Learning the manual's rules is shown by applying them to new programs, not by repeating it.
    - **Which mechanism learned** (same base model, comparable inference budgets, training cost recorded separately):
      A frozen, no manual; B frozen + manual (in-context acquisition); C + tools/experiments (learning by interaction); D +
      persistent skills or memory (external acquisition and reuse); E parameter updates; F the trained model without the
      manual (retained parametric learning).
    - **Three curves at once:** training-task performance, held-out performance in the same family, and performance on
      entirely new families. Training rising while held-out stays flat is not learning.
11. **Memory vs weights.** Where a task family has external memory (procedures, verified claims, retrieval), every checkpoint is
    examined in four conditions: base without memory, base with the updated memory, updated weights without memory, both. Store
    knowledge in the cheapest form that makes it reliably usable (an exact fact in retrieval, a tested procedure in procedural
    memory, a general decision skill in weights); a weight update is justified only by what the memory-only condition cannot do.
12. **Partial-success extraction.** Long episodes (agentic, vemu) fail as wholes but contain verified subgoals, verified failures
    and recoveries. Re-verify each successful segment in a fresh environment before promoting it (procedure memory, SFT record);
    hindsight relabelling of what an episode did achieve; local counterfactual corrections (observed state, actual action, a
    better action that was executed and verified). A failure is classified (wrong action, missing information, wrong order,
    environment nondeterminism, impossible task, verifier wrong) before anything learns from it.
13. **Learning-value selection.** Separate features rather than one hand-tuned scalar: verified correctness, novelty, learning
    progress, decision uncertainty (the boundary rule: sometimes right, sometimes wrong, which is also what makes a policy-gradient
    group informative, cf. dynamic sampling in DAPO, arXiv 2503.14476), transfer potential, regression relevance, cost.
    A starting allocation to ablate, not a validated one: 40% frontier, 20% transfer and composition, 15% prioritised replay,
    15% uncertainty-driven experiments, 10% random coverage. The examiner's distribution stays fixed so a shifted measurement
    cannot pass for improvement.
14. **Retention by prioritised replay.** Five collections (current, historical replay, boundary cases, transfer tests, protected
    evaluations); replay prefers items with demonstrated regression over uniform sampling; compare ordinary mixed replay,
    priority replay, replay with a gradient-alignment term (conflict diagnostic g_new . g_old < 0), a frozen base with per-skill
    adapters (bounded: no adapter per task forever) and periodic consolidation into one shared adapter.
15. **Exploration as experiment.** Epistemic actions (choose the experiment that best separates competing hypotheses about the
    environment per cost) beside instrumental ones, conditioned on the objective and expected reuse; an optional small outcome
    predictor guides exploration but never replaces a real observation when grading.
16. **Method by evidence.** Memory, then verified demonstrations, then SFT, then online RL, then consolidation, as a default, not
    a rule: RL with all-equal rewards in a group has no signal, so bootstrap by verified demonstrations or decomposition first.
    Distil expensive teacher discoveries into short verified procedures, judged on different tasks that need the same principle.

The essential ablation for each domain, at matched compute and separately at matched environment interactions, several seeds:
baseline; memory only; memory + SFT; + RL; + active curriculum. A component that does not beat the simpler variant is removed or
redesigned. The system may decide that no weight update is justified (better data, a corrected verifier, more memory instead).

## Evaluation streams (kept separate)

- Development feedback: available to the learning controller.
- Retention probes: fresh variants of previously acquired capabilities.
- Final transfer tests: sealed, never available to curriculum or method selection; the examiner keeps item-level results for
  later analysis, the learner never receives them.

## Reporting (every experiment)

Per-generation and per-domain learning curves with intervals, forward and backward transfer, retention, sample efficiency,
verified gain per GPU-hour (proposal, prediction, training and gating included), worst-case regression, and the full history of
accepted and rejected updates with each parent checkpoint and its evidence.

## When a skill counts as acquired (six gates, all on independent unseen tests)

Correctness (the outcome checked by the verifier, never the model's account of it); generalisation (variants, changed
descriptions, unseen combinations); robustness (disturbed inputs, missing information, tool failures, injected instructions);
calibration (confidence matches accuracy; abstention improves accepted accuracy); retention and transfer (persists without the
original context; context-use and consolidated learning reported separately); efficiency (time-to-competence, training tokens
and GPU-hours, verified gain per feedback event, execution cost). Thresholds are set per task by difficulty and consequence;
the weakest important category must pass, not only the average; around 200 or more independent held-out cases per important
skill with intervals, several learning seeds and independently generated environments. Each skill keeps one record: definition,
prerequisites, materials, environment, allowed tools, verifier, baseline, learning history, held-out and retention results,
costs. A new skill category changes materials, environment and verifier, never the learning loop.

Skills to train first because every domain reuses them: understanding new documentation, selecting informative experiments,
executing and verifying procedures, diagnosing failure, constructing reusable skills, transferring knowledge, maintaining
reliable memory, improving learning efficiency.

## Risks to design against

Faulty environment feedback (reference implementations, an environment reliability status: a wrong emulator teaches wrong rules
efficiently); reward hacking (the learner never controls grading; hidden tests and verifier definitions outside writable
environments); exploration that finds nothing to learn from; memorisation without transfer; forgetting; contaminated or
contradictory memory (provenance, uncertainty and contradiction stored, not flattened into facts); skill-library explosion;
capacity limits of a 4B model; redundant rollouts; measurement noise accepted as gain; self-confirming world models; drift;
agent, memory, examiner and environment co-adapting (answered only by independent evaluation on environments and distributions
that took no part in learning).

## Sources to check before citing (from a November 2026 research synthesis; none verified yet)

AgentCL (arXiv 2606.02461), When Continual Learning Moves to Memory (2604.27003), Revisiting Replay and Gradient Alignment
(PMLR v330, abbes26a), AgentHER (2603.21357), Agent Lightning (microsoft/agent-lightning), Active Inference and Artificial
Reasoning (Nature Communications 2026), Building World Models from Intrinsic Motivation (2611.13113), Agent-R1, AgentFly, a
bandit-sampled GRPO study on Qwen2.5-1.5B arithmetic. DAPO (2503.14476) is a known, existing paper. From a second synthesis (also unverified unless noted): SkillLearnBench (2604.20087;
self-feedback alone drifts), SkillGenBench (2605.18693), ContinualSkillBench (2608.03874; apparent gains from recent context),
test-time training for few-shot learning (PMLR v267, akyurek25a), IFBench (unseen instruction constraints). Known existing
evaluations: HELM, AgentDojo, PaperBench, SWE-bench, LiveCodeBench, Terminal-Bench, OSWorld, BFCL, ARC-AGI.

## Known gaps to close on the way

- The serving build behind the flywheel paper's published verdicts is in no published source history; examiner runs must use
  builds whose source is reachable.
- Learner-side engineering: a run fingerprint checked on resume, crash-recoverable journals (torn last line), the frontier
  journal keyed by sampling settings, no file-per-call proposer state.
- Appendix-level mechanisms that sound stronger than they are: lexical grounding is provenance, not entailment; verifier kinds
  check different properties (no universal ranking); aggregate gates can hide a rare skill's loss.
