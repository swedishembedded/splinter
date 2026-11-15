<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Generalized autonomous learning roadmap

Scope: what the four repositories need so that one learning process acquires, verifies, retains and transfers many different skills, and becomes cheaper at acquiring new ones, with every improvement independently examined.

**Status: planned.** Package 0 is the critical path; nothing else claims a result before it passes. Update this file in the
same commit that starts, finishes or changes an item. Related ledgers: `environment-supervised-acquisition.md` (the
experiment programme after S4; its learning-to-learn item 10 is this roadmap's E8), `compact-reasoning.md` (reasoning traces),
`continual-learning.md` (CL-P1..P4), `environment-factory.md`, `recursive-improvement.md`; sven `continual-learning.md`
(CL-S1..S3); brain `grpo-corrections.md`. The paper this work reports into is `papers/generalized-continuous-learning`.

## Roles (unchanged boundaries)

Sven is the actor and observer, Brain the trainable model substrate, Splinter the learning scientist and controller, Bench the
independent examiner. No fifth subsystem: extend the existing contracts (campaign state machine, factory plugins and admission,
training capability layer, release lineage, Bench's signed verdicts). Bench is never a source of training examples.

## State on 2026-11-15 (checked against the mains, not assumed)

- Demonstrated: environment-supervised acquisition in one family (S4/RSI4: Lua output prediction, interpreter labels, 4 of 4
  lineages on the registered exam; on a fresh exam, gains at generations 2 and 3, not 1; a matched-data control inconclusive).
- Not demonstrated: that an evolving proposer beats a frozen one at equal cost, transfer to other skills or domains, retention
  under continued learning, cheaper acquisition of new skills.
- Gaps confirmed in the code and ledgers:
  - `EpisodeExecutor` (`crates/pipelines/src/campaign/endpoint.rs`) has no Sven adapter; `recursive-improvement.md` says the
    Sven side landed as `sven_sdk::rollout`, but sven main has no `rollout` module (lost or never merged): the documentation is
    wrong until a compiling cross-repo test says otherwise.
  - Campaign policy-gradient steps do not carry `behavior_logprobs` (they train on the start-of-run proxy, exact on-policy only).
  - Agent trials could reach the benchmark's snapshot store (open item in `recursive-improvement.md`).
  - Sven CL-S1..S3 (memory evidence and lifecycle, recorded retrieval and withholding, skills with origin and lifecycle) and
    Splinter CL-P2..P4 (qualification and withholding, post-release monitor and rollback, fixtures) are open.
  - Policy-gradient projection broadcasts one attempt-level advantage over every action.
  - Splinter's `SkillLibrary` reuses verified sub-answers; that is answer reuse, not a demonstrated general procedure.
  - Bench knowledge 029: held-out documents are not held-out rules (23 of 78 items shared rule-equivalent elements).
  - Reused as is: `crates/eval/src/continual.rs` (accuracy, backward transfer, forgetting, forward transfer, gain curves).

## Concepts

- **A skill is an experimentally defined capability** (a demonstrated ability over a declared task distribution), not a
  prompt, adapter, `SKILL.md` or score. Its implementations are learning artifacts.
- **SkillSpec** (versioned, built on `splinter-knowledge` objectives and the factory): id and version; prerequisites
  (many-to-many with tasks); materials; environments; allowed actions and feedback; candidate strategies; verification class
  and splits (training, development, held-out family, held-out domain); mastery metric, target and retention margin;
  constraints (no test tampering, no hidden-answer access); resource limits; transfer targets. A category without an objective
  verifier is marked as needing calibrated judgment or human evidence, never treated as verified.
- **Environment lifecycle** on top of `factory::Plugin`, `AdmissionReport`, `EpisodeExecutor`: prepare, observe, act, reset,
  verify, diagnose (authorized training feedback incl. failed subgoals), snapshot, close. The agent operates an environment and
  cannot modify its verifier, reference, grading data or evidence ledger.
- **One evidence record** for every method: goal and environment identity; model, adapter and tokenizer identity; observations
  and tool calls; prompt and generated token ids; behaviour log-probabilities (policy distribution distinct from the sampler's);
  verifier outcomes and verified intermediate outcomes (reward atoms); failure class (reasoning vs tooling vs infrastructure);
  resources; exposure and provenance. A missing field an objective needs is a typed refusal. Sven ATIF supplies most of it.
- **Roles:** explorer (investigates, proposes experiments and tasks; its conclusions are not knowledge until verified), solver,
  teacher (an interpreter, emulator, solver or compiler is often the most reliable teacher), verifier (outside the learner).
- **Persistent artifacts**, each with origin, prerequisites, applicability, contracts, evidence strength, revisions: factual
  memory (verified provenance), procedural memory (works on other configurations), executable skill (independent execution
  tests on unseen instances), parametric capability (examiner-verified gain, retention, serving correctness). Utility is
  measured when used, by randomized paired withholding (skill available vs withheld, same tasks and resources).
- **Strategy by situation** (a small explicit auditable router first, no learned meta-controller yet): documentation available
  -> retrieval and guided application; known target answers -> SFT on verified examples; mixed outcomes -> search, verified
  imitation, then policy gradient; all fail with an oracle for states -> execution-trace supervision, decomposition; all fail,
  a teacher solves -> demonstrations without privileged answers at inference; repeated procedure -> verified memory or
  executable skill; poor tool execution -> tool-use examples and harness changes; forgetting -> rehearsal, adapter isolation,
  cumulative retraining; training feedback up but dev flat -> quarantine the reward source; no trustworthy verifier -> build
  one or obtain calibrated human supervision before any training.
- **Learning value** of (task family, intervention): (expected held-out gain - lambda * retention harm) / expected cost,
  estimated by controlled probes on development tasks with uncertainty and some exploration. (RSI4's admission rule selects a
  90%-right task with probability 1 - 0.9^4 = 34%.)

## Examination (Bench)

- **Novelty tiers:** only template, rule, source, tool, composition, domain or learning-domain holdouts support a
  generalization claim; a new seed of a known generator is tier A (instances) only. Rule holdout is checked for rule
  equivalence, not only by document or symbol.
- **Longitudinal skill matrix:** model state after each learning stage x skill families and a new domain, each cell on fresh
  independent items; acquisition, retention and forward transfer computed by `continual.rs` from Bench's aggregates (opaque
  family identifiers where naming one would expose an exam).
- **Promotion policy (one for all candidate types):** tier A gains only -> no generalization claim; tier B gain with retention
  -> skill-specific promotion; tier C gain with retention -> cross-domain transfer; gain with serious regression elsewhere ->
  reject or isolate; underpowered -> inconclusive; contamination -> invalid; a random-reward control also improves ->
  investigate the signal; benefit vanishes without retrieved memory or skills -> scaffold improvement, not parametric learning.
  Family-clustered analysis for generated items; one prespecified error budget across Splinter's evidence gate and Bench's
  exam rotation.
- **Verifier quality as an exam dimension:** known-bad solutions, mutations, empty submissions, tampered environments,
  misleading outputs, random reward; the result must be causally tied to the skill outcome, not to length, format, test edits
  or memorized strings (extends the admission red team).

## Packages (strict dependency gates)

- [ ] **P0 Cross-repo training contract (critical path).** Sven: a training-rollout interface (resolve the missing
  `sven_sdk::rollout`) exporting complete verified attempts with policy traces and failure states. Splinter: the Sven
  `EpisodeExecutor` adapter and strict record validation. Brain: exact token/probability correspondence, serving identity,
  adapter revisions, trainer numerics. Bench: a cross-project smoke examining the resulting candidate. Make the snapshot-store
  isolation an executable security test. Exit: a tiny agent completes sandboxed tasks, Splinter builds a valid dataset, Brain
  trains, Sven serves the candidate, Bench signs a verdict; truncated, mismatched or unprovable traces are refused.
- [ ] **P1 General environment factory.** SkillSpec, prerequisite graph, the lifecycle above, unified competence records;
  Sven environment adapters for tools and simulators; Bench exam protocols and novelty audits beyond program output and
  compiled documents; a knowledge-source compiler (manuals, repositories, API specs, schemas, tests, observations -> facts,
  rules, procedures, preconditions, examples, counterexamples, unknowns), separating what a source says from what is
  established, never minting a verifier from its own untested reading. Exit: one unchanged controller runs Lua prediction, a
  stateful database workflow, a repository repair task and a firmware-emulator task, each with real-model baseline bands and
  passing reference-solution, no-op, reproducibility, adversarial-verifier and isolation tests.
- [ ] **P2 Multi-strategy acquisition.** The strategy router and an active curriculum by learning gain per cost; Sven targeted
  exploration, structured failure diagnosis, resumable experiments; Brain capability-checked methods with job manifests and
  resource accounting. Exit: different interventions chosen automatically for an all-fail task with an oracle, a partly solved
  task and a repetitive procedure, each beating its matched baseline on held-out tests.
- [ ] **P3 Verified consolidation and retention.** Sven CL-S1..S3; Splinter CL-P2..P4; Brain paired-adapter comparison and
  rollback; Bench longitudinal retention and skill-ablation exams. Exit: a learned procedure improves performance, survives
  restarts, and can be disabled or retired without corrupting earlier competence, with evidence for every transition.
- [ ] **P4 Learning from agent trajectories.** Reward atoms from verifiers, diagnostic dataset views, action-dependency
  tracing by snapshot and replay, credit only where the dependency is evidenced (never an LLM's explanation); Brain real-weight
  policy-gradient qualification (log-prob provenance Sven -> Splinter -> Brain, policy vs sampler probabilities, exact prompt,
  tokens, template, model and adapter revisions, refusal past the declared mismatch bound). Exit: credit-aware training beats
  outcome-only training on a registered, powered multi-step distribution at matched compute without retention loss; otherwise
  the simpler method stays.
- [ ] **P5 Meta-learning and open-ended campaigns.** The controller learns which environment, exercise or intervention gives
  the most verified transfer per cost; Sven proposes experiments and investigates new sources within approved campaigns; Brain
  safe reversible updates and efficient multi-candidate comparison; Bench unseen-domain learning-to-learn and retention studies.
  Exit: the improved controller lowers cost-to-competence across several genuinely new domains vs an equivalent fixed one.

## Experiment ladder (each registered with distributions, baseline, failure conditions, budget, test and threshold)

- [ ] E1 Integration: a real Sven trajectory trains Brain with nothing essential lost (= P0 exit). Prerequisite for the rest.
- [ ] E2 Domain transfer: Lua-acquired competence transfers to Python or an invented language, vs frozen and matched-data
  controls, on held-out structural families.
- [ ] E3 Oracle vs RL: oracle-supervised learning beats sparse-reward RL on initially unsolved tasks per GPU-hour.
- [ ] E4 Trace supervision: intermediate execution states improve unseen longer and deeper programs at matched budget.
- [ ] E5 Procedural memory: verified reusable skills raise first-attempt success and cut tool calls on compound agent tasks.
- [ ] E6 Multi-domain retention: a mixed curriculum accumulates skills with registered retention non-inferiority.
- [ ] E7 Exploration strategy: active experiment selection beats random and fixed curricula on the learning curve.
- [ ] E8 Learning to learn: after learning a sequence of domains, freeze controller, memories, procedures and artifacts; on
  several unseen F-like domains (equal manual, tools, feedback and practice), compare A0 untouched model + fixed controller,
  A1 + earlier transcripts in context, A2 + verified procedural memory and skills, A3 adapter from earlier domains, A4 memory +
  adapter, A5 A4 + the improved controller, A6 matched-cost placebo experience; also the improved-information ablation (fixed
  controller given the same accumulated statistics). Primary: held-out success vs total cost (earlier-domain cost counted),
  learning-curve area, cost to a registered threshold; seeds and learning-order permutations. A5 > A0 shows useful accumulated
  experience; A5 > A4 at matched conditions shows the learning process itself improved.

E2-E5 can run in parallel once E1 passes; E6-E8 depend on them. The first generalization study's environments: Lua and Python
execution, novel document-rule systems, stateful database workflows, repository mutations, the firmware emulator, and a
synthetic scientific world with hidden parameters (graded on predictions under unseen conditions; tests learning from
experiments, not imitation). Practice environments are generated independently of Bench's sealed collections.

## A LearningCampaign's required resources

Goal and competence criteria; knowledge sources; tool registry; isolated environment; ground-truth mechanism; training-feedback
policy (what diagnostics may be shown); task generator; learning-strategy policy; experience and skill store; compute and cost
limits; independent development validation; sealed examination authority; retention and rollback; durable campaign state. The
campaign may construct missing pieces (find a compiler and a test suite in a repository), but a constructed environment is
plausible, not proven, until it passes admission; it stops by itself when exploration is no longer informative, the verifier is
compromised, compute is insufficient, or learning causes more regression than gain.

## Target capability

"Learn to develop firmware for this unseen microcontroller family from its manuals, toolchain, emulator, examples and a
reference implementation; verify on unseen problems, keep what you learn, stop when the criteria are met or progress is no
longer justified." Success is firmware that passes independent execution, delivered increasingly cheaply with experience, not
an explanation of the manual. Every report separates scaffold, memory, weight and learning-process improvement.

## Sources to check before citing

Continual Learning Exam, SkillLearnBench, SkillGenBench, ContinualSkillBench, Agent Lightning technical report (see the source
list in `environment-supervised-acquisition.md`; none verified yet).
