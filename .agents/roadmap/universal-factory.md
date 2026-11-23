<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Universal factory: one typed artifact graph for data, learning and evaluation

Scope: generalise the learning factory (`learning-factory.md`, M0-M9 done) from a
fixed six-stage recipe into a typed, composable artifact transformation graph that
ingests sources, builds knowledge, tasks and environments, runs agents and harnesses,
projects datasets, trains through Brain and evaluates - with `learn`, `practice`,
`harness optimise`, campaigns and the task repository's CLI dissolved into operators
and recipes. Milestones U0-U9 below. Supersedes the operators plan of the same day.

## Boundaries

**Splinter orchestrates transformations, Sven executes agent behaviour, Brain performs
model computation.** The factory is extended, never paralleled.

Invariants (each enforced by a test or a dependency check):

1. One durable controller: `splinter-factory` owns recipe execution and stage state.
2. One agent implementation: Sven owns conversations, tools, harnesses, trajectories.
3. One model-computation owner: Brain owns inference, optimisation, model arithmetic.
4. Products are immutable; no stage edits a committed artifact.
5. Every operation declares typed inputs and outputs; unknown parameters are errors.
6. Identity includes every behaviour-changing input; an unpinned model is marked
   non-reproducible, never cached as if it were.
7. Evidence is separate from claims: a generated answer is not its own verification.
8. The experience database is an optional index rebuilt from manifests, never state.
9. Protected evaluation stays protected: no generation or training stage reads it.
10. One implementation per mechanism: no second scheduler, harness interpreter,
    training controller, dataset builder, intervention path or release gate.
11. A task defines success, a harness organises the agent's work, an independent
    verifier decides; harness gates are agent feedback, never scoring evidence.
12. Sandboxing adapts and is recorded; the least isolation is limiting the files and
    network a sandbox reaches; strong isolation only where an environment demands it.

## Verified starting point (2026-11-23)

- `crates/factory`: recipe (`splinter.factory/v1`: tasks, rollouts, dataset,
  training, eval, release), `StageKind` fixed in `splinter-core::factory::identity`,
  plan, resolve and lock, controller with leases, product CAS, slots and attempts,
  `champion:` generations, `hint:` passes, protected plans.
- `crates/knowledge`: capture and sections, hybrid retrieval, claims, concepts,
  skills, objects, task generation, and `factory/` (4.5k lines): procedural
  generator plugins (`plugin.rs`), admission (reference, no-op, reproducibility,
  novelty, red-teaming, success bands), verifier classes, `DivergenceAlarm`.
- `crates/agent`: `typed.rs` (bounded, schema-checked calls), `harness_work.rs`
  (a Sven harness in a confined workspace), `correction.rs` (verified transcript
  continuation via `sven_sdk::fork`).
- `crates/data`: views over experiences (SFT, preference, feedback, verifier,
  retrieval, policy gradient, denoising, continued pretraining).
- `crates/pipelines`: `learn` (22 stages), `practice` (14 stages), `harness_opt`, each
  on the linear `orchestrator::pipeline`; `acquire` (`learn --source`).
- Brain: `crates/trainjob` (pretrain, check_dataset, fine-tune jobs, resume).
- Sven: harness v2 (`compile_harness`, `Engine::load_harness*`, checkpoint/resume,
  `HarnessOutcome`, ATIF with `extra.sven.origin` on workflow-added turns).
- The task repository: tasks, collections, environments, recipes, and a `./bench`
  CLI still used for `task compile` (collection generation) and `examine` (protected
  evaluation authority).

## Architecture

### Artifacts, operations, policies

Three separate concepts:

- **Artifact kinds** (what data exists): `documents`, `knowledge`, `index`, `tasks`,
  `environments`, `attempts`, `dataset`, `model`, `evaluation`, `harness`, `files`.
  Domain schemas live inside an artifact's contract, never as new Rust kinds. A
  manifest is today's `ProductManifest` extended with `kind`, its schema, its inputs
  and metadata (provenance, licence, partitions, reproducibility); the CAS stays.
- **Operations** (what computation): a registered descriptor - name and revision,
  typed input and output ports, a parameter JSON Schema, determinism and caching
  policy, resources, capabilities, implementation identity, a conformance fixture -
  and an implementation that only computes: it receives frozen inputs and writes
  into a staging area; the controller keeps leases, retries, budgets, progress and
  publication. Implementations: native Rust; `agent` (typed call); `harness` (Sven
  v2); `external` (a pinned executable under `splinter.operator/v1`: JSON request
  naming input products and parameters, JSON response naming the products written,
  run in the adaptive sandbox with only its inputs readable); Brain jobs.
- **Execution policies** (where, how often, under what limits): backend, sandbox
  requirements, model roles, attempts, retries, `foreach` sharding, budgets.

### Recipe v2

```yaml
schema: splinter.recipe/v2
name: NAME
inputs:  { corpus: { kind: files, path: ./books } }
models:  { generator: "hosted:deepseek/deepseek-v4.1-flash", policy: "brain:Qwen/Qwen3-4B" }
stages:
  - id: ingest
    uses: source.ingest
    inputs: { files: $inputs.corpus }
    with: { split: section }
    outputs: { documents: documents }
  - id: knowledge
    uses: agent
    foreach: $stages.ingest.documents
    inputs: { document: $item }
    with: { instruction: instructions/extract.md, model: generator, output_schema: schemas/knowledge.json }
    outputs: { records: knowledge }
  - id: book
    uses: recipe
    with: { recipe: recipes/fragments/tasks-to-sft.yaml }
    inputs: { tasks: $stages.admission.accepted }
outputs: { sft: $stages.book.sft }
```

- The compiler resolves references (`$inputs.x`, `$stages.id.port`, `$item`), expands
  subrecipes into a namespaced graph (no nested controller), type-checks ports,
  rejects cycles, unknown operations or parameters, unbound model roles, unbounded
  iteration, invalid harnesses, missing capabilities and protected-data flows, and
  pins every file, harness, executable and imported recipe in the lock. One compiled
  plan serves execution, `splinter plan`, `inspect` and reproducibility checks.
- Model roles bind `connection:model` references (and `champion:RUN`); the compiler
  pins them; a stage names a role, never a raw model.
- In v2 each candidate's evaluation is a stage of its own, so the significance level's
  split among candidates is stated (`comparisons`), no longer inferred from one eval
  section; the baseline arm several evaluations share is one product because its
  identity is the same (plan, agent, baseline, seeds), with no special case.
- `splinter.factory/v1` is replaced outright: today's six sections become stages
  (`tasks.pin`, `attempt.collect`, `dataset.project`, `model.train`, `evaluation`,
  `promotion`), the task repository's recipes are rewritten, v1 is deleted.

### Execution

- **Stage identity** stays `H(contract, operation, parameters, inputs, implementation,
  seed)`, extended with every behaviour-changing dependency (model revision, harness
  plan digest, tool configuration, image, embedding model, trainer version).
- **Collections and `foreach`:** a frozen collection, stable item identities, bounded
  shards dispatched through the operation's backend, a per-item work ledger in the run
  state (logical item vs physical attempt, as slots today), item-level reuse when one
  input item changes, a deterministic reduction into one output manifest. A rejected
  item is recorded; a lost worker's item is retried.
- **Backends:** the Harbor worker (task directories, any agent) and native execution
  (closed-book, runtime and workspace tasks, typed calls, harnesses, fork/continuation,
  through `crates/sandbox`). Exactly-once *committed publication* per logical item;
  side effects are idempotent by identity, not claimed exactly-once.
- **Sandboxing (done, see below):** requirements, not providers; the strongest the
  machine has; recorded per trial; reads, writes and the network limited at least;
  `isolation: strong` where an environment demands process isolation.

### Agents and harnesses

- `agent` stage: one bounded structured call (`typed.rs`), schema-validated, with
  correction. `harness` stage: `harness_work` generalised to `HarnessRequest {harness,
  inputs, mounts, model_roles, environment, limits}` -> `HarnessResult {outcome,
  outputs, trajectory, checkpoint}`; outputs are read only from declared paths or
  publications, validated, size-limited, then committed by the controller.
- `attempt.collect` takes an optional `harness:` (a path or `artifact:` reference)
  per pass; tasks carry an instruction descriptor (`markdown` default,
  `sven-harness`); the outer contract (model, agent build, sandbox, bounds) bounds the
  harness's profile, tools and limits; conflicts fail at plan time.
- **Interventions as data:** `InterventionPlan {task, policy, prior_attempt,
  instruction, resume_point}` - retry, critique, hint (exists as `hint:`), teacher
  correction, verified fork continuation, knowledge-backed correction, decomposition
  - all executed by the one attempt mechanism, with one tool profile, recording format
  and leak check.
- **Interventions, as built (U5):** every repair mode is an `attempt.collect` stage over
  the tasks earlier stages left unsolved (`unsolved_by`), differing only in what the
  actor is shown; the hinted mechanism (shown material, learned under the task's own
  instruction, a trial that repeats the material excluded) is the one filter chain.
  Retry: a new seed or model. Hint: `hint: solution`. Knowledge correction and
  decomposition: `hint: brief` with a per-task brief from a `briefs` input (an `agent`
  stage over `attempts.failures`, the unsolved tasks with their failed trajectories,
  asked for passages or a subgoal plan). Hint sufficiency and re-grading: an `oracle`
  attempt stage proves a task's reference solution passes its verifier before a hint
  from it is trusted. Fork correction (a teacher continuing the student's failed
  trajectory from a kept prefix) needs the prefix's tool calls replayed in the fresh
  sandbox before the conversation continues: sven's `--load-trace PREFIX
  --rerun-toolcalls` does, and the guidance is the continuation's user turn.
- Harness search is an operation producing content-addressed `harness` artifacts with
  their evaluations; a model x harness evaluation matrix separates the effects.

### Knowledge, tasks, environments

- `source.ingest` (files, directories, pinned snapshots; layout-preserving document
  adapters with page, heading, table, equation and figure records and extraction
  confidence), `knowledge.extract|relate|merge|index|query` over `splinter-knowledge`,
  `task.generate` (procedural plugins with byte-identical seeds, and LLM synthesis,
  each declaring its reproducibility), `task.admit` (the environment factory's
  admission and verifier red-teaming, plus source grounding), `environment.build|audit|
  qualify`. Three distinct decisions: task admission, attempt classification, dataset
  admission. Verifier classes (objective, proxy, judge, human) stay explicit.

### Datasets

`dataset.project` with one typed option struct per projection (`Sft`, `Preference`,
`Feedback`, `PolicyGradient`, `Retrieval`, `Cpt`, `Rlvr`, `Distill`, `Replay`) over
one ordered pipeline: candidate selection, admission, curation and deduplication,
family/source-group partition, serialisation; every exclusion typed and counted.
Every supervised target records the context it was produced in and the context the
student is trained to predict it in (privileged context never silently supervised).
RLVR products are executable task collections with independent verifiers. Formats
Brain cannot yet train are added to Brain and `splinter-data`, never a Splinter
trainer.

### Learning, evaluation, promotion

`model.train` (Brain jobs), `evaluation` (development, protected authority, retention,
powered family exams, robustness), `promotion` requiring target improvement, held-out
transfer, retention and the divergence alarm. Continuous learning is a campaign that
runs the same recipe per generation with the previous champion and evaluations as
inputs; bounded local iteration lives inside an operation; no cycle in the graph.

### The task repository

Keeps tasks, collections, environments, knowledge, harnesses, recipes and its
protected material. Its CLI is removed: the collection compilers (vim and others)
become external task-source operators, the examiner becomes the protected-evaluation
authority operator (its sealed tasks never leave it), the remaining commands
(inventory, coverage, suites, cache, pack, keys) become factory operations or are
deleted. Splinter runs everything.

### CLI

`init --template`, `templates`, `validate`, `plan`, `run`, `resume`, `cancel`,
`status`, `inspect`, `fork`, `play`, `eval`, `report`, `verify`, `export`, `index`.
`learn`, `practice` and `harness optimise` are deleted; their behaviour ships as
templates (`knowledge-learning`, `repository-practice`, `harness-search`,
`agent-sft`, `generations`).

## Milestones

| # | Work | Acceptance | Status |
|---|---|---|---|
| S | Adaptive sandboxing | the same recipe runs on a full machine and a hardened container; choice recorded | done: `splinter-confine` (Landlock + seccomp launcher), worker `auto` with probe and `confined` provider, environments state requirements, per-trial `sandbox` record and report table, native process sandbox uses the private `/proc` only where the host allows (`require_private_proc` for untrusted code); live: real Sven on a hosted model ran `auto` -> `confined` here. Open: reaping a run's escaped processes without a pid namespace |
| U0 | Baseline and regression inventory | current factory, practice, learn, harness and Bench recipe behaviour captured as golden manifests | done: the reference recipe's observable products pinned (`tests/it/golden.rs`, `golden::bless` to re-pin); practice/learn/harness behaviour is pinned by their own suites until U8 replaces them |
| U1 | Artifact and operation contracts in `splinter-core` | schemas, golden fixtures, canonical serialisation, negative tests | done: `ArtifactKind`, `Port`, `OperationDescriptor` (`splinter.operation/1`), `splinter.stage/2` keyed by `operation@revision` |
| U2 | Recipe v2 compiler: references, ports, subrecipes, roles, static validation, extended lock | branching graphs, missing dependencies, type mismatches, cycles handled | done: `graph::compile_partly` reports every problem; a stage consumes only stages above it, so declaration order is execution order and no cycle can be written; subrecipes expand under the stage id; `foreach`/`$item` parse (U4 runs them) |
| U3 | Controller dispatches operations; v1 stages become operations; v1 and `StageKind` deleted; Bench recipes rewritten | every existing factory test passes through v2 | done: the controller dispatches on the operation name, parameters are typed per operation (`ops.rs`), `RolloutPass` is internal to `run/pass.rs`, lock `splinter.lock/v2`; the golden test passes unchanged; Bench's four recipes rewritten and validated; `play --pass` is `play --stage` |
| U4 | Collections, `foreach`, work-item ledger, shards, reduction, item-level reuse | fan-outs survive worker crashes; one changed item recomputes only itself | done: `execute::Execute` (an operation that only computes: frozen inputs in, one staged product out); recipe inputs pinned as `_inputs/NAME` (copied, checked against the lock); `foreach` commits each item as `_items/STAGE/ID` keyed by the item's content, at most `concurrency` (1-64) on scoped workers while only the controller thread touches run state and store; reduction = `splinter.items/1` index of produced and rejected items plus each output under its id. Tested with test operations: crash and resume, one changed item after a fork, bounded concurrency, refusals. No built-in computes yet: the first are U5 (`agent`) and U6 (external operators), so the user guide documents `foreach` with them |
| U5 | `agent` and `harness` operations; native backend; harness per pass; intervention plans (all repair modes) | a prompt, a harness and an executable task run with traces, limits, cancellation, publication | done except the native backend (closed-book tasks graded by Splinter's own verifiers, in process, beside the Harbor worker); it is what `task.generate`'s admitted closed-book tasks need, so it moves to U6 with it: `agent` done (`agent_op.rs` on `splinter_agent::schema_call`: instruction + JSON Schema files pinned in the lock and the identity, replies validated and corrected, never-valid input rejected, `output.json` + `call.json`; models reached in process through the host's `Models` with the connection's key, lazily, so a reuse reaches none; live on the hosted model with `foreach`, resume and verify). `harness:` per attempt stage done (pinned `AgentInvocation.harness`; the worker's sven adapter runs `sven harness run` with the instruction as `task`, through the gateway; plan-time refusals; native-sft over it refused). Interventions done: retry (a stage), hint (`hint: solution`), knowledge-backed and decomposition (`hint: brief` over `attempts.failures` + an `agent` teacher; the hinted repetition filter is the one leak check), sufficiency and re-grading (`agent: oracle` + `solved_by`). Fork correction: `continue_from` + fork points (`keep_through`, `guidance`) from briefs; the worker runs `sven --load-trace PREFIX --rerun-toolcalls` with the guidance as the prompt; the record drops the guidance turn (live: a failed vim attempt continued and learned). `harness` operation done (a tool-less harness over the stage's files, in process through `splinter_agent::harness_call`; one that needs a shell or command gate is refused and belongs on an attempt stage). |
| U6 | Knowledge, synthesis, environment operations; external operators; Bench compilers and examiner as operators; Bench CLI removed | a real document yields grounded knowledge and admitted tasks | in progress: external operators done (`external_op.rs`: `operators/*.yaml`, JSON Schema parameters, pinned `code` in the lock and identity, run in the process sandbox with read-only grants for inputs, request and repository, exit 0/3/other). `tasks.pin` pins a produced collection at run time (the factory keeps its pinned collections; run-time readers use them). Bench's compiler is the operator `collection.compile` (matches `bench task compile` byte for byte); products keep executable bits; an evaluation may take its tasks from a pin. Bench's recipes compile tasks/vim as their first stage (no collection file, no `./bench`). The examiner is an operator authority (`authority: { operator }`; machine grants in `~/.config/splinter/operators.yaml`: read, write, network, env by name; Bench's `benchlib.operators.examine`). The Bench CLI is removed (bench 31f5d60d: ./bench, cli.py, commands/ and every module only they reached; docs rewritten). Next: `source.ingest` and `knowledge.admit` done (documents to sections with provenance; proposed objects held to their sections by `splinter_knowledge::objects::admit`; live: a real guide section gave 13 grounded objects, none refused). `task.generate` and the native closed-book backend `answer.collect` done (the generator over a run-local source store; answers through the one solve, graded by `splinter_knowledge::tasks::verifiers::plain_verifier`, shared with the pipelines; live: a real guide section gave 8 admitted recall tasks, 2 refused for no subject, all answered and graded). `environment.build` done (an objective compiled by the environment factory, its admitted instances exported by split for `tasks.pin`). Verifiers that need more than the task and the answer done where they can run in process: each generated task's item keeps the document its evidence is in (`sources/`), so quotation and grounding checks read it; code kinds are generated and answered in their runtime in the process sandbox, graded by the executable and mutation checks (`evidence_verifier` and `runtime_verifier`, shared with the pipelines). A judge and a workspace are a run's own and stay with attempt stages. Remaining in U6: `knowledge.relate|merge|index|query` and `environment.audit|qualify`. |
| U7 | Typed dataset projections over `splinter-data`; privileged-context provenance; family splits | one source gives SFT, preference and RLVR products with no upstream rerun | |
| U8 | Training, evaluation, promotion operations; harness search; generations campaign; templates and `splinter init`; `learn`, `practice`, `harness optimise` deleted | a generated task collection trains through Brain's job API and promotion needs transfer and retention | |
| U9 | Cleanup, documentation reconciliation, fault injection, security tests, scenarios A-E | one implementation path; everything works with the experience database disabled | |

Scenarios: A document to SFT; B document to executable RLVR; C repository to agent
trajectories (SFT and preference); D full training loop with the release gate;
E interrupted, resumed and incrementally changed corpus. Live qualification uses
cheap hosted models; Brain integration is proven against the scripted Brain service
now and live on a GPU host later.

## Decisions

- 2026-11-23: `learn` and `practice` are deleted as verbs; their capabilities ship as
  operators and `splinter init` templates (no aliases).
- 2026-11-23: every execution adapts to the strongest sandbox available and records
  it; where the host is itself an isolating container, limiting what the sandbox can
  read and write is enough, generated code included: `confined` (reads limited to the
  system, the task's image and its mounts; writes and the network confined) is
  sufficient, and strong isolation is required only where an environment asks for
  it (`isolation: strong`). (Superseded the same day: generated code requiring strong
  isolation.)
- 2026-11-23: `splinter.recipe/v2` replaces `splinter.factory/v1` outright.
- 2026-11-23: live qualification uses cheap hosted models; Brain is validated against
  its scripted service until a GPU host is available.
- 2026-11-23: the task repository's CLI is removed; its generators and examiner run
  as external operators through Splinter.
- 2026-11-23: all retired repair modes are ported as intervention plans (U5).
- 2026-11-23: Markdown instructions stay the default; Sven harnesses are an optional
  execution strategy, preferably chosen per pass.
- 2026-11-23: a connection may drop agent headers its endpoint refuses
  (`drop_headers`).

## Open

- The `agent` operation's schema is in the brief and checked on every reply,
  but the provider is not asked to constrain decoding to it: sven's `Method`
  derives its response format from a Rust type. A runtime schema on `Method`
  (a sven SDK addition) would let it.
- Whether `acquire` (`learn --source`) becomes operators in U6 or later.
- Process cleanup without a pid namespace (two `process_runner` tests fail on such
  hosts); the fork-bomb test as root.
