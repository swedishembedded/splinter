<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Factory operators: one engine, every learning method an operator

Scope: after the learning factory (`learning-factory.md`, M0-M9), dissolve the
remaining workflow engines (`learn`, `practice`, `harness optimise`) into operators of
the factory's one DAG engine; give the factory adaptive sandboxing, a native execution
backend beside Harbor, external operators, recipe templates, Sven harnesses as an
execution strategy, harness search and an automatic generations loop. Milestones
O1-O8 below.

## Principles

- **Splinter runs learning recipes; everything else is an operator.** One controller
  (`splinter-factory`), one product store, one trial contract (`TrialResult`). An
  operator never owns a run lifecycle: it is a stage kind, or a policy a stage takes.
- **A task defines success, a harness organises the agent's work, the independent
  verifier decides.** Instruction format, execution strategy and verification are
  three separate declarations.
- **Splinter controls the outer execution contract** (model binding, agent build,
  sandbox, time/cost/resource bounds, reward, release); a harness or operator may
  narrow it, never widen it. A conflict is refused at plan time.
- **Sandboxing adapts, never silently weakens.** The strongest provider the machine
  supports that enforces every requirement is chosen; what was enforced is recorded
  per trial; an environment that demands more than the machine can give is refused.
- **Algorithms are kept, orchestration is deleted** - and only after the operator that
  replaces a verb passes regression tests against representative workloads. No
  aliases or compatibility wrappers survive the deletion.

## Findings this plan rests on (read in the code, 2026-11-23)

- `learn` (`crates/pipelines/src/learn/`, 22 stages) and `practice` (`practice/`, 14
  stages) each run the linear `splinter_orchestrator::pipeline` with a mutable state
  struct as their implicit DAG. Shared steps: policy, `solving::solve_tasks`,
  `verify::verify_set`, `curriculum::frontier`, teacher, `datasets::build`,
  `train::train_recorded`, `release::release_recorded`, a family reserve.
- `learn` tasks are mostly closed-book (`recall`, `taught`, `arithmetic`, `format`),
  answered by the policy directly with no agent or sandbox, and verified by code or a
  calibrated judge of another identity (`kind_needs_judge`). They cannot be Harbor
  tasks without contorting either. `practice` tasks are workspace changes verified by
  hidden tests and already export to Harbor (`knowledge/src/tasks/workspace/harbor`).
- Transcript continuation (`agent/src/correction.rs`, `sven_sdk::fork`) replays a
  student's actions into a fresh tree and refuses unless every observation, the
  reached workspace and the transcript digest match; harness runs
  (`agent/src/harness_work.rs`) and harness search (`harness_opt`, `sven_sdk::search`)
  are in-process through Sven's SDK.
- `harness_opt` applies a harness only to workspace tasks, keeps only the champion's
  YAML and moves an opaque `harness/champion` pointer.
- `crates/sandbox` already confines with Landlock (filesystem, TCP connect with a
  single-port hole), seccomp (no `AF_INET`/`AF_INET6`/`AF_PACKET` sockets, so no UDP)
  and user/pid namespaces where the host allows them, and degrades by `Need`.
- Here (a container): Landlock ABI 8 and seccomp available, `no_new_privs` set, no
  user namespaces (`unshare`, `bwrap` refused), a Docker client without a reachable
  daemon, `proot`. On a full machine: Docker and user namespaces.
- Sven harness v2: `sven_sdk::compile_harness` gives a plan with a digest;
  `sven harness run` runs it headless (`--trajectory`, `--trace`, exit codes 0-5);
  workflow-added user turns carry `extra.sven.origin` (`task`, `follow_up`, `repair`,
  `correction`); gate output is not a trajectory step.

## Architecture

### Execution: two backends, one trial contract

The factory's `ExecutionBackend` seam gets a second implementation:

| Backend | Runs | Isolation | Verification |
|---|---|---|---|
| `harbor` (exists) | Harbor task directories, any agent (Sven, Claude Code, Codex, oracle) | the worker's sandbox, chosen adaptively | the task's own `tests/` |
| `native` (new) | Splinter-native tasks: closed-book answers, runtime code, workspace changes; Sven in process (harness, fork/continuation) | `crates/sandbox`, chosen adaptively | code verifiers, hidden-test workspace checks, calibrated judges |

A task in a `TaskCollection` names its execution class (`harbor` or `native` with its
kind) and its verifier class; a rollout pass is dispatched to the backend that serves
its tasks' class, and refused at plan time for an agent the backend cannot run. Both
backends write `AttemptReport`s classified by the one classifier into `TrialResult`s;
the native backend's trials carry ATIF and, for workspace tasks, the patch.

### Adaptive sandboxing

- An environment states requirements, not a provider: writable paths, read-only
  mounts, network (`none`, `model-gateway-only`, `open`), resources, and a minimum
  `isolation` (`strong`: process and filesystem isolation; `confined`: filesystem and
  network confinement; `any`). `provider: auto` is the default; naming one pins it.
- The worker (and the native backend through `crates/sandbox`) probes once per job:
  Docker daemon reachable, user namespaces, Landlock ABI, seccomp, proot. It picks the
  strongest provider that enforces every requirement: `docker` > `userns` >
  `confined` (proot for the task image's root plus Landlock and seccomp applied by
  `splinter-confine`, a small launcher from `crates/sandbox`: writes only to the
  workspace, logs and tmp; read-only mounts read-only; `model-gateway-only` as TCP
  connect to the gateway port and no UDP or raw sockets) > `proot` (open network only).
- The trial records the provider and the guarantees enforced (`evidence.sandbox`); a
  recipe sees the choice in `splinter plan`. `isolation: strong` refuses `confined`.

### Operators

| Kind | Where in a recipe | Built-in operators (origin) |
|---|---|---|
| task source | `tasks.generate: {operator, ...}` | `harbor-collection` (exists); `repository-mutations` (practice: snapshot, baseline qualification, mutation/history tasks, hidden tests, family reserve); `grounded-questions` (learn: sources, plan, grounded task kinds, exam reserve); `variants` (learn: held-out rewordings, measured never trained) |
| task selection | a pass's `tasks:` | `unsolved_by` (exists); `frontier` (pass@k); `learning-progress`; `signal`; `quota` (curriculum/) |
| repair | a pass's `repair:` over another pass's failures | `hint` (exists); `continue` (verified transcript continuation, practice corrections); `critique` (critique and retry); `teach` (open-book teacher with privileged material, or reference demonstration); `decompose` (subgoals with checkable states) |
| dataset strategy | `dataset.projection` / `objective` / `mix` | `native-sft`, `portable-sft` (exist); `sft-step`; `policy-gradient`; `self-distill`; `rehearsal` mix; hint sufficiency and re-grading filters |
| evaluation | `eval` | development and protected plans (exist); retention suites; powered family exam; checkpoint selection on dev |
| experiment | top-level sections | `harness_search`; `generations` |

**External operators.** A task-source, selection or dataset operator may be an
executable plugin: `operator: external`, `program: [argv]`, declaring the versioned
contract `splinter.operator/v1` (JSON request on stdin naming the input products and
parameters; JSON response naming the output products it wrote under its output
directory). Splinter validates both against the contract's schema, runs the program
in the adaptive sandbox with only its inputs readable and its output directory
writable, and commits the output as an ordinary product. Bench's task generators
(vim, KiCad, CAD, firmware mutations) become external operators: no Splinter rebuild
to change which a recipe uses.

**Templates.** `splinter templates list` and `splinter init --template NAME --out
FILE` write ordinary recipes (`agent-sft`, `repository-practice`,
`knowledge-learning`, `harness-search`, `generations`). A template has no behaviour a
recipe cannot express.

### Sven harnesses

- **Task instruction descriptor.** A Harbor task's `task.toml`
  `[metadata.splinter.instruction]` (or the native task's field) declares
  `format = "markdown"` (default, `instruction.md`) or `"sven-harness"` (a harness
  YAML with its inputs). A harness-format task needs a harness-capable agent; any
  other is refused at plan time.
- **Recipe-level harness** (preferred): a rollout pass's `harness: PATH` or
  `harness: artifact:STAGE` runs the agent under a reusable harness, so the same tasks
  are compared across harnesses and agents.
- **Plan time:** the harness is compiled with Sven's compiler (`compile_harness`);
  the plan digest and every imported file join the pass's identity; the outer
  contract is checked: the harness's `agent.profile`, tools and `execution.limits`
  must be within the recipe's agent, sandbox and bounds.
- **Run time:** the Sven adapter runs `sven harness run` headless in the trial's
  sandbox (Harbor) or `harness_work` (native); the task's verifier grades the final
  state independently. Harness gates are agent-visible feedback, never scoring
  evidence; a gate that reads hidden test material is refused.
- **Projection:** harness trajectories keep every element; standard SFT supervises
  only agent-generated messages and tool calls; harness instructions, repair prompts
  (`extra.sven.origin`) and gate output visible to the agent are context; control
  events are not in the record; the plan digest is in each record's lineage.
- **Harness search** (`harness_search:`): `harness_opt`'s mutation space, model
  proposer, search strategies, paired family comparison and evidence gate, as a stage
  whose products are content-addressed `HarnessCandidate`s (harness, plan, trials,
  comparison) and a result naming the best; promotion beyond the development set
  needs a held-out evaluation. `harness: artifact:<stage>` consumes it.
- **Model x harness matrix:** an evaluation may name several models and harnesses;
  every combination runs under the same tasks and seeds and the report separates the
  harness effect, the model effect and their interaction.

### Generations

`generations: {max, collect: {selector}, dataset: cumulative, rehearsal, eval:
{development, retention, release}, stop: {no_improvement: N}}` makes the controller
plan generation g+1 once generation g's release is committed: its stages live under
`g<NNNN>/`, its policy is g's champion (`champion:` resolved within the run), its
collection selects by the configured selector, its dataset accumulates every
generation's verified records, and the loop stops at `max` or after N generations
without improvement. Each generation's products are immutable; the strategy for a
generation (more rollouts, another agent, a harness search, repairs, training) is
the recipe's, never inferred from a failure.

### CLI after the migration

`init`, `templates`, `validate`, `plan`, `generate`, `run`, `resume`, `cancel`,
`status`, `inspect`, `fork`, `play`, `eval`, `report`, `verify`, `export`, `index`.
`learn`, `practice` and `harness optimise` are deleted after O4-O6.

## Milestones

| # | Work | Acceptance | Status |
|---|---|---|---|
| O1 | Adaptive sandboxing: environment requirements, `provider: auto`, capability probe, `splinter-confine` (Landlock + seccomp over proot), per-trial record | the vim recipe runs here (`confined`) and on a full machine (`docker`/`userns`) unchanged; a requirement no provider meets is refused | |
| O2 | Native execution backend and task classes in `TaskCollection`; dispatch by class | a closed-book and a workspace task run through the factory into `TrialResult`s | |
| O3 | Operator framework: `tasks.generate`, built-in registry, external operators (`splinter.operator/v1`, sandboxed), templates and `splinter init` | a Bench generator runs as an external operator; templates produce valid recipes | |
| O4 | Repository practice as operators: `repository-mutations`, family reserve, `frontier`/`learning-progress`/`signal` selection, `continue` repair, `sft-step`/`policy-gradient`/`self-distill`; regression against `practice`'s tests; delete `practice` | the practice regression workloads give the same tasks, verdicts and datasets | |
| O5 | Knowledge learning as operators: `grounded-questions`, judged verification, `teach`, `critique`, `variants`, `quota`, `rehearsal`, checkpoint selection, powered exam; regression against `learn`'s tests; delete `learn` | the learn regression workloads give the same admitted tasks and gates | |
| O6 | Sven harnesses: instruction descriptor, recipe `harness`, outer-contract checks, harness projection; `harness_search` with `HarnessCandidate` products; model x harness matrix; delete `harness optimise` | one task under plain Sven, a harness with verified repair, and Codex gives compatible trials | |
| O7 | `generations` section with selectors, cumulative datasets, rehearsal, retention and stop rules | a scripted three-generation run promotes, keeps and stops as specified | |
| O8 | Remaining repair modes (`decompose`, knowledge-backed corrections, hint sufficiency and re-grading) and the documentation of the whole | each mode's regression test passes | |

## Decisions

- 2026-11-23: `learn`, `practice` and `harness optimise` are deleted as verbs after
  their operators pass regression tests; their algorithms are kept as operators.
- 2026-11-23: Sandboxing is adaptive with a recorded, never silently weakened choice.
- 2026-11-23: All four retired repair modes (fork corrections, knowledge-backed
  corrections, decomposition, hint sufficiency and re-grading) are ported (O4, O8).
- 2026-11-23: a connection may drop agent headers its endpoint refuses
  (`drop_headers`), for a hosted endpoint that refuses `anthropic-beta`.
- 2026-11-23: Markdown instructions stay the default; Sven harnesses are an optional
  execution strategy, preferably chosen per rollout pass.

## Open

- Whether `acquire` (`learn --source`, a separate competence-driven loop) becomes an
  operator in O5 or a later milestone.
