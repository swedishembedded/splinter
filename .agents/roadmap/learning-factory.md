<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learning factory: one recipe from tasks to an evaluated candidate

Scope: the redesign that makes the installed `splinter` binary the one control plane that takes a recipe in a task repository from task loading through rollouts on any agent, verified trials, datasets, brain training and evaluation, with the experience database reduced to an optional index (milestones M0-M9 below).

## Target

```
splinter validate recipes/vim-mastery.yaml
splinter plan     recipes/vim-mastery.yaml
splinter run      recipes/vim-mastery.yaml --out runs/vim-mastery
splinter inspect  runs/vim-mastery --stage rollouts --viewer harbor
splinter inspect  runs/vim-mastery --stage dataset
splinter verify   runs/vim-mastery
splinter report   runs/vim-mastery
```

The run can be cancelled, resumed, extended with another rollout pass, train several
variants and evaluate them without invoking the task repository's CLI, Harbor, brain
or the experience database by hand. The default dataset path is: Harbor trial ->
validated `TrialResult` -> immutable trial collection -> projection -> brain training.

Existing interfaces may break where they block this; proven mechanisms are reused,
duplicate flows are removed after a parity test.

## Findings this plan rests on (checked against the code, 2026-11-22)

- `crates/orchestrator/src/pipeline.rs` is a linear stage list over shared state; it
  cannot express independent passes, reusable datasets and several candidates.
- `crates/orchestrator/src/jobs.rs` (idempotent submission, leases, reconciliation)
  is worth keeping, but it takes a `splinter_store::workspace::Workspace`, which sits
  on the experience database: the database is not optional today.
- `crates/pipelines/src/harbor.rs` `harbor_outcome` treats reward exactly 1 as pass,
  any other number as fail, a missing reward or an exception as abstain. The task
  repository's `benchlib/score.py` distinguishes model-server, sandbox, verifier and
  budget failures. Two classifiers for one question; only one may survive (M5).
- `crates/data/src/curation.rs` defaults to `HistoryCuration::DropStep`: a safety
  admission and a behavioural curation in one switch (M6 splits them).
- The task repository's `benchlib/adopt.py` admits Sven jobs only; Splinter's
  `crates/agent` runs Sven and Claude Code. Agent independence is M4's acceptance test.
- brain already resumes fine-tuning exactly (`brain/crates/sdk/src/finetune.rs`,
  resume state) and runs typed training jobs (`brain/crates/trainjob`); Splinter uses
  them through brain's API, never in process.
- Licences differ: Splinter is Apache-2.0, the task repository's `benchlib/` is
  proprietary (decided below).

## Rules that change with this plan

- `AGENTS.md` and `.agents/rules/architecture.md` give sven sole ownership of running
  agents. Under this plan Splinter schedules executions through an execution backend
  (Harbor first, through a version-pinned worker process on Harbor's public job API);
  each agent, Sven included, still runs itself. The ownership table is rewritten in
  the M4 commit, not before.
- The experience database loses authority over run state and product reuse (M1).

## Milestones

Built as vertical slices: the first slice is one vim task -> recipe -> Sven through
Harbor -> native ATIF and trusted score -> success-only SFT -> brain LoRA ->
development eval; later slices add an independent agent, a second pass, checkpoint
recovery and protected eval.

| # | Work | Acceptance | Status |
|---|---|---|---|
| M0 | Versioned contracts: recipe, `TaskCollection`, `ExecutionSpec`, `TrialResult`, `DatasetProduct`, `ModelCandidate`, `EvalResult` (JSON Schema, Rust type, canonical serialisation, golden fixtures) | fixtures validate and round-trip | done: `splinter-core::factory` contracts, generated schemas (`make schemas/generate`), golden fixtures |
| M1 | `ArtifactRepository` and `RunRepository` (SQLite WAL, one controller) with no experience-database import; the database becomes an `ExperienceIndex` | run completion and recovery with the database disabled | done: `splinter-store::factory` (SQLite run state with fenced leases, release and take-over; CAS product store with import); no experience-database import |
| M2 | Persistent product DAG; a stage's identity digests its schema, resolved parameters, upstream product digests, implementation digest and seed; crash-safe publication | unchanged inputs are skipped, changed inputs recompute only downstream | done: `splinter-factory` plan, stage identities, one commit path, fork, verify, status |
| M3 | Task loader for Harbor task directories and collections, environment/knowledge/skill staging from pinned artifacts | the installed binary runs task-repository material with no rebuild | done: collections over Harbor task directories (`generate:` and `splinter generate`), environments, pinned task and file copies; the task repository holds the vim course |
| M4 | Harbor worker package, argv-only agent invocation, `connections.yaml` model routing | Sven and one other agent run one collection through the same trial contract; `harbor view` opens the trials | done: `workers/harbor` (run, reconcile, play; sven, claude-code, oracle, nop; userns, proot, docker); live: oracle and Sven on a hosted model under proot. Not run here: userns, docker, a real Claude Code |
| M5 | One outcome classifier (classification separate from score), logical slot vs physical attempt, `unsolved_by` passes frozen at pass start | no valid graded result lost; a retry never repeats a completed slot | done: the one classifier (the task repository's 30 golden records), slots and attempts, retries, lost attempts, frozen `unsolved_by`, crash reconcile |
| M6 | Dataset from trial collections: `native-sft` first, `portable-sft` second, admission split from curation, `supervision.jsonl` lineage and the exclusion waterfall | a training dataset built with no experience import | done: native-sft (a live Sven 5.9 trial is a fixture) and portable-sft (rebuilt from the ATIF trajectory) |
| M7 | Training variants through brain's job API, candidate identity, resume, automatic serving and a load check before a candidate counts | an interrupted variant resumes; a completed one is reused | done: `Trainer` seam, `BrainTrainer` against a scripted brain service (resume key, served adapter, refused parameters). Not run here: a real brain |
| M8 | `splinter eval`: development and protected plans, the existing examination, verdict and signing logic behind its own authority | base and candidates evaluated under the same conditions | development plans done (shared baseline, paired sign test, alpha split among candidates); protected plans: Splinter side done (authority command, Ed25519 check), the task repository's `evaluate` command open |
| M9 | Retire the duplicate controllers (`learn`, `practice`, `generations`, `campaign` as separate lifecycles; the task repository's run/adopt/export/profile path), fault-injection suite | every test in the plan's section 19 passes | partial: see Open |

## Decisions

- 2026-11-22: code moved from the task repository (classifier, staging, packing,
  examination) is relicensed to Apache-2.0 by its copyright holder and lives in
  Splinter; the moved files carry Splinter's header.
- 2026-11-22: verification is model-free. Unit and integration tests in this
  repository prove every stage with stub models and stub agents; a live end-to-end
  run with a real model follows the work. Live checks are kept small: Sven on the
  hosted service's cheap models and Claude Code as the independent agent, only to
  show a path works.
- 2026-11-22: old paths are deleted once a parity test passes (Splinter's `learn`,
  `practice`, `campaign`, `generations` lifecycles; the task repository's run,
  adopt, export, profile and gateway commands); samples are ported to recipes,
  papers keep their recorded results.
- 2026-11-22: interactive play moves to `splinter play`, an interactive rollout
  pass whose trials land in the run's collection like any other.
- 2026-11-22: the task repository is restructured into the target layout by
  moving its files.
- 2026-11-22: YAML for every factory file (recipes, connections, environments,
  agent presets, eval plans); a TOML file the factory reads is converted.
- 2026-11-22: commits stay local; nothing is pushed.

## Design

| Piece | Home |
|---|---|
| Contracts, outcome classifier, stage identity | `splinter-core::factory` (pure, no I/O) |
| JSON Schemas and golden fixtures | `schemas/factory/`, `crates/core/tests/it/factory/` |
| Run state (SQLite WAL, one controller) and the product store (filesystem CAS, atomic manifests) | `splinter-store::factory` (no experience-database import) |
| Recipe loading, connections, planning, controller, stages | `splinter-factory` (application tier) |
| Harbor worker | `workers/harbor/` (Python, Harbor's public job API, pinned Harbor) |
| CLI | `splinter validate/plan/run/status/inspect/cancel/resume/verify/eval/fork/index/export/report/play` |

## Open

- **Retirement (M9).** The task repository's run, stage, adopt, task play and task
  replay commands are retired. Its `gateway serve`, `export`, `unpack`,
  `task practice` and `task session` stay: Splinter's practice, campaign and hinted
  repair drivers (`crates/pipelines/src/bench/practice`, `bench/repair`,
  `campaign/`) still run them as subprocesses, and those flows (multi-generation
  campaigns, hinted repair, the request-based views beyond native-sft) have no
  factory equivalent yet, so under the parity rule they are not deleted. Next:
  re-express campaigns as forks of recipe runs and hinted repair as a pass mode,
  then delete the drivers and the commands they need.
- **`splinter index RUN`** (the experience database as an optional index over a
  run's trials) is not implemented: the Harbor task import it reuses needs a
  sandbox this container cannot start, so it could not be tested here.
- **Protected evaluation authority**: the task repository needs an
  `evaluate --plan ID --candidate FILE --out FILE` command signing a
  `splinter.eval/1` result the way it signs verdicts.
- **Sven trajectories** (5.9.2): a reply's reasoning and a request's token account
  are filed as agent steps of their own after the reply's tool-call step; native-sft
  reads both shapes, and the cause in sven's step assembly is not yet found.
- **Live end-to-end with a real model and brain** (the decision of 2026-11-22): a
  full `recipes/vim-mastery.yaml` run on a GPU host.
