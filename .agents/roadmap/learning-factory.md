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
| M7 | Training variants through brain's job API, candidate identity, resume, automatic serving and a load check before a candidate counts | an interrupted variant resumes; a completed one is reused | done: `Trainer` seam, `BrainTrainer` against a scripted brain service (resume key, served adapter, refused parameters). Live (2026-11-23): a real brain on CPU trained Qwen3-0.6B LoRA r4 from two hosted-Sven trials, served the adapter, and the candidate was evaluated. `splinter index RUN` files a run's trials in the experience database as an optional index |
| M8 | `splinter eval`: development and protected plans, the existing examination, verdict and signing logic behind its own authority | base and candidates evaluated under the same conditions | development plans done (shared baseline, paired sign test, alpha split among candidates); protected plans done: the task repository's examiner (`bench examine`, its sealed tasks never leave it) examines the candidate, and its signed verdict counts only when the promotion gate's check passes |
| M9 | Retire the duplicate controllers (`learn`, `practice`, `generations`, `campaign` as separate lifecycles; the task repository's run/adopt/export/profile path), fault-injection suite | every test in the plan's section 19 passes | done within the scope decided 2026-11-23: `campaign` and `generation` are forked recipe runs held to a `champion:` model; hinted repair is a `hint:` rollout pass; Splinter's Bench practice and repair drivers and the task repository's run, stage, adopt, play, replay, gateway, export, unpack, practice and session commands are gone; section 19 below |

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
- 2026-11-23: M9 retires the flows that run on task-repository material
  (`campaign`, `generation`, the Bench practice and repair drivers); `learn`,
  `practice` (a repository's tests), `corrections` and `absorb` build their tasks
  from documents, repositories and sessions and stay outside the factory.
- 2026-11-23: trials run elsewhere join a recipe through `attempt.import`, an
  operation that runs nothing: the worker matches each Harbor trial to a pinned
  task by Harbor's own content digest of the task directory and reports it as a
  trial it ran, the controller grades it with the one classifier, and each job (a
  recipe input) is pinned by its tree digest. It replaces reading Harbor jobs
  through `experiences ingest-harbor`, which needs tasks imported into the
  experience database (refused for a task image built with `RUN`). Every factory
  record names its trial's stage, agent, model and import origin
  (`metadata.trial`).

## Design

| Piece | Home |
|---|---|
| Contracts, outcome classifier, stage identity | `splinter-core::factory` (pure, no I/O) |
| JSON Schemas and golden fixtures | `schemas/factory/`, `crates/core/tests/it/factory/` |
| Run state (SQLite WAL, one controller) and the product store (filesystem CAS, atomic manifests) | `splinter-store::factory` (no experience-database import) |
| Recipe loading, connections, planning, controller, stages | `splinter-factory` (application tier) |
| Harbor worker | `workers/harbor/` (Python, Harbor's public job API, pinned Harbor) |
| CLI | `splinter validate/plan/run/status/inspect/cancel/resume/verify/eval/fork/index/export/report/play` |

## Section 19 checks

Each of the plan's required tests, and where it is checked (factory tests are in
`crates/factory/tests/it/`, worker tests in `workers/harbor/tests/`).

| Check | Where |
|---|---|
| identical recipe twice reuses everything | `factory_run::a_recipe_runs_every_stage_once_and_a_second_run_reuses_every_product` |
| controller killed during execution | `factory_run::a_controller_killed_mid_rollout_resumes_without_repeating_a_finished_attempt`; worker `test_reconcile_*` |
| worker killed mid-trial | `factory_run::the_infrastructures_failures_are_retried_and_never_become_training_negatives` (a lost attempt is launched again); worker `test_reconcile_names_the_slots_whose_trial_never_finished` |
| reward-0 trial is a graded failure | `factory_run::a_recipe_runs_every_stage_once...` |
| model API failure is not a training negative | `factory_run::the_infrastructures_failures...`; core `classify::an_endpoint_that_answered_no_request...` (found live with a real Claude Code) |
| corrupt ATIF quarantined, evidence kept | `factory_run::a_corrupt_trace_keeps_its_evidence_and_is_quarantined_from_training` |
| Sven, Codex and Claude Code, one trial contract | worker `test_agents.py` (real Sven; scripted claude and codex); live: real Sven and a real Claude Code through the gateway |
| second pass runs only the frozen unsolved set | `factory_run::a_recipe_runs_every_stage_once...` |
| changed curation reuses rollouts | `reuse::a_changed_dataset_or_eval_plan_reuses_what_is_upstream...` |
| changed LoRA rank: new candidate, rest reused | `factory_run::a_changed_variant_retrains_only_itself...` |
| brain killed after a checkpoint resumes | `brain_trainer::a_variant_is_a_resumable_brain_job...` (resume key; brain resumes exactly) |
| changed eval plan reuses the candidate | `reuse::a_changed_dataset_or_eval_plan...` |
| experience database disabled | the factory has no experience-database dependency (`splinter-factory` imports no `expdb`) |
| experience database index rebuilt | `pipelines factory_index::a_factory_runs_trials_are_indexed_as_graded_experiences_once` |
| extra skill/knowledge files pinned and audited | worker `test_arguments_files_prompt_and_environment_reach_the_agent...`, `test_an_agent_file_that_does_not_match_its_digest_is_refused`; mounts pinned by tree digest |
| protected eval material never reaches training | `resolve::examination_tasks_never_reach_a_dataset_under_any_name`; `factory_run::a_protected_plan_keeps_the_examiners_verdict_only_when_it_verifies` |
| archived trial opens in Harbor's viewer | worker `test_paths_are_relative_to_the_output_directory_and_the_job_is_viewable` |
| run copied to another machine verifies | `reuse::...a_copied_run_verifies_alone`; `factory_run::a_tampered_product_fails_verification` |
| remote endpoint lost: bounded retries, explicit records | worker `test_an_unreachable_upstream_is_a_502_record`; `factory_run::the_infrastructures_failures...` (retries, then kept as what it was) |
| two controllers on one run | `factory_run::a_run_refuses_changed_inputs_and_a_second_live_controller` |

Not run here: userns and docker sandboxes (no user namespaces, no daemon), a real
codex binary, a GPU run of the full recipe. A real Claude Code reached the hosted
endpoint, which rejects its `anthropic-beta` header; that endpoint's limit, recorded
and classified as a model error.

## Open

- **Full GPU run** of `recipes/vim-mastery.yaml` (or the campaign recipes) on
  Qwen3-4B: this host has no GPU; the CPU run on Qwen3-0.6B is the evidence.
- **Not ported from the retired repair drivers:** fork corrections from a
  student's last good step, knowledge-backed corrections, decomposition into
  subgoals, and the hinted retry's opt-in sufficiency run and re-grading. The
  hinted pass carries the default rationalisation (hint shown, learned without it,
  quoting excluded). A campaign's frontier is a pass's `unsolved_by`, not a
  pass-rate threshold over earlier generations.
- **Hosted endpoint and Claude Code:** the hosted endpoint rejects Claude Code's
  `anthropic-beta` header, so a real Claude Code trial there is a model error (as
  classified); the endpoint, not Splinter, would have to accept or drop it.
- **`ReplayPolicy::{FailureTargeted, LearningProgress}`** in `splinter-data` are
  unreachable from the CLI since `generation run` went; kept so recorded ledgers
  still read.
