<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Engineering cleanup: one path per mechanism

Scope: the technical debt the campaigns left in Splinter's code (parallel intervention paths, two multi-generation drivers, dataset options threaded through one request, the size of `splinter-pipelines`, hand-copied contracts, dead and experiment-named code), each with its target design, safety net, size and order.

Audit of 2026-11-21 against splinter eef918392 (read-only; every file and line below was read). Execution waits
for the conformance and mutation suites the item names under "Safety net": they are the net a refactor is held to.
Prefer deleting and consolidating over adding a layer; an abstraction is introduced only where the variation is real.

## Order

SC-1 (silent no-ops) first, alone, because each part is a defect today. Then SC-2 and SC-3, independent of each other.
SC-4 after SC-2 and SC-3 (it moves the files they rewrite). SC-5 after SC-4. SC-6 and SC-7 any time. SC-8 is
deferred: see "Not now".

## SC-1 Silent no-ops in the dataset build and the fork (S, open)

- **Problem.** `dataset build` has 36 flags, 21 of them documented "For the <view> view". `pipelines/src/datasets.rs`
  refuses most flags on a view they do not apply to (lines 240-320), but six reach no guard and are dropped without a
  word: `--only-actions` (read by sft-step and policy-gradient only), `--roles` (sft-step, policy-gradient, preference,
  feedback), `--writer`, `--token-limit`, `--max-family-share` and `--describe-with` (voice only). A critic or verifier
  build with `--roles main` trains on every role.
- **Problem.** `bench/repair/fork.rs:228` forks every student attempt under `Profile::Compact`, whatever profile the
  student ran. A task whose tools come from an MCP server offers no tool under compact (sven knowledge "a compact
  profile with an MCP-only allow list offers no tool"), so every fork correction of such a task continues with a tool
  set the student never had.
- **Target.** Each of the six flags refused on a view that does not read it, by the same `Refused` message the other
  guards use (SC-3 later makes this structural). The fork takes the profile the student's trial recorded (its
  `agent.profile` in the trial's sven configuration) and refuses a trial that records none.
- **Safety net.** A CLI test that passes each view-specific flag to every view that does not read it and expects a
  refusal (generated from `ViewName::ALL`). A `bench_interventions` spec whose scripted student ran the default profile
  and whose fork is checked to run it too.
- **Cost of leaving it.** Wrong training data with a clean manifest; repairs of MCP-tool tasks that can never pass.

## SC-2 One intervention interface and one execution path (L, open)

- **Problem.** Four implementations of "a model works a failed attempt again" with two execution paths:
  - `bench::repair` (`Intervention` trait, `mod.rs:343`): `ForkCorrection`, `KnowledgeCorrection`, `Retry`,
    `Decompose`, run in-process on sven's fork (`splinter_agent::fork_at`) inside the examiner's task session,
    records `splinter.bench-repair.v1`.
  - `bench::repair::hinted` (rationalisation): Harbor trials of staged task copies through the examiner's gateway and
    sven adapter, its own key, record and report formats (`splinter.bench-rationalise.v1`), its own filter chain
    (verified, real observations, no leak, sufficiency). `InterventionKind::HintedRetry` is in the enum the router
    uses, yet "no failure class routes to it" (`route.rs:47-53`): the enum describes two different things.
  - `curriculum::corrections` (`splinter corrections`, 396 lines) on `splinter_agent::correction` over repository
    snapshots, and `critique` (`splinter critique`, on `splinter_agent::repair`).
- **Cost.** The two paths diverge in what they can run: the in-process fork offers a fixed tool profile and no MCP
  servers (SC-1), the Harbor path runs whatever the task image declares. Filters exist on one path only: a
  knowledge correction shows the teacher reference material and is never checked for leaking it, which the hinted
  path's leak filter does (knowledge 200, 201). Every new intervention picks a path and copies its keys and records.
- **Target.**
  - An intervention is a plan, not an executor: `AttemptPlan { actor, prefix: Option<KeepThrough>, briefing,
    instruction: Original | WithHint(..) }`. `Retry`, `ForkCorrection`, `KnowledgeCorrection`, `Decompose` and the
    hinted retry each only build plans.
  - One `AttemptRunner` seam with two implementations where the variation is real: the examiner's trial (Harbor,
    with the student's profile and the task's MCP servers; a prefix replayed by sven's fork inside it) and the
    repository-snapshot runner `corrections` uses. Both return one graded attempt.
  - One filter chain (`verified`, `real_observations`, `no_leak`, `sufficient`) applied to every plan; a filter that
    does not apply to a plan says so in the record rather than being absent.
  - One record format and key derivation; `REPAIR_FORMAT` and `RATIONALISE_FORMAT` become one versioned format, the
    old records read by a migration in `splinter-store`.
  - `critique` stays: it is a prompt-feedback loop, not a fork; it moves onto `AttemptRunner` only if a test needs it.
- **Safety net (first).** An intervention conformance suite over the scripted bench double: for each intervention,
  the record key is stable across reruns, a decided record is never redone, a refused one stores nothing, an
  MCP-tool task runs under each intervention, the leak filter rejects a trace that names its hint. Golden record
  digests of today's `bench_interventions` and `hinted/filters_tests` specs, so the migration is checked byte for byte.
- **Depends on.** SC-1 (profile). Sven's fork must be able to replay inside a Harbor trial (the examiner's task
  session already hosts the fork's commands, `bench task session`).

## SC-3 Dataset build: per-view option types and composable curation filters (M, open)

- **Problem.** `BuildRequest` (`pipelines/src/datasets/request.rs`) is one struct of 20 fields plus `VoiceBuild` and
  `RlBuild`; `build_with` checks 11 applicability rules by hand (`datasets.rs:240-320`) and then dispatches on
  `ViewName` (`datasets.rs:350-412`). The trajectory curation steps (`--repository-history`, `--only-actions`,
  `--roles`, `--control`, `--images`, `--serialise-calls`, `--sampled-only`) are builder methods on two or three views
  each, so the same filter is wired per view.
- **Target.**
  - `ViewSpec` as an enum whose variants carry only their own options (`SftStep { pack, only_actions, control, images,
    .. }`, `PolicyGradient { reward, advantage, pack, sampled_only, overlong, cliff_distill, .. }`, `Preference {
    pairing }`, `Voice { writer, token_budget, .. }`, ...). On the command line each view's options are a clap
    argument group; a flag outside its view's group is a parse error. The 80 lines of refusals are deleted.
  - Curation shared by the action-level views (`sft-step`, `policy-gradient`, `self-distill`, and the run-judging
    `preference` and `feedback`) as an ordered `Vec<Curation>` (history, actions, roles, images, serialised calls)
    applied in one place before projection, each counting its exclusions under the names the manifests carry today.
  - The `View` trait in `splinter-data` (`lib.rs:596`) stays as it is.
- **Safety net.** Golden dataset ids: one fixed store, every view and every option combination the specs use today,
  dataset id and manifest digest recorded before the change and required equal after (a dataset id is the manifest
  digest, so an unintended change in any record shows). The SC-1 refusal test, then turned into a parse test.
- **Depends on.** SC-1.

## SC-4 `splinter-pipelines` is two crates' worth (M/L, open)

- **Problem.** `crates/pipelines` holds 78 top-level modules, about 73k lines of source and 42k of tests: every
  stage and pipeline, the examiner's command-line contract (`bench/`, 11.4k lines: checkout, practice, examine, the
  task session, artifacts, splits, scores, verdict checks; plus `campaign/examiner.rs`), the hub bundles and the
  campaign. 25 files sit between 760 and 800 lines against the 800-line limit, a sign the limit is met by trimming
  rather than by structure. Two modules are both called practice: `practice` (repository practice) and
  `bench::practice` (examiner practice).
- **Target.** An adapter crate `splinter-bench` in the adapters tier, beside `splinter-agent` and `splinter-model`,
  holding the examiner's contract only: `BenchCheckout`, the `bench` commands and their typed outputs, the task
  session, artifact and split readers, verdict verification. The stages that use it (`bench::practice`'s stage,
  `bench::repair`, `campaign`'s evaluate stage) stay in pipelines. `architecture.toml` records the tier and forbids
  `splinter-bench` -> `splinter-pipelines`. Rename `practice` to `repository_practice`.
- **Safety net.** The existing integration specs (`bench_practice`, `campaign/agentic`, `bench_interventions`,
  `practice_episodes`) move unchanged and pass unchanged; the architecture gate runs on the new tier.
- **Depends on.** SC-2 and SC-3 (they rewrite the files that move).

## SC-5 Retire `generation run` (L, open; carried from `campaigns.md` and `acquisition.md`)

- **Problem.** Two multi-generation drivers: `pipelines::generations` (1.3k lines, `splinter generation run|matrix`)
  and `pipelines::campaign` (`splinter campaign run|status|rollback`). Both release through the one promotion rule
  (baa35b8fe), but each has its own ledger, replay choices and stage order.
- **Target.** As `acquisition.md` "Generations on the coding path" lists: a campaign rollouts/examiner pair over a
  repository (rollouts through repository practice's solve stage, examination by the in-process panels), a gate stage
  that decides with `promotion::decide_unexamined` when no signed verdict exists, the failure-targeted and
  learning-progress replay choices as campaign curriculum options. Then delete `pipelines::generations`, the
  `generation` command and their specs; move `scripts/qualify/plan.toml`'s generation steps to `campaign`.
  `generation matrix` (the retention matrix) becomes `campaign matrix` over the campaign's champion history.
- **Safety net.** `campaign::generations` (7 specs) extended with a repository campaign; the `generations` specs'
  assertions (retention limit, regression limit, ledger contents) ported to it before the module is deleted.

## SC-6 Hand-copied training-records contract (S, open)

- **Problem.** `crates/data/tests/fixtures/training-records/v2/` is a copy of brain's fixture directory. It is
  identical today (`diff -r` against brain 6d9a31d14 is empty), and nothing keeps it so.
- **Target.** A gate `check/gates/contract-fixtures` that compares the copy's tree digest with the same directory in
  the brain checkout Cargo resolves for the revision `Cargo.lock` pins (the git source cargo already fetched; no
  network). A mismatch names the files. Brain's half: publish the fixture's digest with its contract version.
- **Safety net.** The gate's own bats test with a one-byte change.

## SC-7 Dead and experiment-named code (S, open)

- `campaign::factory` (326 lines): after the in-memory arm loop was retired its only consumer is
  `tests/it/campaign_factory.rs`. Delete both unless the environment-factory ledger names a stage that will use it.
- `agent/src/policy_endpoint.rs` answers as `"minimind"` (lines 144, 196, 207): a library double named after one
  sample. Take the id from the caller.
- `InterventionKind::HintedRetry` (see SC-2) until SC-2 lands: give it its own type rather than a router variant no
  route can produce.

## Not now

- **The in-process examination (SC-8, deferred).** `pipelines/src/{exam,exam_set,exam_seal,sealed,canaries}.rs` and
  `release/` (6.7k lines) plus `eval/src/{gate,verifiers,promotion}` (7.3k) examine Splinter's own repository tasks.
  Sealed exams are the examiner's by ownership, but it cannot yet examine repository-snapshot tasks. Release decisions
  already go through the one promotion rule; moving examination out waits for the examiner to own those tasks.
- **The loop drivers** (`learn`, `practice`, `absorb`, `consolidate`, `improve`, `campaign`): revisit after SC-5, as
  campaign stages, one at a time and only where a spec shows the same stage order.
- **The `View` and `Intervention` traits and the stage ledger:** sound; the debt is around them, not in them.
