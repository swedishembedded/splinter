<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Asynchronous GRPO: continuous grouped rollouts into an online brain job

Scope: the controller that keeps sven rollouts of many task groups in flight against brain serving the current policy, scores complete groups, filters the ones with no signal, bounds staleness by policy version, applies token backpressure from the learner, streams qualifying groups into brain's streaming policy-gradient job, and records every group's lifecycle durably so a run resumes without repeating finished rollouts.

Reference design: TRL's `AsyncGRPOTrainer` (pinned under `~/resources/reference/trl`). The split of
responsibilities: sven runs each rollout and records its requests exactly; brain serves, trains and publishes
versions; Splinter decides which tasks are drawn, which groups qualify and when the run ends.

## Design

- **Unit of generation is a rollout, unit of advantage a group.** A group is `G` independent attempts of one task.
  Groups progress independently; there is no batch barrier. A group is sent to the learner when its last attempt is
  scored.
- **Lifecycle per group** (durable, fenced by the run's controller lease): `pending -> running -> scored ->
  {filtered | stale | sent}`, `sent -> {applied | stale}`, `applied -> committed` (and `applied -> sent` when a
  restarted learner lost an uncommitted update). A resumed controller relaunches only `pending`/`running` rollouts and
  re-sends `scored`/`sent`/`applied` groups the learner has not reported committed (it deduplicates by group key).
- **Qualification** (Splinter's decision, measured never guessed): zero-variance groups are dropped before training
  but counted in reward statistics (overall reward, fraction of zero-std groups, their mean reward); a rollout whose
  requests do not all carry exact token ids and behaviour log-probabilities from a policy trace is not training
  material (no proxy tokens in async mode).
- **Staleness**: each rollout's policy-version span (oldest and newest version among its requests); a group whose
  oldest version is more than `max_staleness` behind the learner's published version is dropped and counted.
- **Backpressure**: a token allowance `max(max_staleness, 1) * tokens_consumed_per_step` (estimated from the job's
  step budget until the first steps report) bounds how far rollout production runs ahead of learning; new rollouts
  are admitted only while the allowance and the rollout-slot cap (environment resources) allow.
- **Records**: one rollout record per model call (exact prompt ids, generated ids, behaviour log-probabilities,
  optionally the sampler's top-K, the policy version), `reward` and `group` from the scored group.
- **Telemetry**: queue depth, in-flight rollouts, tokens/s generated, backpressure seconds, staleness histogram,
  dropped-stale and filtered counts, reward statistics; enough to say whether a run is generation- or
  training-bound.

The plan this ledger implements, its package numbers (A01-A53) and the review of it against the code are in
`.agents/research/async-grpo-plan.md`. Rules from it: asynchrony, estimator, packing and curriculum are never changed
in the same experiment; online runs sample untruncated until brain replays the sampler's support; a group is
re-sent until brain reports it committed (applied is not durable).

## Open

- [ ] Revision manifest for cross-repository runs (A01): brain, sven and Splinter commits, the training-record
      contract digest, model and tokenizer digests, hardware, resolved configuration; a run refuses mixed revisions.
- [x] Pure controller core (A23, A25-A26): `splinter_factory::online::control` (no I/O, time an argument), shared
      types in `splinter_core::factory::group`, seams in `splinter_factory::online::seams` (`RolloutExecutor`,
      `Learner` with `Applied`/`Dropped`/`Committed` events, `TaskSelector` with `RoundRobin`, `GroupLedger`,
      `RunClock`), the loop in `online::driver`. Decisions: the token gate applies when a group starts and counts
      admitted rollouts at the mean tokens of a finished one (none finished yet: the `rollout_tokens` setting, a
      pessimistic bound such as the sampling budget, so a slow first group holds no other back); queued tokens are
      what the learner will hold (a queued group counts only its exact scored rollouts); the step estimate
      is the mean of the last `step_window` step reports, never below the job's step budget (so the allowance always
      holds a full step); an inexact rollout is never sent, and a group whose exact rollouts' rewards do not differ
      is filtered `not-exact`; an infrastructure failure is retried `max_failures` times then abandoned, never
      scored; truncation is counted, not filtered; backpressure time and the step estimate restart on resume, every
      other statistic is rebuilt from the ledger. Tests `splinter-factory/it/online::`.
- [x] Durable group ledger with resume (A28-A30): tables `groups` and `group_rollouts` in `state/controller.db`
      (additive, schema version unchanged) under the controller's fence, records as JSONL in the row until the group
      is terminal; `RunLedger` renews the lease. Kill-and-resume tests over the real run state: an executor killed
      mid-run, a learner connection dropped after it took a group, a cancel followed by a learner restart; no
      finished rollout started again, every group committed exactly once.
- [x] Discrete-event simulation (A27): `online::simulation`, at least 100k finished rollouts per run of the test
      (seed drawn and printed) over drawn worlds: group size 2-8, 1-64 slots, staleness 0-4, slow or fast training,
      Pareto(1.2) durations, all-equal rewards, publication every 1-3 steps, verifier outage windows, worker
      failures, unscored and inexact rollouts, controller kills, learner restarts and cancellations; no deadlock
      within the horizon, slots and the learner queue bounded, every group and rollout terminal and accounted, no
      group committed twice; it asserts every scenario occurred (interruptions, commits, filters, stale drops,
      failures, abandonments, full answers, resends). It found two deadlocks the unit tests did not (knowledge 217).
      Contracts it relies on, for the learner adapter lane and brain's streaming job: the learner accepts while it
      holds less than a step (a 429 only once it holds a step's worth); a learner that loses uncommitted work (a job
      restart) fails the run's learner calls, so the resumed controller sends again (an unreported restart would
      leave applied groups waiting).
- [x] Rollout executor on the sven SDK (A22): `splinter_pipelines::episodes::online` (`run_online`, the
      `OnlineEpisodes` executor): each rollout one practice episode on a scoped thread, an isolated sven session in a
      fresh copy of the task's tree, the policy trace and the optional top-K (`top_logprobs`) on every request,
      untruncated sampling at the run's temperature (a sampler with top_k, top_p below 1 or min_p above 0 is refused
      by name), the episode's deadline, `cancel` stopping every running rollout; the task's verifier rewards it
      afterwards and a verifier error, verifier timeout or infrastructure failure is an `Err` (retried by the
      controller, never reward 0), an ungraded episode an outcome with no reward. An episode is keyed by its content
      and its group (seed, task, policy, configuration, group key, index), so a rollout relaunched after it finished
      is answered from its record. Each session's requests are recorded by a request log of its own
      (`SolveOptions::request_log`, sven's `RecordingProvider` through `sven_sdk::drivers::request_log`) and filed
      on the trajectory as its recorded calls (`run::corpus::file_calls`, shared with the corpus import).
      Tests `splinter-pipelines/it/online_episodes` (scripted provider through the SDK).
- [x] Records one per model call (A14): `splinter_data::calls::project_calls` (see the validation map, "Per-call
      records (online)"); the canonical producer writes brain's additive v2 fields (`behavior_top_ids`,
      `behavior_top_logprobs`, `policy_version`, `sampler`, `truncated`); `CallRecord::record` makes the v2 line once
      the group is scored. A rollout is exact only when every call is; the scrub must leave every call's text as the
      model saw it (a run whose tool output names its scratch tree is inexact, counted `scrubbed`: a fixed sandbox
      path is the remedy, not yet done). Not yet checked against brain's own `check_dataset` with these fields (brain
      `agrpo-score` is landing them; `make test/contract/brain` once it has).
      Dependency: needs sven `sdk: record a session's model requests to a log of its own` and `sdk: re-export the
      identity a provider reports of its reply` (sven branch `agrpo-exec`), plus sven's per-request identity and
      top-K; the lock pins the public sven remote, which has none of them, so this builds with `make deps/local`
      until those commits are published and the lock is re-pinned.
- [x] Learner adapter for brain's streaming policy-gradient job: the client `splinter_model::served_train::StreamJob`
      (create with the stream document, offer complete groups with `?epoch=N`, events read on a thread and resumed
      with Last-Event-ID, finish; 404/409/410 or another epoch is `StreamError::Restarted`), its scripted double
      (`splinter_model::scripted_stream`: steps taken when events are read, publish/commit cadence, 429 only while
      a step is queued, a restart at a step, a pause), and `splinter_factory::online::brain::BrainLearner` on the
      controller's seam (`BrainTrainer::stream`). A duplicate of a key this job did not accept in its own epoch is
      reported committed (a resumed job reports its checkpoint's keys only as duplicates). Built against the
      scripted double; brain `agrpo-stream` not yet exercised (open: run `make test/contract/brain`-style checks
      once it lands). A truncated rollout is no longer trainable (brain trains none of it, so it could leave a
      group of one).
- [x] The `train.online` factory operation (A24): `splinter_factory::run::online`, over a pinned executable
      collection; params `base, groups, group_size, max_concurrent_attempts, max_staleness, rollout_tokens,
      step_tokens, queue_tokens, publish_every, max_updates, estimator, temperature, top_logprobs,
      checkpoint_every, max_failures, seed, params`; the rollouts through the `OnlineRollouts` seam
      (`splinter_pipelines::episodes::online::FactoryRollouts` on the sven executor), the learner through
      `Trainer::stream`, every group in the run state under a namespace per stage identity, a restarted learner
      resumed up to 8 times. Product: `candidate.json` (the newest published version a checkpoint holds,
      load-checked), `report.json` (online statistics, learner telemetry, tokens generated and trained,
      experience utilisation, useful tokens per second), `experiences.jsonl` (committed groups). Its candidate
      goes to `evaluation` unchanged (the evaluation's base lookup now takes either training operation). Open:
      the command line does not yet hand a factory run its online rollouts (`Services::online` is `None` there;
      it needs a context and the served-id-to-model mapping for brain), and the candidate is the last committed
      published version (a finish that commits steps past the last publication is not served yet).
- [x] The GPU-free toy end to end (A31): `splinter-pipelines/it/online_factory`: dataset.rlvr -> train.online ->
      evaluation on a scripted brain streaming job and a scripted sven policy, the controller killed mid-stage and
      resumed, no group trained twice, the experience manifest equal to the committed groups, the candidate
      evaluated.
- [ ] Qwen3-0.6B and Qwen3-4B on one card (A34-A36): synchronous against asynchronous at equal generated tokens,
      trained tokens, updates, GPU hours and wall time; useful throughput, experience utilisation, held-out and
      retention.
- [ ] Merging append-only calls of one session into one sample (A15), against the per-call reference, once the
      re-forwarded context is measured.
- [ ] Adaptive difficulty (tasks near the capability frontier) once the loop is measured.
