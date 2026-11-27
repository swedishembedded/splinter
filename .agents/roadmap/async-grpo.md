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

- [x] Revision manifest for cross-repository runs (A01, lane agrpo-prov): `splinter_core::revision`. A run pins
      `revisions.json` (`RunRevisions`: the build's Splinter, sven and brain commits, dirty flags, executable digest,
      the record contract it writes) when it starts; a resume by a build of other commits is refused
      (`RunError::RevisionsChanged`, one line per difference; a dirty or unknown tree differs unless it is the same
      executable) unless the recipe says `revisions: mixed`. Every `model.train`, `train.online` and `evaluation`
      product holds `revisions.json` (`StageRevisions`): the build, what each brain server says it is
      (`ServedRevision` from `GET /v1/capabilities` `server` and `GET /v1/devices`: engine version and commit,
      backend, hardware, record contract), the base and tokenizer digests brain reported for the stage (a training
      job's `base_digest`; the published versions' lineage and the stream frame's `tokenizer_digest`), the input
      digests and the resolved configuration. A server whose record contract differs from the pinned
      `RECORD_CONTRACT_DIGEST` (the digest of this build's copy of brain's example set, held to it by a unit test) is
      refused by name before any stage (`RunError::ContractMismatch`); one that reports none (a frozen older brain,
      which other runs use on purpose) trains with a warning naming its endpoint and is recorded with `engine` and
      `training_records` `null`, unless the recipe says `contract: required` (default `checked`), which refuses it. A local build (`make deps/local`) embeds its
      checkouts' HEADs and dirty flags (`crates/splinter/build.rs`, `BuildInfo::embedded` with `LocalCheckouts`), not
      the lock's commits. `train.online`'s report carries the manifest and `versions` (every published version with
      its run-level `lineage {run, epoch, update, base_digest, adapter_digest}`, `PublishedVersion`). The copy of the
      example set had drifted from brain's (knowledge 218); it is brain's again. Tests: `splinter-core` `revision::`,
      `build::`, `splinter-factory` `it::revisions` (stage manifests, the resume refusal and `mixed`, the contract
      refusal, the brain trainer against a scripted server), `splinter-pipelines` `it::online_factory` (versions and
      the report's manifest). Open: the evaluation stage's manifest names no model digests (brain's chat replies
      carry `base_digest`, which the evaluation does not collect); sven's per-request `policy_lineage` is not yet read
      into trajectory records.
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
      failures, abandonments, full answers, resends). It found two deadlocks the unit tests did not (knowledge 217),
      and later a third: a resent applied group answered `duplicate` stayed applied while the learner held it queued
      and dropped it; any answer but `full` now makes it `sent`, and a group counts toward the allowance only once
      the learner answered `accepted` (knowledge 219; the failing seeds pinned in `simulated_runs_that_once_failed_finish`).
      A deadlock fails the test once nothing is recorded for 10^8 simulated ms (it used to hang until a horizon).
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
- [x] One rendering of a request on its own, offline and online (2026-11-27, knowledge 232): the policy-gradient and
      step views read a single agent's requests that do not extend one another request by request, each action's
      context the request it answered, through the same `requests::sent_and_answered` the per-call records use; a
      step no request answered is counted `requests_unpaired`, a rollout with none paired keeps only its baseline
      record, and the trajectory is never rendered where requests were recorded (it had been, under Splinter's
      generic prompt and no tools: 134 forward and 79 backward rollouts of the Vim runs). An answer whose arguments
      are not JSON is counted `unparsed_arguments` on both paths (online it was `unreadable_request`). Open: an
      answer sven's runaway-argument guard stopped has no trajectory step; offline it is no action, online the
      rollout's calls outnumber its traced requests unless sven files that request's trace, so the rollout is
      counted `unpaired` and never trained. Check on an online run before relying on rollouts that hit the guard.
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
      goes to `evaluation` unchanged (the evaluation's base lookup now takes either training operation). `splinter
      run` hands it its rollouts (`splinter_sdk::factory::online_rollouts`, sven sessions through brain's driver
      against the id the job publishes). The stage records its job (`stream-job.json` beside the run's state) and a
      resumed run cancels the job its stopped controller left (brain starts no other job of a run while one runs;
      knowledge 221); a job is asked for again while the server is busy or out of reach, an offer the server does
      not answer is made again (both for up to five minutes), and a job that continues the run counts the version it
      publishes from its checkpoint as committed. Open: the candidate is the last committed published version (a
      finish that commits steps past the last publication is not served yet); a continued job on a server that
      still serves the run publishes nothing at its start, so a last epoch that trains no step finds no committed
      version (brain naming the checkpoint's version in the `stream` frame would settle it); the report's learner
      telemetry covers the last controller's leg only (the controller's own counters are rebuilt from the ledger).
- [x] The real stack on the CPU (lane agrpo-e2e2, `make test/online`, `scripts/online/cpu-stack.sh`): a toy qwen3
      base (`sample-models-qwen3 toy`), `brain serve --openai --training`, tasks.pin -> train.online over
      `tasks/toy-answers` (answered in the reply, `[agent] answer_file`), the controller killed once a group is
      committed and resumed, brain killed with groups in flight and started again (the in-stage restart takes the
      run's next epoch and sends again what it had not committed). Checked by the script: versions published and a
      reply naming the served version, applied and committed receipts, no group committed twice nor applied after
      its commit, the experiences equal to the committed groups, the report's counters adding up. Both estimators
      pass (clipped; score-centred with `top_logprobs: 2`). Findings fixed on the way: brain's step figures are
      floats, which the learner read as no tokens (knowledge 220); brain held the CPU lane for the whole stream job
      and refused a byte-level toy's records (brain knowledge 584, 585). Not covered: dataset.rlvr and evaluation
      (they need a Harbor worker), a GPU.
- [x] The GPU-free toy end to end (A31): `splinter-pipelines/it/online_factory`: dataset.rlvr -> train.online ->
      evaluation on a scripted brain streaming job and a scripted sven policy, the controller killed mid-stage and
      resumed, no group trained twice, the experience manifest equal to the committed groups, the candidate
      evaluated.
- [ ] Qwen3-0.6B and Qwen3-4B on one card (A34-A36): synchronous against asynchronous at equal generated tokens,
      trained tokens, updates, GPU hours and wall time; useful throughput, experience utilisation, held-out and
      retention.
      **Partial (lane agrpo-real, 2026-11-25).** A container restart lost the node-local run directories, and the
      cards were then withdrawn. The figures below were read from the runs before that; the settings are in
      `~/project/shared/tmp/agrpo/agrpo-real/cfg` and the scripts in `.../scripts` (run.sh, batch.sh, analyze.py,
      table.py, probe.py).
      - Task family `tasks/integer-arithmetic` (base pass rates in `tasks/README.md`): Qwen3-0.6B has 59 train and 28
        held-out tasks in (0.1, 0.9); Qwen3-4B has 10 and 10, since its answers are mostly all right or all wrong.
      - Smoke (A34, Qwen3-0.6B, 12 groups of 8, `max_staleness 2`, clipped, rank 16): three versions published,
        9 groups applied and 9 committed, none applied or committed twice. At version 1 the ratio mean was 0.998 but
        the min and max were 0.05 and 2.77 (clip 7.8%), an int8 KV cache artefact (brain knowledge 588); with
        `--kv-cache f32` they were 0.90 and 1.09 (clip 0). Every later run serves with f32 KV.
      - Defects the real stack showed, all fixed on this branch: rollouts drawn before the policy was served
        (80b8dda88); a learner trained in thinking mode for no-think rollouts (9662d6623, knowledge 223); a
        finished run's result read before the job had ended (80b8dda88); an unanswered finish (80b8dda88); and the
        synchronous-run deadlock (40fdab3b4, knowledge 222). The last one stopped the first S arm; the S figures
        below are from the run after the fix.
      - S against A, Qwen3-0.6B, ONE seed each (seed 1; the planned seeds 2-3 and the score-centred arm C were not
        run). 128 groups of 8 over the 59 tasks, `step_tokens 1024`, `queue_tokens 4096`, `publish_every 1`,
        clipped, temperature 1.0, `max_output_tokens 512`, rank 16. Served with `--kv-cache f32 --ctx 4096
        --max-batch 16`, one GH200 shared with another tenant's 63 GB server (binary set bin-760c04d3b-s5). Held-out:
        28 tasks x 32 draws, before (base) and after (candidate) on the run's server, 95% two-level bootstrap.

        | figure | S (max_staleness 0) | A (max_staleness 2) |
        |---|---|---|
        | wall s (GPU s: one card) | 235 | 220 |
        | optimiser steps | 6 | 7 |
        | generated / trained tokens | 24716 / 6787 | 23981 / 7565 |
        | groups dropped stale | 1 | 0 |
        | zero-variance groups (of 128) | 95 | 92 |
        | useful tokens/s (trained/wall) | 28.8 | 34.4 |
        | experience utilisation (trained/generated) | 0.275 | 0.315 |
        | learner wait s | 154 | 178 |
        | backpressure s | 53.7 | 0 |
        | staleness mean / max (versions) | 0 / 0 | 0.19 / 1 |
        | train reward, first / last quarter of rollouts | 0.805 / 0.973 | 0.711 / 0.902 |
        | held-out before | 0.693 [0.593, 0.780] | 0.695 [0.589, 0.787] |
        | held-out after | 0.790 [0.643, 0.913] | 0.781 [0.631, 0.915] |
        | held-out after - before | +0.097 [-0.031, 0.231] | +0.086 [-0.001, 0.169] |

        What this shows: both arms ran to the end with no group trained twice, and both were generation-bound (the
        learner waited 65-81% of the wall). A was 6% shorter in wall time and trained 11% more tokens from 3% fewer
        generated ones, and S lost 54 s to backpressure. Most of the generation went to groups whose rewards did
        not differ (72-74% of groups). What it does not show: any difference between the arms beyond noise (one
        seed each), or a held-out gain (both intervals include 0 or nearly 0). A pilot of A at 64 groups (an earlier
        binary) gave +0.086 [-0.006, 0.174]. The held-out gain is mostly a format: the base often ends a correct sum
        with chatter, which the grader scores as wrong, and the trained policy ends on the number. Products did
        not improve (mul 3x3 unchanged).
      - Rollout cost: a 0.6B rollout of about 37 tokens took 2.1 s on average (1 s resolution, 512 rollouts, 16
        slots), while a direct request through brain at the same concurrency took about 0.5 s. That leaves roughly
        1.6 s per rollout outside the model (sven session, tree, verifier); this is inferred, not profiled. From
        start to the first rollout took 28 s.
      - Profile of a short Qwen3-4B run: brain roadmap `async-policy-gradient.md`, end-to-end item. One runaway
        rollout (three attempts at the 2048-token context) took 74% of a 175 s run, and the learner waited 84%.
      Proposals (bottleneck, baseline, evidence, change, oracle), ranked by measured share:
      1. Runaway rollouts: 129 s of 175 s (4B profile); a context-exhausted rollout is retried twice as an
         infrastructure failure. Change: `max_output_tokens` (landed here, after-numbers not measured), and
         context exhaustion recorded as truncation, not retried. Oracle: the same 4B workload's wall and truncated
         count.
      2. Zero-variance groups: 72-74% of groups, so 69-72% of generated tokens are never trained. Change: draw
         tasks near the frontier (drop a task for the run once its groups come back all-pass), the adaptive-difficulty
         item. Oracle: experience utilisation and held-out at equal generated tokens.
      3. Learner starvation, 65-84% of the wall: per-rollout overhead (about 1.6 s of 2.1 s, inferred) and 16 slots.
         Change: profile the episode path first (sven session start, tree copy, verifier), then raise the slots
         with brain's `--max-batch`. Oracle: rollouts/s and useful tokens/s at a fixed task set.
      4. Startup, 27-28 s before the first rollout (two base loads plus the first publication): share the frozen
         base (brain A37-A39).
      Open: seeds 2-3 of S and A, the score-centred arm C (configs `cfg/cmp-C-*.with`), the Qwen3-4B pair, the
      after-measurement of `max_output_tokens`, and retention.
- [ ] Merging append-only calls of one session into one sample (A15), against the per-call reference (brain A40-A41,
      knowledge 619 there: under `token-sum` a merged record trains its per-call records' gradient). Done
      (lane rl-merge-producer, 2026-11-27): the producer `splinter_data::session::SessionRecord::merge` turns an exact
      call projection into brain's packed rollout record, or an `Unmergeable` naming the call and first differing
      position (not exact, missing ids, sampler, tokenizer or tools differ, not append-only, messages diverge, longer
      than the block); property tests hold its answers to the per-call answers and its divergence finder to brain's
      rule. Offline: `dataset.project` `sessions: merged` (policy-gradient only, default `per-call`, identity unchanged
      when unnamed) writes each append-only exact session as one record and any other one record per action, with the
      counts in `dataset.json` `sessions` and `report.md`; a packed record no longer mixes exact and proxy turns. The
      trainer already sends `pack_answers`, `max_block` and `policy_gradient.loss_agg`. Contract: brain reads the
      merged group and refuses a rewritten one with `session_not_append_only` (`make test/contract/brain`). Measured on
      178 recorded vim sessions (knowledge 252): 0 of 118 Qwen3 sessions are append-only (the template drops the
      earlier empty think block), 58 of 60 Qwen3.8-27B sessions are (the two others: sven reordered argument keys),
      and those forward 4.9x fewer ids merged.
      History as sampled, the standard (lane args-verbatim, 2026-11-28, knowledge 254, supersedes 252): brain sends
      each reply's `sampled` record (generated ids, prompt length, digest and generation header) and splices a turn
      sent back with it as its ids; sven sends every answer back verbatim with its record, with no option, and a
      rewrite of the history (compaction, system prompt, tool list, a deferred tool's load) is a `ContextRewritten`
      segment boundary. The producer is now `SessionRecord::segments`: one merged record per append-only segment,
      each with the rollout's one reward, cut before a call that would pass the block; it falls back per call only
      when a call is untraced or the samplers differ (`untraced`, `mixed_sampler`). A call trains its traced ids when
      the server's record names them. Online: `rollout_of` writes the segments under `Sessions::Merged`, and
      `records_of` already reads each line as a `SessionRecord`. Offline: `sessions: merged` writes the same segments.
      The comparisons that trimmed text before matching a request's history to the answer it repeats are gone.
      Verified on a CPU brain (Qwen3-0.6B, f32 KV): two two-call sessions merged into one record each, brain's
      `check_dataset` accepted both (packed, 2 records, 86 supervised ids) and its `score` reproduced every call's
      behaviour log-probabilities (worst per-id difference 5.6e-5 over 4 calls). Open: start the stream job with
      `pack_answers`, a `max_block` holding the longest segment and `loss_agg`; the merge rate on traced online
      sessions; `token-sum` against `seq-mean`.
- [ ] Adaptive difficulty (tasks near the capability frontier) once the loop is measured.
