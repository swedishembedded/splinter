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
  {filtered | stale | sent} -> consumed`. A resumed controller relaunches only `pending`/`running` attempts and
  re-sends `scored`/`sent` groups the job has not acknowledged (the job deduplicates by group key).
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
- [ ] Pure controller core (A23, A25-A26; no I/O): group state machine, staleness bound, token allowance gated when a
      group starts (in-flight rollouts counted, so partial groups cannot deadlock the allowance), admission,
      statistics; unit tests with a deterministic clock and scripted outcomes. Placement from the paused lane: shared
      types in `splinter-core::factory::group`, the ledger in `splinter-store::factory::runstate::groups`, the
      controller and seams in `splinter-factory::online` (design and trait signatures in that lane's notes).
- [ ] Durable group ledger in the run state with resume (A28-A30): kill mid-run, resume, no finished attempt
      repeated, no group trained twice, nothing committed lost.
- [ ] Discrete-event simulation of the controller (A27): at least 100k simulated episodes across slow generation,
      slow training, heavy-tailed durations, all-equal rewards, frequent publication, verifier outages, worker failures
      and cancellation; queues bounded, no deadlock, every attempt reaches an accounted terminal state.
- [ ] Rollout executor on the sven SDK (A22): concurrent isolated sessions against the served policy, the policy trace
      on every request, infrastructure failures separated from verified failures (an unknown outcome is never reward
      zero); records from sven's per-request recording, one per model call (A14).
- [ ] Learner adapter for brain's streaming policy-gradient job (scripted double first, then the real API).
- [ ] The `train.online` factory operation (A24) over an executable task collection (`dataset.rlvr` frozen as the
      first curriculum); its candidate goes through the existing evaluation and promotion stages.
- [ ] The GPU-free toy end to end (A31), then Qwen3-0.6B and Qwen3-4B on one card (A34-A36): synchronous against
      asynchronous at equal generated tokens, trained tokens, updates, GPU hours and wall time; useful throughput,
      experience utilisation, held-out and retention.
- [ ] Merging append-only calls of one session into one sample (A15), against the per-call reference, once the
      re-forwarded context is measured.
- [ ] Adaptive difficulty (tasks near the capability frontier) once the loop is measured.
