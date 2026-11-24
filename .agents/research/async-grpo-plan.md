<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Async GRPO across brain, sven and Splinter: the plan and its review (2026-11-24)

The plan (written by the user, 2026-11-24) makes asynchronous GRPO the execution architecture for online RL in the
stack: a composition of an experience source, the existing policy-gradient objective and brain's generic training loop,
not a new trainer. TRL's `AsyncGRPOTrainer` (pinned at `~/resources/reference/trl`, commit 6267b72) is a golden
reference, never a runtime dependency. This note keeps the plan's structure, the review of it against the code as read
on 2026-11-24, and where each work package is tracked. Status lives in the ledgers: brain
`.agents/roadmap/async-policy-gradient.md`, Splinter `.agents/roadmap/async-grpo.md`, sven
`.agents/roadmap/training-instrumentation.md` items 12-13.

## Ownership (agreed)

Splinter: task selection, curriculum, reward from verified outcomes, trajectory-to-record projection, rollout
scheduling, held-out evaluation and release. sven: agent execution and exact capture of what each request sampled.
brain: sampling probabilities, advantages, admission and staleness of training experience, policy-gradient
mathematics, optimiser, policy publication, GPU scheduling. Neither brain nor sven learns what a task, worker or
verifier is.

## Phases and packages (the plan's A01-A53, grouped)

| Phase | Content | Packages | Owner |
|---|---|---|---|
| 0 | revision manifest, baseline inventory, baselines B0-B3 | A01-A02 | all |
| 1 | one policy-gradient config (EC-1/EC-2), estimator variants, step-wide normalisation, sparse head targets, score centring, CPU oracles | A03-A08 | brain |
| 2 | exact experience contract: trace v2, three distributions, capture levels, termination cases, sven per-request capture, shared fixtures (EC-5) | A09-A13 | brain, sven |
| 3 | token-native projection and multi-turn reconciliation | A14-A15 | Splinter |
| 4 | policy identity, leases, publication transaction, cache isolation | A16-A18 | brain |
| 5 | live experience source, training session API, complete-group admission, receipts | A19-A21 | brain |
| 6 | incremental execution backend, continuous groups, `train.online` factory op | A22-A24 | Splinter |
| 7 | budgets, token backpressure, staleness, discrete-event simulator | A25-A27 | brain, Splinter |
| 8 | durable receipts, checkpoint reconciliation, process-kill suite | A28-A30 | brain, Splinter |
| 9-12 | colocated scheduling, batch planner and packing, head optimisation, distributed actors | A37-A48 | brain |
| E | model-free integration, tiny-model conformance, Qwen3-0.6B/4B runs, held-out and retention | A31-A36, A49-A53 | all |

## Review: what the code says about the plan

Correct and adopted:

- No `AsyncGrpoTrainer`: the estimator, the experience source and the scheduling mode are independent axes over
  `model::Objective` and `fit_with`. EC-1/EC-2 land before the new estimator is exposed, so the score-centred
  estimator is a variant of the shared `pg_batch` path both `Grpo` and `ExternalPolicyGradient` use.
- Sparse token terms in the LM-head loss (`d z_v = p_v sum_j c_j - c_v`), one pass, generic for every model on the
  shared head; the plan's formula is the one implemented.
- Normalisation over the whole optimiser window, invariant under repartitioning (TRL issue 7249 is the defect).
- Complete-group admission with typed refusals; group-level token reservation to avoid the partial-group deadlock
  (7.3): the controller gates admission when a group starts, counting in-flight rollouts.
- Separate "applied" from "committed": a consumed group is durable only once the checkpoint that contains its update
  is; the job reports both, and a client re-sends what is not committed.
- The categorical-policy oracle (4.2): exact enumerated expectations of each estimator, including complete groups
  for group-standardised advantages, with no transformer.
- Learning effectiveness judged by held-out and retention through the existing release gate; equal-compute and
  equal-experience comparisons; the ablation order A-G (asynchrony and estimator never changed together).

Corrections from the code:

- **Truncated samplers make today's behaviour log-probability the wrong denominator.** brain's policy trace is the
  token's whole-row log-probability at the temperature (knowledge 520, 528), while agent runs sample with a cut
  sampler (top_k 20, top_p 0.95 in knowledge 528's workload). The sampled token's real behaviour probability is
  `p / mass(support)`. The clipped surrogate trained on such records starts from a ratio of `mass < 1` per token, not
  1. Until the support is replayed (the trainer's probabilities renormalised over the sampler's support, TRL's
  `sampling_mask`), online policy-gradient runs sample untruncated (top_k 0, top_p 1, min_p 0); the score-centred
  estimator refuses truncated records by name. Support replay is a brain package after the baseline.
- **sven's per-request identity is not yet per request.** `extra.sven.requests` carries ids and log-probabilities per
  request but no policy version, adapter or log-probability kind; the step-level identity is first-wins. A step whose
  requests straddle a refit is mis-attributed today (sven item 12).
- **Leases mostly exist.** brain captures a request's adapter identity at admission and only marks a replaced adapter
  draining; each release has its own KV namespace. The real gap is that a re-published route refreshes only when its
  lane is idle, so under continuous load a new version may never become routable (brain item "routes refresh under
  load"). A run-level identity (run, committed update, digests) is added on top of the serving version.
- **Multi-turn merging is an optimisation, not a prerequisite.** One record per model call is already exact (each
  call's own prompt ids). Merging append-only calls into one sample saves the re-forwarded context, which for long
  agent sessions grows with the square of the turn count; it is the first training-side optimisation to measure
  (Phase 10, against the per-call reference), not Phase 3 work.
- **Durable layout follows the store's packed design**, not a directory tree per group or attempt (file-per-record
  layouts are refused in this stack): group rows in the run's controller database, records as JSONL in the row until
  the group is consumed.
- **The TRL baseline (B4) is golden fixtures only:** advantages, masks, score-centring coefficients and support
  normalisation computed by the pinned TRL code on CPU, in an isolated tool. No TRL training run.
- **Brain's in-process `Grpo` stays synchronous** (the reference); asynchrony is for externally produced experience
  first.

Open decisions (proposed defaults, revisit with numbers): staleness in published versions (brain job) and admission
by token allowance (Splinter); the 15% useful-throughput bar for a performance change; episode-pinned policy binding
as the first mode.
