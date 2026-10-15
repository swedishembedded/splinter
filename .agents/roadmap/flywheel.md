# The flywheel - roadmap

The loop generate -> execute -> verify -> learn -> evaluate -> promote, run
unattended, repeatedly, with every generation regression-protected, over sven
(agent execution) and brain (training, inference). Splinter stays the single place
where learning, experiments, evidence and releases are decided. Update this file in
the same commit that finishes or changes an item. The paper
`papers/splinter-brain-sven-flywheel` holds what has been verified; this file holds
what is still to do.

## Baseline facts (checked 2026-10-16)

- Present: content-addressed experience store with crash-safe ingest, the paired release
  gate (sign test clustered by family, per-release retention, frozen anchor, serve
  agreement), immutable releases with CAS aliases, rollback and lineage, workspace practice
  with hidden tests and an exposure audit, task admission with typed rejection reasons,
  LoRA SFT, DPO, full fine-tune and consolidation, run records with cross-process cancel,
  a `doctor` preflight.
- Missing: a declarative experiment file; a unified environment lifecycle; efficiency and
  safety dimensions in the gate; pipeline resume; a scheduler; a coverage report; GRPO
  as a stage; a `report` verb; retention/deletion enforcement on derivatives.
- Splinter did not build against current sven and brain at the start of this work
  (sven `Agent::events()` changed; brain `ManifestPoll`/`LoadedAdapter` changed); the green
  baseline is item 0.

## Items, in dependency order (S <= 1 day, M 2-5, L > 1 week of agent time)

0. Green baseline (M): `make build test check` pass at the local sven and brain HEADs;
   the lock pins revisions that are on the remotes (push brain and sven first).
1. Experiment spec (M): a versioned, validated experiment file (goal, sources, generator
   dimensions, student and teacher roles, resource limits, evaluation policy, seed)
   consumed by `learn`/`practice` (`--experiment <file>`), plus a candidate/training-run
   manifest unifying `Candidate`, `TrainingSummary` and brain's adapter manifest.
2. Contracts and provenance (S-M): typed `Provenance.policy_identity` and `sampling`
   added skip-if-default (golden tests prove addresses do not change); an unknown-field
   policy per record; a cross-project contract suite (sven session -> ATIF -> experience
   -> dataset -> brain accepts it), extending `make test/e2e/brain-sven`.
3. Isolation (M): an explicit refuse-on-host policy (a workspace requires a container for
   untrusted tasks instead of silently falling back to the process sandbox); decide
   network denial for the process sandbox; tests for leaked children, disk exhaustion and
   cancellation. Environment lifecycle (prepare/reset/execute/verify/snapshot/restore/
   cleanup) as a thin adapter over `Sandbox` only when a second environment kind exists.
4. Generation and coverage (M): coverage dimensions and an unsupported-families list in
   a report; per-experiment admission report; adaptive priorities from measured gaps.
5. Experience store (M): a store-wide frontier index with flaky and unsupported classes;
   deletion/retention enforcement on derivatives; PII classification.
6. Evaluation (M): efficiency (tokens, tool calls, wall time per success) and safety
   (unauthorised access, leakage, verifier tampering) dimensions in the release gate;
   a frozen-final-test ledger so a final exam cannot be tuned on; a runnable
   negative-control suite (corrupt adapter, changed test definitions, injected regression,
   leakage); the serve check honouring the request temperature.
7. Pipeline resume (M): stage outputs content-addressed so a stage restarts; run
   liveness (a died process must not read as cancelled); required for any run that can
   cross the job wall clock.
8. Doctor and acceptance (M): `doctor` reports the memory cap, job end time, GPU free and
   lock holder, per-gate PASS/FAIL/BLOCKED as JSON, non-zero exit on any non-PASS required
   gate; one acceptance target (`make test/e2e/flywheel`) running generate -> solve ->
   verify -> train -> exam -> gate -> promote -> serve -> second generation -> rejected
   degraded candidate -> rollback; a `report` verb (JSON and human) built from `status` and
   lineage.
9. On-policy learning (M, needs brain and sven items): a GRPO stage over brain's external
   rollout records, fed by sven trajectories carrying token ids, logprobs and sampling
   parameters; `Objective::Reward` routed to brain's rollout format instead of the
   export-only one. Gated on the sven and brain items in their roadmaps.
10. Scheduler (L, last): one GPU pool, job identity and idempotency keys, leases and
    heartbeats, GPU/disk reservation; only after the single-node cycle is solid.
11. Fast continual learning (see `continual-learning.md`).

## Evidence discipline

Gains are judged only by a measurement the policy under training cannot influence
(held-out, retention, anchor); an unmeasured number is absent, never zero; a final test
set tuned on is no longer a final test set; underpowered comparisons are reported as
inconclusive and more tasks are collected rather than the threshold changed.
