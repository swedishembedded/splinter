# The flywheel - roadmap

Scope: the unattended generate, execute, verify, learn, evaluate and promote loop over sven and brain, and what it still lacks.

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
12. Task directories as the generation contract (M) - done except the live run. `learn
    --task-harness` runs `harnesses/flywheel/invent` (`splinter_agent::flywheel`) once per
    source window on a workspace that persists for the stage, with a run-level deadline
    (`--task-harness-timeout`, never the request deadline) and `sven`, `git`, `python3`,
    `bash` and the task runtime on `PATH`. The new commits of `tasks/` are read as committed;
    each task directory becomes a `workspace-change` task through
    `WorkspaceGenerator::import_task_directory` (hidden `tests/test.sh` run with `TESTS_DIR`,
    reference change from `solution/solve.sh`, the task's `harness.yaml` kept as the privileged
    `solve-process` item), a malformed directory is refused with the missing piece named, and
    the tasks stage reports invent's own yield (`log.jsonl`) beside what it admitted and
    refused. Solve runs a task's own process (the generic one otherwise). A validate stage
    (`--validate-harness`) keeps a passing attempt only when every focus decides keep and
    reports a validator run without a decision apart from the discards. The old
    `sections/NN.md` / `tasks.json` proposer is removed. Open: an end-to-end run against a live
    brain server (the fake `sven` specs cover the subprocess contract only); the task runtime
    must be in `workspace.runtime_dirs` for the host-side verifier and the in-process solve
    to find it. The acquisition (`learn --source`, the skill-graph rounds every
    Lua experiment uses) does not run the harness: its rounds write tasks with the generator
    model. A live run (2026-10-27) took that path with the flags dropped; the combination is now
    refused (8a0fecf01). Open: the acquisition's round task writer as the invent harness, so the
    measured loop and the harness pipeline are one path.

## Evidence discipline

Gains are judged only by a measurement the policy under training cannot influence
(held-out, retention, anchor); an unmeasured number is absent, never zero; a final test
set tuned on is no longer a final test set; underpowered comparisons are reported as
inconclusive and more tasks are collected rather than the threshold changed.

## Findings of the second-wave experiments (2026-10-16), outstanding

- Retention for a facts campaign: the gate's retention check holds a candidate to each earlier
  release's held-out families, which a facts campaign never taught, so it cannot see forgetting of
  taught facts (knowledge 061). Hold it to the release's trained facts under their stored variants.
- Early stopping on a monitor set of unseen families is meaningless for invented facts; the default
  picks an early step. Offer a monitor drawn from the trained facts' variants, or refuse to monitor
  when the held-out families cannot be learned.
- Pipeline resume (item 7) would have saved most of the wall time of every rerun here: a run that
  fails at its last stage repeats the generator's tasks stage (about 8 to 25 minutes on a shared card).
- A learn on a single source file is refused after its rollouts; check the number of independent
  groups before spending them (knowledge 056).
- The generator omits the subject it names from its own question in about half of its proposals at 4B
  and 8B; the brief, not the rule, is the next thing to test.
- The serve check reports the beginning of the server's output in its not-measured reason after the
  notices of every model it skipped; show the last lines first.
