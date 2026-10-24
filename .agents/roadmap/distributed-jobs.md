<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# distributed-jobs - roadmap

Scope: Submitting work that runs outside the controller and reconciling it after a restart - the immutable job specification, the launcher seam, a cluster simulated as local processes, and the durable record a controller observes (`splinter-core::job`, `splinter-sandbox::job`, `splinter-store::jobs`, `splinter-orchestrator::jobs`).

## Delivered

- [x] **A job is a value.** `JobSpec` names the program, the worker plan
      (nodes x ranks per node x accelerators), the content-addressed inputs,
      the output roles, a wall clock and a retry policy; its digest is the
      job's identity, and `{rank}`/`{world}` are the only per-worker
      substitution. A specification that names no program, no worker or no
      attempt is refused.
- [x] **A launcher starts detached workers.** `JobLauncher` has launch,
      observe and stop. `LocalLauncher` starts every rank as its own process
      in its own session, writes a launch directory once (so a retried
      submission cannot start a second set of workers), reports which ranks
      are alive, and can kill one rank while the others run. A launcher
      reopened after the controller died finds the same workers instead of
      starting more. A killed worker that its parent has not reaped is a
      zombie and reads as gone. Tests: `crates/sandbox/tests/jobs.rs`, with
      two real ranks and a controller that is dropped and reopened.
- [x] **Submission is idempotent and updates are fenced.** The specification
      and each record are content-addressed documents; the pointer
      `job/<spec digest>` is claimed with a compare-and-set from "never set",
      so two controllers submitting the same job produce one job, and a state
      update names the record it replaces, so neither overwrites a state it
      did not see.
- [x] **The controller states only what it can prove.** `JobController`
      submits, reconciles unfinished jobs against the launcher and launches the
      next attempt when the policy allows one. A clean exit completes the job;
      a nonzero exit fails it or moves it to `Retrying`; workers that are gone
      with no observed exit are `Lost`, never `Completed`. A record another
      controller moved first is left alone. Tests: the module tests in
      `crates/orchestrator/src/jobs.rs`.
- [x] **A job has an owner with an expiry.** A record carries a lease naming
      the controller and the instant it stops being valid. While a lease is
      live, another controller leaves the job alone; an expired one is taken
      over, and the takeover is fenced by the record compare-and-set. Every
      reconciliation and retry renews the lease.
- [x] **Jobs are a user-visible command.** `splinter jobs submit -- <program
      and args...>` submits one detached local job, with flags for the worker
      counts, the accelerators per worker, the wall clock (a worker past it
      is stopped and fails with 124) and the attempts;
      `jobs list`, `jobs reconcile`, `jobs retry`, `jobs show <id>` and
      `jobs cancel <id>` operate on the durable record. The SDK exposes the
      same through `Splinter::jobs()` (`submit`, `list`, `reconcile`, `retry`,
      `show`, `cancel`) on
      `LocalLauncher`; the controller name and the attempt root are
      configuration values (`--job-owner`, `--job-attempt-root`), and the
      binary is the supervisor each worker runs under. Tests:
      `crates/splinter/tests/jobs_cli.rs` (submit is idempotent, list and show
      report it, reconcile completes a clean exit and fails a nonzero one from
      another process, retry launches the next allowed attempt, cancel stops
      the workers).
- [x] **A worker's exit outlives the process that submitted it.** A launcher
      built with a supervisor starts each worker under it; the supervisor
      records the exit code in the attempt directory (written once, then
      renamed), so a controller that did not start the job reads completed or
      failed instead of lost, and a worker that outlives the specification's
      wall clock is stopped with its process group and recorded as 124. A
      launch without a supervised exit still reports no exit code rather than
      inventing one. Tests: `crates/sandbox/tests/jobs.rs` and the wall-clock
      case in `crates/splinter/tests/jobs_cli.rs`.
- [x] **A completed job's outputs become artifacts and lineage.** On reconcile,
      a clean exit reads the newest `commit-<8 digit step>.json` in the attempt
      directory, verifies every file it names against the declared length and
      SHA-256, keeps the bytes in the artifact store, and records one document
      linking the job specification, the attempt, the step, the plan text, the
      world size and each artifact's role, address and checksum. No manifest
      leaves the job completed with no outputs; a malformed manifest, a missing
      or altered file, or a path (or symlink) out of the attempt directory
      fails the job instead. Registration is idempotent: an artifact is
      addressed by its bytes and the document by its canonical form. Tests:
      the module tests in `crates/store/src/job_outputs.rs` and
      `crates/orchestrator/tests/controller.rs`.
- [x] **The checkpoint contract is versioned and held to fixtures.** `JobPlan`
      and `CheckpointManifest` (with `CheckpointFile`, `ManifestPath` and
      `Sha256Sum`) refuse unknown fields, absolute or `..` paths, and
      checksums that are not `sha256:<64 lowercase hex>`. The bytes a trainer
      writes live in `crates/orchestrator/tests/fixtures/` and a test parses
      both with the schemas, so a one-sided change fails.
- [x] **A scheduler's terminal state reaches the controller.**
      `Observation::Finished` carries `Completed`, `Failed`, `Cancelled`,
      `Preempted` or `Lost` and an optional exit code. A local exit maps zero
      to completed and nonzero to failed; Slurm maps COMPLETED, FAILED,
      NODE_FAIL and TIMEOUT, CANCELLED and PREEMPTED to the matching outcome.
      A preempted attempt retries only when `retry_on_preemption` is set and
      attempts remain; cancellation is terminal and only a completed outcome
      completes the job. Tests: `crates/sandbox/tests/jobs.rs`,
      `crates/sandbox/tests/slurm.rs` and
      `crates/orchestrator/tests/controller.rs`.
- [x] **A run that stopped heartbeating reads lost.** A recorded run writes
      append-only heartbeat signals while its process is alive; a reader
      reports `lost` once the newest is older than the stale bound, and a run
      recorded before heartbeats keeps the status its events ended with. The
      `runs list` and `runs show` views apply it, and the storage reference
      documents the status.

## Decisions

- The experience database and the artifact files stay the two authorities
  they already are: job metadata, attempts and leases are documents and a
  pointer in expdb, and the bulk files (adapters, datasets, checkpoints) are
  plain content-addressed files. No second database, and no scheduler-specific
  column in either.
- A job submission is identified by its specification digest, not by a
  scheduler job id. The scheduler id is an attribute of one attempt; losing
  it (a requeue, a controller restart) does not create a second job.

## Open

- [ ] No Kubernetes adapter. Slurm is reachable without code changes now
      (`--job-launcher slurm` with `--slurm-sbatch/-squeue/-scancel` and
      repeatable `--slurm-arg`; the orchestrator factory names a missing
      program by its setting) and is tested end to end against stand-in
      scheduler programs. A real cluster run is still unproven, and a
      scheduler's FAILED/TIMEOUT/PREEMPTED states read as `Lost` until the
      accounting path carries them.
- [ ] No pipeline stage runs a training job through the controller. The train
      stage still calls brain in-process. Brain's GPT trainer does accept a
      plan now (`--job-plan`/`--job-rank`); the Qwen/LoRA trainer does not yet.
- [ ] Artifact storage backends. Adapters and datasets are plain files at a
      content address; a parallel filesystem works as it is, while an object
      store needs a fetch path brain can read (brain resolves a local path
      today), not only another backend under Splinter.
