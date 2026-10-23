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

## Open

- [ ] No user-visible command submits a job yet: the controller is an API,
      and a `jobs` verb (submit, status, reconcile) with its documentation is
      the next step.
- [ ] No scheduler adapter. Slurm and Kubernetes implement the same
      `JobLauncher` trait; nothing in the controller names either.
- [ ] No pipeline stage runs a training job through the controller. The train
      stage still calls brain in-process, and brain's training verbs do not
      accept a job plan yet.
- [ ] Leases and heartbeats for a running job. The compare-and-set fences
      record updates, but nothing bounds how long a controller may claim to
      own a job before another may take it over.
- [ ] Artifact storage backends. Adapters and datasets are plain files at a
      content address; a parallel filesystem works as it is, while an object
      store needs a fetch path brain can read (brain resolves a local path
      today), not only another backend under Splinter.
- [ ] Run liveness: a run whose process died still reads `running`.
