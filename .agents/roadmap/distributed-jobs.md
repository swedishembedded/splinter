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
      zombie and reads as gone. Tests: `crates/sandbox/tests/it/jobs.rs`, with
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
      `crates/splinter/tests/it/jobs_cli.rs` (submit is idempotent, list and show
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
      inventing one. Tests: `crates/sandbox/tests/it/jobs.rs` and the wall-clock
      case in `crates/splinter/tests/it/jobs_cli.rs`.
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
      `crates/orchestrator/tests/it/controller.rs`.
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
      completes the job. Tests: `crates/sandbox/tests/it/jobs.rs`,
      `crates/sandbox/tests/it/slurm.rs` and
      `crates/orchestrator/tests/it/controller.rs`.
- [x] **A distributed training job is submitted with its plan.**
      `jobs submit --distributed` writes `job-plan.json` into each attempt
      directory before any worker starts, names the job by the specification
      digest in hex, takes its world size from the worker plan, allocates a free
      loopback coordinator per attempt, and appends `--job-plan <file>` and
      `--job-rank {rank}` to every worker command. The bytes it writes match
      the trainer's plan fixture exactly, a retry writes its own plan with the
      same job identity, and a single-rank job without the flag writes none.
      Tests: `crates/sandbox/tests/it/distributed_plan.rs`,
      `crates/orchestrator/tests/it/contracts.rs` and
      `crates/splinter/tests/it/jobs_distributed_cli.rs` (two stand-in ranks
      commit files, the controller completes the job, and both files are kept
      as artifacts whose sha256 matches).
- [x] **A local two-rank Qwen run completes through the controller.** A
      Qwen3-0.6B LoRA job (2 steps, rank 4, block 128, batch 1, one Tesla P40
      per rank) submitted with `--distributed --ranks-per-node 2` reached
      `completed` through the controller, wall time 15.9 s from submission
      through reconciliation. Both ranks read the generated plan and exited
      zero.
- [x] **A run that stopped heartbeating reads lost.** A recorded run writes
      append-only heartbeat signals while its process is alive; a reader
      reports `lost` once the newest is older than the stale bound, and a run
      recorded before heartbeats keeps the status its events ended with. The
      `runs list` and `runs show` views apply it, and the storage reference
      documents the status.

- [x] **A distributed Brain job is submitted with its plan.** `jobs submit
      --distributed` writes `job-plan.json` into the attempt directory before
      any worker starts (job id = the specification digest, world from the
      worker plan, loopback coordinator per attempt), appends
      `--job-plan`/`--job-rank`, and the launcher substitutes rank and world.
      A real two-rank Qwen3-0.6B LoRA run on one machine completed exit 0 in
      6.6 s from launch to record, with byte-identical adapters in `out0` and
      `out1`, rank 0's step-2 commit found beside the nested model-store copy,
      and the adapter registered as an artifact with a matching sha256
      (10,140,344 bytes). Tests: plan fixture bytes, plan-written-before-ranks,
      disjoint rank records, per-attempt retry, and a two-rank stand-in that
      commits and registers.
- [x] **A Slurm allocation is detected and can be restricted (2026-11-21).**
      `splinter_orchestrator::cluster::ClusterAllocation` is read from
      `SLURM_JOB_ID`, `SLURM_JOB_NODELIST` (compressed host lists expanded:
      ranges, lists, zero padding, products; malformed input refused by
      name), `SLURM_GPUS_ON_NODE`/`SLURM_JOB_GPUS` and `SLURMD_NODENAME`,
      only by `config::detect_cluster`, which the device flags call, so a
      test or an embedding that builds a `Config` stays single-machine inside
      an allocation. `--cluster auto|off`, `--cluster-nodes HOSTLIST` (refused
      outside the allocation), `--cluster-exec-prefix`, `--slurm-srun`,
      `--slurm-step-arg` and `--slurm-single-node-step-arg` shape it, and
      `splinter doctor` shows the job, its nodes, the GPUs per node, this
      node and the nodes a run uses. Measured inside job 5600435: `doctor
      --cluster-nodes n442` reports nodes n424,n442,n445,n467 with 4 GPUs
      each and uses n442. Tests: `crates/orchestrator/src/cluster.rs`,
      `crates/pipelines/tests/it/doctor.rs` (`cluster`), the device-flag and
      CLI flag tests.
- [x] **A brain server on every node of the allocation (2026-11-21).**
      With an allocation that spans nodes (or `--cluster-nodes`) and no
      `--brain-endpoint`, `orchestrator::brain_service::start_brain` (the one
      start path of the runtime and of the SDK's device flags) starts
      `ManagedBrain::start_fleet`: one `brain serve` per node as an `srun
      --overlap` step (`splinter_sandbox::step`: nodes, tasks and GPUs
      stated, the inherited `SLURM_GPUS` dropped, the step id recorded, the
      exec prefix applied), every card of the node as replicas, bound to the
      node's address on a system-picked port read back from the connection
      record, one owner-only key, shared adapters and artifact store, an
      `endpoints` file; attempt concurrency is the fleet's decode slots;
      teardown is `SIGTERM` to `srun`, then `scancel` of the step after a
      grace; a node that fails is named and the others are stopped.
      `doctor` lists the cards of every endpoint. Measured 2026-11-21 in job
      5600435 on n442 (brain bcd67e5ad, `doctor --cluster-nodes n442`): the
      step started, reported ready and listed n442's four cards from
      10.150.4.178 in 7.9 s wall from command start to exit, and left no
      process on n442. Tests: `crates/model/tests/service.rs` (`fleet`),
      `crates/sandbox/src/step.rs`, the doctor card tests.
- [x] **A distributed job runs across the allocation as one step
      (2026-11-21).** `--job-launcher auto` (the new default) is
      `slurm-step` inside an allocation that spans nodes:
      `splinter_sandbox::job::NodeStepLauncher` runs every rank as a task of
      ONE `srun --overlap` step (a Slingshot VNI is per step) on the first
      `nodes` usable nodes, `--ntasks-per-node` and `--gpus-per-node`
      stated, `--slurm-single-node-step-arg` for a one-node job, the `srun`
      under the supervisor so the step's exit is recorded; `{rank}`,
      `{local_rank}` and `{world}` map to `SLURM_PROCID`, `SLURM_LOCALID` and
      `SLURM_NTASKS` in the step script (brain reads no scheduler variable).
      The plan's coordinator is on rank 0's node's address, and
      `--distributed` writes a fresh owner-only `job.secret` per attempt and
      appends `--job-secret` (brain refuses a plan without one). Measured
      2026-11-21 in job 5600435 on n442 (brain bcd67e5ad, host TCP
      collective, `--network=single_node_vni`): a 4-rank Qwen3-4B LoRA
      (rank 8, 8 steps, batch 1, 16 two-turn records) ran as step
      5600435.142, 16 s from launch to exit, exit 0, the four rank adapters
      byte-identical (sha256 c632a989...), `jobs reconcile` completed it.
      Tests: `crates/sandbox/tests/it/distributed_plan.rs` (a stand-in
      `srun` running both ranks of one step), the launcher factory tests,
      `crates/core/src/job.rs`.
- [x] **The fleet serves TLS, ranks get the site's environment, brain says
      which actions join a plan, and `serve` hands the servers to other
      programs (2026-11-21).** A fleet makes a certificate authority of its
      own (name-constrained to its nodes' names and addresses), issues each
      node a certificate, discards the authority's key, serves
      `--tls-cert`/`--tls-key`, publishes `ca.pem`, and this process's model
      clients trust it through `sven_sdk::http::trust_model_authorities`
      (sven aa2e5b02); every Splinter request to brain is built by
      `sven_sdk::http::model_client_builder`. Plaintext only with
      `--cluster-insecure-plaintext`. `scripts/cluster/apptainer-step`
      enters an image on a node's host and `scripts/cluster/rank-env`
      sources the site's files inside the task and pins it to the CPUs of
      its card's NUMA node as sysfs says; `--cluster-rank-exec-prefix` is
      what a job's ranks start under (`{local_rank}` expands there too).
      `jobs submit --distributed` refuses a brain action that takes no
      `job_plan` per `brain caps --json` (the served listing omits
      host-resolved parameters), and `doctor` names those that do.
      `splinter serve` starts the servers, prints endpoints, key file and
      authority, and keeps them until interrupted. Measured 2026-11-21 in
      job 5600435 on n442+n445 (brain 7248e0604 from lane C2, Qwen3-4B):
      `doctor` listed the 8 cards over TLS in 11.2 s wall; `ask` answered
      in 17.9 s wall from fleet start to teardown; an 8-rank LoRA (rank 8,
      8 steps, batch 1, block 128, bf16 base) ran as one step
      (5600435.173) in 22 s from launch to exit, exit 0, the eight adapters
      byte-identical, brain reducing on the device over NCCL with "Using
      network Libfabric" and the cross-node channels via
      NET/Libfabric/*/GDRDMA; no process was left on either node. Tests:
      `crates/model/tests/service.rs` (TLS by default, plaintext by name),
      `crates/model/src/{fleet_tls,distributed}.rs`,
      `scripts/cluster/test_rank_env.py`, sven's
      `crates/http/tests/model_authorities.rs`.
- [x] **The flywheel loop serves on a fleet (2026-11-21).** Inside an
      allocation that spans nodes (or with `[serve.fleet]`; `serve.fleet =
      false` keeps per-card servers), `loop.py` starts one `splinter serve`
      for the whole campaign, serves every model there at once (the base,
      and each adapter written into the fleet's adapter directory and found
      by its sha256), runs bench with the fleet's `--endpoints-file` and
      `--endpoint-ca` (bench 2198163c), and trains through the fleet
      (`--brain-endpoint ... --brain-ca-file ... --brain-artifact-store
      ...`), so trainers take cards from its idle replicas. Measured
      2026-11-21 on n442 (brain built by lane C2, Qwen3-4B, a 4-task
      collection from sven-smoke): the fleet was ready 10 s after start,
      explore (2 tasks x 4 attempts) and the exam (2 tasks, 2/2 passed, 6
      model requests, all HTTP 200 over TLS) ran through bench's gateway in
      37 s, and the loop tore the fleet down with no process left. The
      training leg reached brain's `brain/train` over TLS and was refused
      ("selecting on the monitoring loss needs eval_every > 0"): an
      unmonitored supervised run (no monitoring group could be carved from
      a one-source dataset) still sent its patience. Fixed in
      `train::monitoring` (an unmonitored run gets patience 0 and keeps no
      evaluations); rerun on the fleet, learn took 69 s, the candidate's
      exam ran with its adapter served by the fleet, and the generation
      decided (champion kept, mean delta +0.000 over 2 paired tasks). The
      decide step had asked `bench report RUN --against BASE`, which bench
      parsed as `--against-package` and answered unpaired; it now runs
      `bench compare BASELINE CANDIDATE`, and bench refuses abbreviated
      options (bench d2c6a497). Also found: Python 3.13+ verifies certificates in
      X.509 strict mode, so fleet certificates now carry the authority key
      identifier, key usage and the server-auth purpose.
- [x] **Node-local model staging, used when the site announces a
      scratch (2026-11-21).** `--cluster-stage-models DIR|auto|off`
      (default `auto`): `config::detect_cluster` reads the node-local
      scratch the site announces to every step (`SLURM_TMPDIR`, `SNIC_TMP`,
      `LOCAL_SCRATCH`; on this cluster the task prolog sets
      `SNIC_TMP=/scratch/local` in every step, the host's job-private 1.7 TB
      disk, while `TMPDIR` is `/tmp`), and every fleet server and every brain
      rank of a job run as allocation steps passes brain's `--stage-models
      <scratch>/brain-models --stage-from <models-dir>` (brain 9eec9e86f),
      once one step on every node, under the exec prefix, made or found the
      directory writable (`NodeSteps::check_writable`). An announced scratch
      a node cannot write is dropped with the reason; a named one fails the
      run; nothing announced stays off. `doctor` shows the directory and its
      source, or why there is none. Tests: stand-in `srun` and brain in
      `crates/orchestrator/src/brain_service.rs`,
      `crates/sandbox/src/step.rs`, `crates/model/tests/service.rs`,
      `crates/sdk/src/jobs.rs`, the doctor and loop tests. Measured
      2026-11-21 in job 5600435 on n442 (4 cards, Qwen3-4B, a brain build
      with `--stage-models`): `splinter ask` through the TLS fleet staged
      the base under `/scratch/local/brain-models` (7.6 GB on the node) and
      answered in 26.4 s wall cold, 19.1 s warm, fleet start and teardown
      included, one run each; no process was left. brain's own measurement
      of a larger case: a 27B base on 8 ranks, 124.7 s to a finished step
      from the shared filesystem, 92.3 s staged cold, 54.5 s staged warm.
- [x] **The campaign's examiner examines on the fleet (2026-11-21).**
      Inside an allocation that spans nodes, a direct-completion `[examine]`
      first offers the candidate and its predecessor to every server of the
      fleet, then runs `bench examine` with the fleet's `--endpoints-file`,
      key and `--endpoint-ca` (bench 0c37ddae), and bench refuses unless
      every server lists exactly the model it examines; outside such an
      allocation bench serves on its own card as configured. Tests: the
      examiner's flag test, the campaign suite; bench's
      `tests/integration/test_examination_endpoints.py`. Not yet run on GPUs.
- [x] **The training stage trains one run over several cards and servers
      (2026-11-22).** Supervised and policy-gradient runs (`served_train`'s
      `run`) follow `BrainEndpoints::train_span`: one card; `cards` of one
      server (a one-node fleet, or `--brain-train-cards N`); or one job over
      every server (`job_servers`): the same `train` request to each, with
      its `cards`, its ranks of one plan (`first_rank` in server order,
      `dp` = world size), the SHA-256 of the job secret and an explicit
      seed; followed on every server, the result from rank 0's, a failure
      or cancel on any server cancelling the others (brain f137f5f94 and
      after). A fleet starts every server with a fresh owner-only
      `--train-job-secret`, the first with `--train-job-coordinator`, and
      reads `GET /v1/capabilities` for `cards` and `job` before it spreads
      runs (one card otherwise, said). `doctor` says which mode applies.
      Tests: `served_train/job_servers.rs` against scripted servers (fields
      per server, rank offsets, secret digest, a failure on one server, a
      cancel), the fleet's flags and span, the device flags. `splinter
      serve` publishes the span (`train_cards`, the job secret, the
      coordinator) and the flywheel loop passes it on with
      `--brain-train-*` (before that its training through a fleet took one
      card). Measured 2026-11-22 on n442 (brain 1fc6074a2, Qwen3-4B, a
      9-record dataset from the flywheel's tiny campaign, 4 steps, rank 8,
      bf16 base): one run as one job over two servers on the node (gpu0-1
      holding ranks 0-1, gpu2 rank 2), brain reducing on the device over
      NCCL (3 ranks), `splinter train` 82 s wall with exit 0, nothing left
      running. The loop on a one-node fleet with all four cards is not yet
      measured: its candidate step found the trained adapter unserved, a
      brain defect with `--stage-models` (the staged checkpoint and the
      store's entry are served as two bases, so a served adapter's base is
      ambiguous), reported to brain with a CPU reproduction.

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
      scheduler programs. A real cluster run is still unproven; the adapter
      maps the scheduler's terminal states to distinct outcomes, and job
      accounting (exit codes from a scheduler) is not wired yet.
- [ ] A local-server test failed once under load and never again:
      `crates/model/tests/service.rs`
      `the_server_a_run_owns_serves_the_store_..._as_wide_as_it_decodes`
      failed in one `cargo test -p splinter-model --tests` run on n424
      (2026-11-21, while other test binaries ran at once) and passed in 9
      runs after, alone and with the other binaries; the panic text was not
      kept. Suspects to rule out when it recurs: the stand-in writes its
      arguments file with `echo > $0.args` after it is spawned, and the
      test reads it right after `start` returns, so a slow shell could leave
      it empty; keep the panic line.
- [ ] `--job-launcher slurm` (sbatch) still registers a plan's ranks on
      the loopback: rank 0's node is not known at submission.
- [ ] Artifact storage backends. Adapters and datasets are plain files at a
      content address; a parallel filesystem works as it is, while an object
      store needs a fetch path brain can read (brain resolves a local path
      today), not only another backend under Splinter.
