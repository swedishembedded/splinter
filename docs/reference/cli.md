<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Command reference

Generated from the command-line grammar; do not edit it. `splinter <command> --help`
prints the same text for one command. Worked examples are in the guides; this page
lists every command, argument and option.

## Global options

Every command takes these.

- `--device <DEVICE>`: Which compute the brain server may schedule: `gpu`, `gpu1`, `gpu0,gpu1`, `cpu0-7`, `vulkan`, ... in brain's grammar. Without it, every device the machine has.
- `--backend <BACKEND>`: How the hardware is driven: wgpu, vulkan, cuda or cpu. Without it brain picks, which on a CUDA host is not CUDA.
- `--vram-budget <VRAM_BUDGET>`: The VRAM, in GiB, this process was granted: every card the server budgets from is capped at it. Without it the whole card is budgeted.
- `--brain-endpoint <BRAIN_ENDPOINT>...`: A running `brain serve --openai` to request every local model from (its base URL, ending in `/v1`); repeat for more. Without one, a server is started from `--brain-bin`.
- `--brain-key-file <BRAIN_KEY_FILE>`: The file holding the endpoints' API key.
- `--brain-ca-file <FILE>`: A PEM file of certificate authorities the endpoints' https:// certificates may chain to (the `ca.pem` a fleet that serves TLS under its own authority publishes); added to the trusted roots, never in place of verification.
- `--brain-train-cards <N[,N...]>...`: With --brain-endpoint: the cards a data-parallel training run takes on each endpoint, in their order (`2,1`); one number for one endpoint trains on that many of its cards, several make one job over the endpoints, which needs --brain-train-job-secret and --brain-train-coordinator. Without it a run takes one card.
- `--brain-train-job-secret <FILE>`: The job secret the endpoints were started with (`brain serve --train-job-secret FILE`, 64 hex digits)
- `--brain-train-coordinator <HOST:PORT>`: The address the first endpoint's server binds as the job's coordinator (`brain serve --train-job-coordinator HOST:PORT`)
- `--brain-bin <BRAIN_BIN>`: The `brain` binary a server is started from (default: the first on `PATH`)
- `--cluster <MODE>`: Whether a Slurm allocation this process runs inside is used: `auto` (default) serves models on every usable node of a multi-node allocation and spreads a distributed job's ranks over them; `off` keeps everything on this machine.
- `--cluster-nodes <HOSTLIST>`: The nodes of the allocation this process may use, as a Slurm host list (`n[442,445]`); default: every node of the allocation. Naming nodes uses them even when only one is named.
- `--cluster-exec-prefix <WORDS>`: The words a program started on a node runs under, split at spaces: a step starts on the node's host, so a container this process runs in is entered again there (`apptainer exec --nv IMAGE`, or a wrapper script that also sets the node's environment)
- `--cluster-rank-exec-prefix <WORDS>`: The words a distributed job's ranks are started under instead of --cluster-exec-prefix, split at spaces; `{local_rank}` in them is a rank's place on its node, the card a wrapper pins it to.
- `--slurm-srun <PATH>`: The `srun` program steps on the allocation's nodes are started with (default: the first on `PATH`)
- `--slurm-step-arg <ARG>...`: An argument every step passes to `srun` (`--mem=64G`); repeat for more.
- `--slurm-single-node-step-arg <ARG>...`: An argument a step on a single node passes to `srun` besides `--slurm-step-arg` (on an HPE Slingshot fabric, `--network=single_node_vni`, without which a single-node step has no fabric access); repeat for more.
- `--cluster-insecure-plaintext`: Serve plaintext HTTP between the allocation's nodes instead of TLS: only on a cluster network you trust, since the servers' key then crosses it readable. By default the servers serve TLS under a certificate authority made for them, whose key is discarded at once.
- `--cluster-stage-models <DIR|auto|off>`: Where brain stages the model store's entries on each node's own disk for the fleet's servers and a distributed job's brain ranks (the first process of a node copies an entry, verified; the others read the copy instead of the shared filesystem): `auto` (default) under the node-local scratch the site announces to every step (SLURM_TMPDIR, SNIC_TMP or LOCAL_SCRATCH), checked writable on every node, else not at all; an absolute directory there; `off` never. The directory must be bound into the container a step enters.
- `--config <FILE>`: The machine configuration file: connections and operator grants (default: `$SPLINTER_CONFIG`, else `$XDG_CONFIG_HOME/splinter/config.yaml`, else `~/.config/splinter/config.yaml`). See `splinter config`.
- `--state <DIR>`: The state root (default: ~/.splinter)
- `--job-owner <NAME>`: The controller name recorded with each job this process submits and the claim it takes on one (default: this machine's host name)
- `--detach-on-interrupt`: On Ctrl-C (or SIGTERM), exit at once and leave the brain training jobs of the command running, instead of cancelling them through brain's API (`DELETE /v1/jobs/{id}`) first; running the same command again continues each job from its last saved state. A second Ctrl-C always exits at once.
- `--job-attempt-root <DIR>`: Where a submitted job's attempt directories are written (default: `jobs` under the state root)
- `--job-launcher <KIND>`: How each submitted job's workers are started: `local` runs them on this machine, `slurm` submits the job to Slurm's batch system, `slurm-step` runs every rank as a task of one step on the usable nodes of the allocation this process runs inside (default: `auto`, which is `slurm-step` inside an allocation that spans nodes and `local` otherwise)
- `--slurm-sbatch <PATH>`: The `sbatch` program a `slurm` job is submitted with; required when `--job-launcher slurm` is given.
- `--slurm-squeue <PATH>`: The `squeue` program that reports a `slurm` job's state; required when `--job-launcher slurm` is given.
- `--slurm-scancel <PATH>`: The `scancel` program that stops a `slurm` job; required when `--job-launcher slurm` is given.
- `--slurm-arg <ARG>...`: An argument every `slurm` submission passes to `sbatch` before the script (for example `--partition=debug`); repeat for more.
- `--models-dir <DIR>`: brain's model store, where `local:<org>/<model>` names a checkpoint (default: the store `brain pull` fills)
- `--policy-base <CHECKPOINT>`: The checkpoint the policy trains and `policy:default` serves: a path, or a name in the model store (default: Qwen/Qwen3-0.6B)
- `--assistant-model <REF>`: The model that plans a learning run, writes its tasks, teaches what the policy cannot answer and judges, when the command names none for a role (default: the strongest chat model in the model store that fits a card, else the policy)
- `--judge-model <REF>`: The model that judges answers in place of the assistant, so the judge is another model than the one that wrote what it grades.
- `--front-door-model <REF>`: The model that reads a sentence (`splinter "learn ..."`) in place of the assistant.
- `--min-calibration-controls <N>`: The fewest controls a judge is measured on before its verdicts count [default: 16]
- `--docker <PATH>`: A docker client: with --workspace-image, a project's commands run in a container with no network.
- `--workspace-image <IMAGE>`: The image, pinned by digest (`name@sha256:...`), a project's commands run in under --docker.
- `--workspace-scratch <DIR>`: Where trees are laid out for repository tasks (default: under the state root)
- `--cargo-home <DIR>`: A cargo home the commands of a Cargo project build offline from.
- `--bf16-base`: Hold the frozen base the policy trains at bf16, half the bytes of fp32: what a 7B base needs to train on one 24 GiB card. A fact about this machine, so it applies to every command that trains, a sentence's run included.
- `--gpu-contexts <N>`: The foreign GPU contexts that existed when this command started, as the GPU lease counted them; recorded with the run.
- `--json`: Print machine-readable JSON on stdout instead of text.
- `-v, --verbose`: Report more as work goes (each stage's full summary)
- `--allow-remote`: Permit models reached over the network (remote:...); without it every model runs locally.
- `--policy-context-tokens <TOKENS>`: Limit the policy model's context to this many tokens (default: the largest its checkpoint supports); for a policy too large for the card at that size. Another model is limited by its reference, `local:<checkpoint>@<tokens>`.
- `--remote-concurrency <N>`: How many requests to a model reached over an API may be in flight at once [default: 4]
- `--brain-adapter-manifest <FILE>`: The release manifest the brain endpoints follow (`brain serve --adapter-manifest FILE`): an adapter they do not list, such as a release candidate, is offered by publishing it there and waited for. A manifest offers one adapter at a time.
- `--brain-adapter-dir <DIR>`: The adapter directory the brain endpoints serve (`brain serve --serve-adapters DIR`): an adapter they do not list is offered by writing it there and waited for. Every adapter written there is a model of its own, so a champion and its candidate are served at once, as an examination of the pair needs.
- `--brain-artifact-store <DIR>`: The artifact store the brain endpoints train through (`brain serve --training --artifact-store DIR`): a directory both they and this machine reach, in which datasets are published, named by their BLAKE3 digest, and trained adapters are found. With endpoints, a supervised or policy-gradient training run is a job of the service and needs it.
- `--attempt-concurrency <N>`: How many attempts are asked of the brain server at once, in all (set to its --max-batch) [default: 16 per endpoint]
- `--solve-deadline <SECONDS>`: How many seconds one task's solve may take on the wall clock [default: 300]
- `--test-output-cap <BYTES>`: Bytes kept of each output stream of a test run [default: 1048576]
- `--allow-unconfined-workspace`: Let a project's commands (the solver's shell, the test runs) see the whole filesystem where the kernel cannot confine them to their tree. Without it such a run is refused; the environment record says when it was allowed.
- `--allow-workspace-network`: Let a project's commands (the solver's shell, the test runs) use the network. Without it they cannot connect anywhere, a model server on the loopback included; the environment record says when it was allowed.
- `--workspace-max-processes <N>`: Bound the processes and threads one project command may start. Part of the environment record, so tasks generated without it name a different environment.
- `--workspace-env <NAME=VALUE>...`: A variable every project command runs with (for example `PYTHONDONTWRITEBYTECODE=1`, so a Python project leaves no bytecode caches in the tree the solver reads); repeat for more. Part of the environment record, so tasks generated with it name a different environment than tasks generated without it.
- `--workspace-memory-gib <GIB>`: Bound the address space of each process of a project command, in GiB (32 when not given). A command that maps more fails to allocate. Part of the environment record, so tasks generated with a bound name a different environment from those without.
- `--workspace-tree-memory-gib <GIB>`: Bound the memory the processes of one project command hold together, in GiB (24 when not given); 0 names no bound, which offers the environment of the tasks stored before the bound existed (they are refused otherwise). Part of the environment record.
- `--runtime-dir <DIR>...`: A directory searched before the system's for the interpreters code runs in (`lua` is Lua 5.1 and no other: `make deps/lua` builds the pinned one); repeat for more.
- `--temperature <T>`: The sampling temperature, 0 to 2, asked of a model reached through sven's providers (the provider's own default when not given). The same flag sets the pass@k sampling of a verb that measures it; when given twice, the last wins.
- `--top-p <P>`: Nucleus sampling (`top_p`, above 0 up to 1) asked of a model reached through sven's providers; the server's own default when not given.
- `--top-k <K>`: Sampling among the K most likely tokens (`top_k`) asked of a model reached through sven's providers; the server's own default when not given.
- `--sampling-seed <SEED>`: The seed every request of a model reached through sven's providers carries, so that a sampled attempt is reproducible; recorded with the attempt. The server draws its own when not given.
- `--thinking`: Let a model served on this machine reason before it answers (the request says so); without it the server is asked for no reasoning block. Recorded with the attempt.
- `--thinking-budget <N>`: With --thinking: the most tokens the model may think in one response; the server closes the thought there and the model goes on to its answer. Recorded with the attempt. Unbounded when not given.
- `--coding-prompt <ID>`: The coding-agent system prompt a solver of a repository works under: `coding-agent-1` or `coding-agent-2`. The registered protocol's prompt when not given; the prompt's id and digest are recorded with each attempt.

## Commands

- [splinter corrections](#splinter-corrections) - Continue the attempts that failed or got stuck of an existing experience set with a teacher, from the state the student actually reached, and keep the continuations the task's own hidden checks verify as an experience set a dataset can be built from
- [splinter ask](#splinter-ask) - Answer a question with a model, closed-book unless --open-book
- [splinter status](#splinter-status) - Show the policy in use, recent runs and what the stores hold
- [splinter init](#splinter-init) - Write a template, a task repository that works as written, into a directory
- [splinter templates](#splinter-templates) - List the templates `init` writes
- [splinter validate](#splinter-validate) - Check a factory recipe and everything it names, reporting every problem at once
- [splinter plan](#splinter-plan) - Show the graph of stages a factory recipe runs and the work each does
- [splinter generate](#splinter-generate) - Write a factory collection's task directories with the command its file names (the task repository's own compiler)
- [splinter run](#splinter-run) - Run a factory recipe into a run directory, each stage reused when its product is already committed
- [splinter resume](#splinter-resume) - Continue a factory run from what it recorded
- [splinter cancel](#splinter-cancel) - Ask a running factory run to stop at its next stage or batch
- [splinter inspect](#splinter-inspect) - Show a factory run's products and reports, or open its trials
- [splinter view](#splinter-view) - Open a factory run in the browser
- [splinter fork](#splinter-fork) - Start a factory run on a changed recipe, reusing the products of an earlier run that the change leaves unchanged
- [splinter report](#splinter-report) - Print a factory run's report
- [splinter export](#splinter-export) - Write a factory run's products as a portable bundle
- [splinter index](#splinter-index) - File a factory run's trials in the experience database (an optional index across runs
- [splinter play](#splinter-play) - Play one session of a factory run's interactive pass on this terminal
- [splinter doctor](#splinter-doctor) - Check that a run can start here
- [splinter serve](#splinter-serve) - Start the brain servers (one per usable node of a multi-node allocation, else one here), print where they are and keep them until interrupted
- [splinter baseline](#splinter-baseline) - Print the `baseline-v1` record of what a registered run runs on
- [splinter source](#splinter-source) - Capture and inspect sources
  - [splinter source add](#splinter-source-add) - Capture a file (a document), a directory (a repository) or cmd:COMMAND
  - [splinter source list](#splinter-source-list) - List the stored sources
  - [splinter source show](#splinter-source-show) - Show one stored source
- [splinter env](#splinter-env) - Compile a learning objective into validated, versioned training environments
  - [splinter env classify](#splinter-env-classify) - Say how each skill of an objective could be verified (an objective check, a proxy, a calibrated judge or human evidence), with the reasons and the plugin that would build it
  - [splinter env plugins](#splinter-env-plugins) - List the environment plugins, their difficulty knobs and what each verifies by
  - [splinter env compile](#splinter-env-compile) - Compile an objective into environments
  - [splinter env list](#splinter-env-list) - List the environments in a store with their status
  - [splinter env show](#splinter-env-show) - Show one environment's admission report
  - [splinter env export](#splinter-env-export) - Write an environment's stored instances of one split as Harbor task folders, one directory each under --out, for a run that needs them on disk (`make env/measure`)
  - [splinter env band](#splinter-env-band) - Record a base policy's measured pass rate on an environment's dev instances and judge it
- [splinter objects](#splinter-objects) - Knowledge objects (rules, thresholds, deadlines, exceptions, facts, procedure steps) a model proposes from documents, admitted by code
  - [splinter objects propose](#splinter-objects-propose) - Ask a model for the typed objects each document of a directory states, one document at a time in date order
- [splinter experiment](#splinter-experiment) - Check an experiment plan
  - [splinter experiment check](#splinter-experiment-check) - Validate a plan against the catalog of families and show its schedule
- [splinter absorb](#splinter-absorb) - Learn from the sessions you held with an agent
- [splinter consolidate](#splinter-consolidate) - Take the generation the policy is in into the weights
- [splinter improve](#splinter-improve) - Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs
- [splinter session](#splinter-session) - Take in recorded agent sessions, secrets removed
  - [splinter session add](#splinter-session-add) - Take recorded agent sessions in as sources
  - [splinter session list](#splinter-session-list) - List the stored sessions
- [splinter claims](#splinter-claims) - What a person taught in sessions
  - [splinter claims extract](#splinter-claims-extract) - Have a model propose the claims each stored session teaches
  - [splinter claims gate](#splinter-claims-gate) - Rule on a claim set
  - [splinter claims forget](#splinter-claims-forget) - Forget a live claim
  - [splinter claims list](#splinter-claims-list) - List the stored claim sets
  - [splinter claims show](#splinter-claims-show) - Show a claim set
  - [splinter claims ledger](#splinter-claims-ledger) - Show the ledger
- [splinter tasks](#splinter-tasks) - Generate and inspect task sets
  - [splinter tasks history](#splinter-tasks-history) - Read tasks from a repository's history
  - [splinter tasks variants](#splinter-tasks-variants) - Write differently worded questions about each task of a task set, into a task set of their own
  - [splinter tasks combine](#splinter-tasks-combine) - Make harder tasks of the admitted single-fault tasks of a task set
  - [splinter tasks drills](#splinter-tasks-drills) - Make comprehension drills from a repository's clean snapshot
  - [splinter tasks pick](#splinter-tasks-pick) - Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking)
  - [splinter tasks check-requests](#splinter-tasks-check-requests) - Drop the tasks whose request does not state the contract their hidden check enforces (a returned value or a raised error that no failing test's title speaks of, so the request cannot lead to the fix)
  - [splinter tasks harbor](#splinter-tasks-harbor) - Harbor tasks
    - [splinter tasks harbor discover](#splinter-tasks-harbor-discover) - List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in
    - [splinter tasks harbor import](#splinter-tasks-harbor-import) - Import Harbor tasks (each a task directory) as workspace tasks in one task set
    - [splinter tasks harbor export](#splinter-tasks-harbor-export) - Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly
    - [splinter tasks harbor episodes](#splinter-tasks-harbor-episodes) - Run a directory of practice task folders (or one folder) as episodes
  - [splinter tasks list](#splinter-tasks-list) - List the stored task sets
  - [splinter tasks show](#splinter-tasks-show) - Show a task set's tasks, or one task
- [splinter verify](#splinter-verify) - Check a factory run
- [splinter critique](#splinter-critique) - Critique the failed experiences of a set and retry them with it
- [splinter judge](#splinter-judge) - Measure a judge against labelled experiences
  - [splinter judge calibrate](#splinter-judge-calibrate) - Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by
  - [splinter judge measure](#splinter-judge-measure) - Measure a judge on the controls a task set's own references give
- [splinter experiences](#splinter-experiences) - Inspect experience sets and experiences
  - [splinter experiences list](#splinter-experiences-list) - List the stored experience sets
  - [splinter experiences show](#splinter-experiences-show) - Show an experience set, or one experience with its annotations
  - [splinter experiences replay](#splinter-experiences-replay) - Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed
  - [splinter experiences ingest-harbor](#splinter-experiences-ingest-harbor) - Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict
  - [splinter experiences ingest-bench](#splinter-experiences-ingest-bench) - Admit the trials of exports written by `bench export` as experiences
  - [splinter experiences diagnose](#splinter-experiences-diagnose) - Why each failed trial of bench exports failed
  - [splinter experiences capabilities](#splinter-experiences-capabilities) - A policy's competence per capability of the examiner's taxonomy, from bench exports or practice runs
  - [splinter experiences import](#splinter-experiences-import) - Import experience sets, with their experiences' annotations and trial register, from another state root
  - [splinter experiences ingest-hf](#splinter-experiences-ingest-hf) - Ingest a bundle `publish` wrote, at a pinned revision of its Hugging Face dataset repository
  - [splinter experiences compare](#splinter-experiences-compare) - Compare two experience sets of the same tasks
  - [splinter experiences compare-arms](#splinter-experiences-compare-arms) - Compare graded trials of several arms (Harbor jobs or trials files) with a reference arm
- [splinter dataset](#splinter-dataset) - Build datasets from experience sets, and export them
  - [splinter dataset conversations](#splinter-dataset-conversations) - Build a policy-gradient dataset from graded conversations with a counterpart (a client, a student)
  - [splinter dataset decision-fixtures](#splinter-dataset-decision-fixtures) - Write labelled decision fixtures (one example per labelled choice question) from the run records of an ablation benchmark output directory, split by task family and seed so that no task is in two splits
  - [splinter dataset check](#splinter-dataset-check) - Have the trainer read a stored dataset before any device is claimed
  - [splinter dataset verify-speech](#splinter-dataset-verify-speech) - Hear every recording of an audio dataset (a Hugging Face audio folder's metadata.jsonl) through a recogniser and check it says its transcript
  - [splinter dataset export](#splinter-dataset-export) - Copy a dataset and its manifest into a directory
- [splinter goal](#splinter-goal) - Goal-directed learning
  - [splinter goal admit](#splinter-goal-admit) - Admit documents into a selection of learning material
- [splinter release](#splinter-release) - The releases and the aliases pointing at them (a factory run's `promotion` stage makes a release)
  - [splinter release list](#splinter-release-list) - List the releases and the aliases pointing at them
- [splinter rollback](#splinter-rollback) - Point an alias back at the release its current one was trained from
- [splinter eval](#splinter-eval) - Evaluate a factory run's candidates under another plan into a new run (`RUN --plan PLAN --out DIR`)
  - [splinter eval matrix](#splinter-eval-matrix) - The continual-learning results matrix of a lineage
- [splinter exam](#splinter-exam) - Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them
- [splinter select](#splinter-select) - Choose the step of a run kept with its evaluations by what each writes on the dev suite
- [splinter exam-report](#splinter-exam-report) - Analyses done after an exam from what its report kept
  - [splinter exam-report labels-export](#splinter-exam-report-labels-export) - Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said
  - [splinter exam-report labels-import](#splinter-exam-report-labels-import) - What a person's labels say of the judge
  - [splinter exam-report memorisation](#splinter-exam-report-memorisation) - How much of each arm's answers is the text the candidate was trained to produce again
- [splinter exam-set](#splinter-exam-set) - Make and show frozen exams
  - [splinter exam-set create](#splinter-exam-set-create) - Capture the sources, reserve families of them, write the exam from that text alone and freeze it
  - [splinter exam-set power](#splinter-exam-set-power) - Simulate the planned paired test to size an exam before it is paid for
  - [splinter exam-set show](#splinter-exam-set-show) - Show a frozen exam
- [splinter runs](#splinter-runs) - List, inspect and cancel runs
  - [splinter runs list](#splinter-runs-list) - List the recorded runs
  - [splinter runs show](#splinter-runs-show) - Show one run's record
  - [splinter runs cancel](#splinter-runs-cancel) - Ask a run in progress to stop
- [splinter jobs](#splinter-jobs) - Submit work that runs outside this process, list it, reconcile it after a restart, retry a failed attempt, show one job or cancel one
  - [splinter jobs submit](#splinter-jobs-submit) - Submit a program as one job and return as soon as its workers are started and detached
  - [splinter jobs list](#splinter-jobs-list) - List the submitted jobs and the state each was last seen in
  - [splinter jobs reconcile](#splinter-jobs-reconcile) - Ask each unfinished job's launcher what became of it and record the answer
  - [splinter jobs retry](#splinter-jobs-retry) - Start the next attempt of every job whose last attempt failed and whose retry policy allows another
  - [splinter jobs show](#splinter-jobs-show) - Show one job, its state and the specification it runs
  - [splinter jobs cancel](#splinter-jobs-cancel) - Stop a job's workers and record it cancelled
- [splinter state](#splinter-state) - Show, maintain, verify, repair, archive and restore the state
  - [splinter state status](#splinter-state-status) - What the experience database holds, as files
  - [splinter state maintain](#splinter-state-maintain) - Merge small files, index what is not indexed and retire finished writers
  - [splinter state unpin](#splinter-state-unpin) - Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected
  - [splinter state verify](#splinter-state-verify) - Check the database and every file it tracks, and report each one that is missing or damaged
  - [splinter state repair](#splinter-state-repair) - Recover what verify finds
  - [splinter state archive](#splinter-state-archive) - Pack the database and the files it tracks into one archive
  - [splinter state pack](#splinter-state-pack) - Rewrite the experience database into the packed layout
  - [splinter state restore](#splinter-state-restore) - Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first
- [splinter lineage](#splinter-lineage) - Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it
- [splinter publish](#splinter-publish) - Publish experience sets, datasets, candidates and signed aggregate verdicts as one tagged, immutable revision of a private Hugging Face dataset repository
- [splinter config](#splinter-config) - Read and change this machine's settings (connections and operator grants), one validated file
  - [splinter config show](#splinter-config-show) - Print the file's path and its effective content
  - [splinter config get](#splinter-config-get) - Print one value
  - [splinter config set](#splinter-config-set) - Set one value, given as YAML (`set connections.hosted.token env:SVEN_KEY`, `set connections.brain '{protocol
  - [splinter config reset](#splinter-config-reset) - Remove one value, or with --all return the file to the empty default
  - [splinter config schema](#splinter-config-schema) - Print the JSON Schema of the file
  - [splinter config path](#splinter-config-path) - Print the file's path
  - [splinter config import](#splinter-config-import) - Move the settings of the older `connections.yaml` and `operators.yaml` beside the file into it, then remove them

## splinter

A learning agent with its own model. Tell it what to learn - a document, a repository, a command's output - and it generates tasks from it, solves and verifies them, trains a candidate on what passed, and releases it only if it measurably improves without forgetting. With no command, a REPL on the current policy: each line is handled exactly like `splinter "<line>"`.

```text
splinter [OPTIONS] [SENTENCE] [COMMAND]
```

Arguments:

- `<SENTENCE>`: A sentence, classified by the policy into one of the commands below; an unclear or unconfident reading is asked back, never run.

Subcommands:

- [`corrections`](#splinter-corrections) - Continue the attempts that failed or got stuck of an existing experience set with a teacher, from the state the student actually reached, and keep the continuations the task's own hidden checks verify as an experience set a dataset can be built from
- [`ask`](#splinter-ask) - Answer a question with a model, closed-book unless --open-book
- [`status`](#splinter-status) - Show the policy in use, recent runs and what the stores hold
- [`init`](#splinter-init) - Write a template, a task repository that works as written, into a directory
- [`templates`](#splinter-templates) - List the templates `init` writes
- [`validate`](#splinter-validate) - Check a factory recipe and everything it names, reporting every problem at once
- [`plan`](#splinter-plan) - Show the graph of stages a factory recipe runs and the work each does
- [`generate`](#splinter-generate) - Write a factory collection's task directories with the command its file names (the task repository's own compiler)
- [`run`](#splinter-run) - Run a factory recipe into a run directory, each stage reused when its product is already committed
- [`resume`](#splinter-resume) - Continue a factory run from what it recorded
- [`cancel`](#splinter-cancel) - Ask a running factory run to stop at its next stage or batch
- [`inspect`](#splinter-inspect) - Show a factory run's products and reports, or open its trials
- [`view`](#splinter-view) - Open a factory run in the browser
- [`fork`](#splinter-fork) - Start a factory run on a changed recipe, reusing the products of an earlier run that the change leaves unchanged
- [`report`](#splinter-report) - Print a factory run's report
- [`export`](#splinter-export) - Write a factory run's products as a portable bundle
- [`index`](#splinter-index) - File a factory run's trials in the experience database (an optional index across runs
- [`play`](#splinter-play) - Play one session of a factory run's interactive pass on this terminal
- [`doctor`](#splinter-doctor) - Check that a run can start here
- [`serve`](#splinter-serve) - Start the brain servers (one per usable node of a multi-node allocation, else one here), print where they are and keep them until interrupted
- [`baseline`](#splinter-baseline) - Print the `baseline-v1` record of what a registered run runs on
- [`source`](#splinter-source) - Capture and inspect sources
- [`env`](#splinter-env) - Compile a learning objective into validated, versioned training environments
- [`objects`](#splinter-objects) - Knowledge objects (rules, thresholds, deadlines, exceptions, facts, procedure steps) a model proposes from documents, admitted by code
- [`experiment`](#splinter-experiment) - Check an experiment plan
- [`absorb`](#splinter-absorb) - Learn from the sessions you held with an agent
- [`consolidate`](#splinter-consolidate) - Take the generation the policy is in into the weights
- [`improve`](#splinter-improve) - Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs
- [`session`](#splinter-session) - Take in recorded agent sessions, secrets removed
- [`claims`](#splinter-claims) - What a person taught in sessions
- [`tasks`](#splinter-tasks) - Generate and inspect task sets
- [`verify`](#splinter-verify) - Check a factory run
- [`critique`](#splinter-critique) - Critique the failed experiences of a set and retry them with it
- [`judge`](#splinter-judge) - Measure a judge against labelled experiences
- [`experiences`](#splinter-experiences) - Inspect experience sets and experiences
- [`dataset`](#splinter-dataset) - Build datasets from experience sets, and export them
- [`goal`](#splinter-goal) - Goal-directed learning
- [`release`](#splinter-release) - The releases and the aliases pointing at them (a factory run's `promotion` stage makes a release)
- [`rollback`](#splinter-rollback) - Point an alias back at the release its current one was trained from
- [`eval`](#splinter-eval) - Evaluate a factory run's candidates under another plan into a new run (`RUN --plan PLAN --out DIR`)
- [`exam`](#splinter-exam) - Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them
- [`select`](#splinter-select) - Choose the step of a run kept with its evaluations by what each writes on the dev suite
- [`exam-report`](#splinter-exam-report) - Analyses done after an exam from what its report kept
- [`exam-set`](#splinter-exam-set) - Make and show frozen exams
- [`runs`](#splinter-runs) - List, inspect and cancel runs
- [`jobs`](#splinter-jobs) - Submit work that runs outside this process, list it, reconcile it after a restart, retry a failed attempt, show one job or cancel one
- [`state`](#splinter-state) - Show, maintain, verify, repair, archive and restore the state
- [`lineage`](#splinter-lineage) - Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it
- [`publish`](#splinter-publish) - Publish experience sets, datasets, candidates and signed aggregate verdicts as one tagged, immutable revision of a private Hugging Face dataset repository
- [`config`](#splinter-config) - Read and change this machine's settings (connections and operator grants), one validated file

### splinter corrections

Continue the attempts that failed or got stuck of an existing experience set with a teacher, from the state the student actually reached, and keep the continuations the task's own hidden checks verify as an experience set a dataset can be built from

```text
splinter corrections [OPTIONS] --attempts <EXPERIENCE-SET> --tasks <TASK-SET> --teacher <REF>
```

Options:

- `--attempts <EXPERIENCE-SET>` (required): The experience set of the attempts to continue, by id or unique prefix.
- `--tasks <TASK-SET>` (required): The task set the attempts are of, by id or unique prefix.
- `--teacher <REF>` (required): The model that continues them (`remote:`/`local:`)
- `--max <N>`: The most attempts this run continues, one per task [default: 8]
- `--max-tool-calls <N>`: The calls one continuation may make: the student's own limit, which the remaining budget is counted against.
- `--budget <DUR>`: Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.

### splinter ask

Answer a question with a model, closed-book unless --open-book

```text
splinter ask [OPTIONS] <QUESTION>
```

Arguments:

- `<QUESTION>` (required): The question.

Options:

- `--open-book <SOURCE-ID>`: Show the model this stored source's text with the question.
- `--retrieve <SOURCE-ID>...`: Show the model the passages of these stored sources that bear on the question, found by meaning and by exact words, instead of all of one.
- `--passages <N>`: How many passages `--retrieve` shows. Default: `6`.
- `--reranker <REF>`: A model that reads the candidates search found beside the question and moves the ones that bear on it forward.
- `--policy <REF>`: The model asked. Default: `policy:default`.

### splinter status

Show the policy in use, recent runs and what the stores hold; with a factory run directory, the run's stages and slots

```text
splinter status [OPTIONS] [RUN]
```

Arguments:

- `<RUN>`: A factory run directory; none shows the policy.

### splinter init

Write a template, a task repository that works as written, into a directory; writes over no file

```text
splinter init [OPTIONS] --template <NAME> [DIR]
```

Arguments:

- `<DIR>`: The directory to write it into; created when missing. Default: `.`.

Options:

- `--template <NAME>` (required): The template (`splinter templates` lists them)

### splinter templates

List the templates `init` writes

```text
splinter templates [OPTIONS]
```

### splinter validate

Check a factory recipe and everything it names, reporting every problem at once; runs nothing

```text
splinter validate [OPTIONS] <RECIPE>
```

Arguments:

- `<RECIPE>` (required): The recipe, relative to the task repository (the working directory)

### splinter plan

Show the graph of stages a factory recipe runs and the work each does

```text
splinter plan [OPTIONS] <RECIPE>
```

Arguments:

- `<RECIPE>` (required): The recipe, relative to the task repository (the working directory)

### splinter generate

Write a factory collection's task directories with the command its file names (the task repository's own compiler)

```text
splinter generate [OPTIONS] <COLLECTION>
```

Arguments:

- `<COLLECTION>` (required): The collection file, relative to the task repository.

### splinter run

Run a factory recipe into a run directory, each stage reused when its product is already committed

```text
splinter run [OPTIONS] --out <DIR> <RECIPE>
```

Arguments:

- `<RECIPE>` (required): The recipe, relative to the task repository (the working directory)

Options:

- `--out <DIR>` (required): The run directory; an existing run of the same recipe is continued.

### splinter resume

Continue a factory run from what it recorded

```text
splinter resume [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

### splinter cancel

Ask a running factory run to stop at its next stage or batch

```text
splinter cancel [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

### splinter inspect

Show a factory run's products and reports, or open its trials

```text
splinter inspect [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

Options:

- `--stage <STAGE>`: Only the stages whose key starts with this (`rollouts`, `datasets/sft`)
- `--viewer <VIEWER>`: Open the stage's trials in a viewer instead (`harbor`)

### splinter view

Open a factory run in the browser: every stage's product, item and file, and decisions written to a review file (--reviews)

```text
splinter view [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

Options:

- `--reviews <FILE>`: The file decisions are appended to, one JSON line each (a repository's review file); without it the view only reads.
- `--port <PORT>`: The loopback port to serve on; any free one when 0. Default: `0`.

### splinter fork

Start a factory run on a changed recipe, reusing the products of an earlier run that the change leaves unchanged

```text
splinter fork [OPTIONS] --recipe <RECIPE> --out <DIR> <RUN>
```

Arguments:

- `<RUN>` (required): The run whose products are reused.

Options:

- `--recipe <RECIPE>` (required): The changed recipe, relative to the task repository.
- `--out <DIR>` (required): The new run directory.

### splinter report

Print a factory run's report

```text
splinter report [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

### splinter export

Write a factory run's products as a portable bundle

```text
splinter export [OPTIONS] --out <FILE> <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

Options:

- `--out <FILE>` (required): The bundle to write (`.tar.zst`)

### splinter index

File a factory run's trials in the experience database (an optional index across runs; the run's products stay its record)

```text
splinter index [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

### splinter play

Play one session of a factory run's interactive pass on this terminal; the run is then continued, so the session reaches the dataset and what is trained on it

```text
splinter play [OPTIONS] --stage <STAGE> --task <TASK> <RUN>
```

Arguments:

- `<RUN>` (required): The run directory.

Options:

- `--stage <STAGE>` (required): The interactive attempt stage (`interactive: true` in the recipe), by its id.
- `--task <TASK>` (required): The task to play, by its id in the collection.
- `--continue <SESSION>`: Continue the conversation of this earlier session of the stage (its number); the earlier session is left as it is.
- `--no-run`: Only play: do not run the recipe again afterwards (the next `resume` takes the session in)

### splinter doctor

Check that a run can start here: the state root is writable with room to spare, the policy's weights fit on a GPU, and every role's model resolves to weights on disk. Loads no model and uses no network. Exits 1 when a check fails

```text
splinter doctor [OPTIONS]
```

### splinter serve

Start the brain servers (one per usable node of a multi-node allocation, else one here), print where they are and keep them until interrupted

```text
splinter serve [OPTIONS]
```

### splinter baseline

Print the `baseline-v1` record of what a registered run runs on: the three commits, toolchain, binaries, model and tokenizer digests, seeds and hardware

```text
splinter baseline [OPTIONS]
```

Options:

- `--brain-bin <FILE>`: The brain binary the models are served with; its digest and version are recorded.
- `--model <DIR>...`: A model directory (or a GGUF file's directory) to record by file digest, with its tokenizer; repeatable. Hashing a large model takes minutes.
- `--exam-seed <N>`: The registered exam seed, recorded as given.
- `--training-seed <N>`: The registered training seed, recorded as given.
- `--require-clean`: Exit 1 and say why when the tree is dirty, a commit is unknown or the binary could not be digested.

### splinter source

Capture and inspect sources

```text
splinter source [OPTIONS] <COMMAND>
```

Subcommands:

- [`add`](#splinter-source-add) - Capture a file (a document), a directory (a repository) or cmd:COMMAND
- [`list`](#splinter-source-list) - List the stored sources
- [`show`](#splinter-source-show) - Show one stored source

#### splinter source add

Capture a file (a document), a directory (a repository) or cmd:COMMAND... (a command run), and store it. URLs are refused

```text
splinter source add [OPTIONS] <PATH|cmd:COMMAND...>...
```

Arguments:

- `<PATH|cmd:COMMAND...>...` (required): PATH, or cmd: followed by the command and its arguments.

Options:

- `--usage-policy <POLICY>`: The usage policy the content is under: redistributable, research_only, noncommercial, restricted_DUA or unknown. Without it the terms are unstated, which no release can be distributed under. The policy is part of the source's identity.

#### splinter source list

List the stored sources

```text
splinter source list [OPTIONS]
```

#### splinter source show

Show one stored source

```text
splinter source show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The source's id, or a unique prefix of it.

### splinter env

Compile a learning objective into validated, versioned training environments: classify how each skill can be verified, generate and admit environments, red-team their verifiers

```text
splinter env [OPTIONS] <COMMAND>
```

Subcommands:

- [`classify`](#splinter-env-classify) - Say how each skill of an objective could be verified (an objective check, a proxy, a calibrated judge or human evidence), with the reasons and the plugin that would build it
- [`plugins`](#splinter-env-plugins) - List the environment plugins, their difficulty knobs and what each verifies by
- [`compile`](#splinter-env-compile) - Compile an objective into environments
- [`list`](#splinter-env-list) - List the environments in a store with their status
- [`show`](#splinter-env-show) - Show one environment's admission report
- [`export`](#splinter-env-export) - Write an environment's stored instances of one split as Harbor task folders, one directory each under --out, for a run that needs them on disk (`make env/measure`)
- [`band`](#splinter-env-band) - Record a base policy's measured pass rate on an environment's dev instances and judge it

#### splinter env classify

Say how each skill of an objective could be verified (an objective check, a proxy, a calibrated judge or human evidence), with the reasons and the plugin that would build it. Builds nothing

```text
splinter env classify [OPTIONS] <--text <TEXT>|--file <FILE>>
```

Options:

- `--text <TEXT>`: The objective in plain language; each line is one skill.
- `--file <FILE>`: A file holding the objective; each line is one skill.
- `--source <PATH>...`: A source the objective rests on (a document, a repository); repeat for more. A skill about a document is only built when a source is given.
- `--constraint <TEXT>...`: What must hold of any environment built (a language, that real ratings or a calibration set exist); repeat for more.

#### splinter env plugins

List the environment plugins, their difficulty knobs and what each verifies by

```text
splinter env plugins [OPTIONS]
```

#### splinter env compile

Compile an objective into environments: a difficulty ladder per skill that has a plugin, each sampled, held to the admission gates and the verifier red team, and stored with its report. A beaten verifier is quarantined; an environment whose gates pass waits for its base pass rate (`env band`) before it may enter a campaign

```text
splinter env compile [OPTIONS] <--text <TEXT>|--file <FILE>>
```

Options:

- `--text <TEXT>`: The objective in plain language; each line is one skill.
- `--file <FILE>`: A file holding the objective; each line is one skill.
- `--source <PATH>...`: A source the objective rests on (a document, a repository); repeat for more. A skill about a document is only built when a source is given.
- `--constraint <TEXT>...`: What must hold of any environment built (a language, that real ratings or a calibration set exist); repeat for more.
- `--store <DIR>`: The store to write; default `environments` under the state root.
- `--environments <N>`: How many environments to build at most, across all skills. Default: `4`.
- `--instances <N>`: How many train and how many dev instances each environment is drawn for and tested on. Default: `4`.
- `--seed <N>`: The seed the samples are drawn from; drawn and printed when absent.
- `--repositories <DIR>`: The directory of repositories with their test suites (the `tasks/` directory: a `catalog.toml` and one directory each); enables the repository plugin on real projects.
- `--emulator <FILE>`: The emulator binary (`vemu`), run as a command and never modified; enables the firmware plugin.
- `--documents <DIR>`: A directory of Markdown or text documents to compile questions from (at least three: train and dev never share one); enables the document plugin. Name the same directory with --source so a document skill is built.

#### splinter env list

List the environments in a store with their status

```text
splinter env list [OPTIONS]
```

Options:

- `--store <DIR>`: The store; default `environments` under the state root.

#### splinter env show

Show one environment's admission report: every gate, the attacks that beat the verifier, the base pass rate

```text
splinter env show [OPTIONS] <ENV-ID>
```

Arguments:

- `<ENV-ID>` (required): The environment's id.

Options:

- `--store <DIR>`: The store; default `environments` under the state root.

#### splinter env export

Write an environment's stored instances of one split as Harbor task folders, one directory each under --out, for a run that needs them on disk (`make env/measure`). The folders are a working copy: remove them when the run is done, the store keeps the instances

```text
splinter env export [OPTIONS] --out <DIR> <ENV-ID>
```

Arguments:

- `<ENV-ID>` (required): The environment's id.

Options:

- `--split <train|dev>`: The split to write. Default: `dev`.
- `--out <DIR>` (required): The directory to write into (it must not hold files yet)
- `--store <DIR>`: The store; default `environments` under the state root.

#### splinter env band

Record a base policy's measured pass rate on an environment's dev instances and judge it: inside the band the environment is admitted to campaigns, outside it is rejected with the direction to turn the difficulty knobs

```text
splinter env band [OPTIONS] --model <NAME> --attempts <N> --passes <N> <ENV-ID>
```

Arguments:

- `<ENV-ID>` (required): The environment's id.

Options:

- `--model <NAME>` (required): Who was measured (a model and its serving settings)
- `--attempts <N>` (required): How many attempts were made.
- `--passes <N>` (required): How many of them were graded solved.
- `--store <DIR>`: The store; default `environments` under the state root.

### splinter objects

Knowledge objects (rules, thresholds, deadlines, exceptions, facts, procedure steps) a model proposes from documents, admitted by code

```text
splinter objects [OPTIONS] <COMMAND>
```

Subcommands:

- [`propose`](#splinter-objects-propose) - Ask a model for the typed objects each document of a directory states, one document at a time in date order

#### splinter objects propose

Ask a model for the typed objects each document of a directory states, one document at a time in date order. Admission is code: a quote must be verbatim in its document, every number in an object must be in its quote, names must be identifiers (and outcomes in --outcomes when given), and objects the documents' headers (authority, date, version) cannot separate stay flagged as conflicts. A reply with refused proposals is sent back with the reasons, up to --repairs times. Writes the pack as JSON

```text
splinter objects propose [OPTIONS] --out <FILE> <DIR>
```

Arguments:

- `<DIR>` (required): A directory of Markdown documents; a `---` header may give `authority`, `version` and `date`.

Options:

- `--out <FILE>` (required): Where the pack is written.
- `--pack-id <ID>`: The pack's name. Default: `pack`.
- `--outcomes <NAME,...>...`: The outcomes a rule may conclude, comma separated; open when absent.
- `--repairs <N>`: How many times a reply with refused proposals is sent back, per document. Default: `2`.
- `--single-pass`: Ask for every kind in one pass instead of one pass per family of kinds (limits and exceptions, formulas, periods and steps, facts and tables, behaviour). One pass per family finds far more; one pass is quicker.
- `--no-examples`: Leave out the worked example, from an unrelated domain, that each pass shows the model.
- `--chunk-words <WORDS>`: Put each document in runs of paragraphs of at most this many words, one run at a time.
- `--policy <REF>`: The model asked. Default: `policy:default`.

### splinter experiment

Check an experiment plan: which families each generation trains on, which earlier ones it is held to, which are the sealed final exam

```text
splinter experiment [OPTIONS] <COMMAND>
```

Subcommands:

- [`check`](#splinter-experiment-check) - Validate a plan against the catalog of families and show its schedule

#### splinter experiment check

Validate a plan against the catalog of families and show its schedule: families disjoint by domain, no sealed family scheduled for training or retention, retention only on families an earlier generation trained. A plan that is not valid is refused with every violation. A valid plan records its sealed final exam in the state root, so a later run that is not given the plan cannot read those families either

```text
splinter experiment check [OPTIONS] <PLAN>
```

Arguments:

- `<PLAN>` (required): The plan file.

Options:

- `--catalog <FILE>`: The catalog of families (default: catalog.toml beside the plan)

### splinter absorb

Learn from the sessions you held with an agent: take them in, extract what you taught, teach it from many records, train again from the base and release only if every claim the champion answered still is and a new one is

```text
splinter absorb [OPTIONS] <SESSIONS>...
```

Arguments:

- `<SESSIONS>...` (required): ATIF files, or directories searched for *.atif.json.

Options:

- `--policy <REF>`: The policy to update: a release is made under its alias. Default: `policy:default`.
- `--generator <REF>`: The model that proposes claims and writes their paraphrases and forms (default: the policy)
- `--teacher <REF>`: The model that answers with the claim in front of it, as the policy would (default: the policy)
- `--judge <REF>`: A model that decides the answers the claim's terms cannot, whether a later claim supersedes, reinforces or is apart from an earlier one, and whether the person's words assert a claim's statement; another model than the teacher, the generator and the policy.
- `--entail`: Whether the judge also refuses a claim whose cited words do not assert its statement (`--entail=false` turns it off); it applies when a judge is configured. Values: `true`, `false`.
- `--passes <N>`: How many times each session is read for claims, in different orders; only a claim every pass proposed is ruled on. Default: `2`.
- `--paraphrases <N>`: Differently worded questions written per claim; a third of them are kept out of training as the claim's own stopping and gate set. Default: `12`.
- `--rehearsal-share <SHARE>`: The share of the training draws that are the base's own answers to general tasks, in [0, 1); 0 turns it off. Default: `0.25`.
- `--rehearsal-seed <N>`: The seed the rehearsal's form tasks are built with [default: drawn and printed]
- `--epochs <N>`: The most passes over the night's records. Default: `6`.
- `--steps <N>`: The step budget, in place of --epochs.
- `--rank <N>`: LoRA rank of the adapter. Default: `8`.
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--continue-from-release`: Continue the current release on the claims not yet absorbed, with a replay of earlier releases' records, instead of training again from the base on every live claim. An ablation: the default makes supersession and forgetting exact.
- `--replay-fraction <F>`: With --continue-from-release, the fraction of each earlier release's records replayed. Default: `0.25`.
- `--subjects <FILE>...`: Files (JSON Lines of {"id", "terms"}) saying what the facts the sessions are known to be about (`extra.absorb_subject` of their trajectory) look like in a statement; the report then says whether the claims cover them.
- `--sealed-probes <FILE>...`: Files of sealed probes (JSON Lines of {"question", "reference"?, "name"?}): a training record containing a probe's question, or an 8-word run of a probe beyond its claim's statement, is refused. The gate never reads them.
- `--stance-pairs <N>`: The preference pairs the stance pass trains on after the supervised candidate: facts held against a confident falsehood, a statement accepted without support invented for it, questions nobody could answer declined. Derived afresh each night from the live claims, split evenly over the three kinds and bounded to half the night's records. Default: `96`.
- `--stance-beta <BETA>`: The DPO temperature of the stance pass [default: brain's]
- `--stance-tries <N>`: The replies drawn before a stance pair is given up for want of a reply to reject. Default: `8`.
- `--no-stance`: Train no stance pass: the supervised candidate is the night's.
- `--consolidate-after-records <N>`: Consolidate the generation into the weights once its claims hold more trained records than this: retraining the adapter from the base each night has become impractical, so a full checkpoint becomes the next generation's base and its claim set starts empty. Default: `2000`.
- `--no-consolidate`: Never consolidate: the generation's claims accumulate.
- `--consolidate-mode <MODE>`: How a due consolidation takes the generation in: `full` trains every weight, `merge` folds the adapter in. Default: `full`.
- `--retention-replay-share <SHARE>`: The share of the training draws that are records of the claims the generation's base already holds, in [0, 1) Default: `0.25`.
- `--dry-run`: Stop after the claims are extracted and ruled on; train nothing.
- `--no-release`: Stop at the trained candidate: do not run the gate.

### splinter consolidate

Take the generation the policy is in into the weights: a full checkpoint, trained from the generation's base on its claims (or the adapter folded in), gated against the release in use, becomes the base of the next generation, whose claim set starts empty. The claims it holds must keep being answered at every release after

```text
splinter consolidate [OPTIONS]
```

Options:

- `--policy <REF>`: The policy to consolidate: the checkpoint is released under its alias. Default: `policy:default`.
- `--mode <MODE>`: `full` trains every weight of the generation's base on the claims' records, beside a sample of the records of the claims the base already holds and the base's own answers; `merge` folds the adapter into the base's weights and trains nothing (the only mode that fits a model whose weights and gradients do not fit the cards) Default: `full`.
- `--from <RELEASE>`: The release to consolidate, by id or unique prefix (default: the one the policy points at)
- `--rehearsal-share <SHARE>`: The share of the training draws that are the base's own answers to general tasks, in [0, 1); 0 turns it off. Full mode only. Default: `0.25`.
- `--rehearsal-seed <N>`: The seed the rehearsal's form tasks are built with [default: drawn and printed]
- `--retention-replay-share <SHARE>`: The share of the training draws that are records of the claims the base already holds, in [0, 1). Full mode only. Default: `0.25`.
- `--epochs <N>`: The most passes over the records. Full mode only. Default: `6`.
- `--steps <N>`: The step budget, in place of --epochs. Full mode only.
- `--lr <LR>`: The peak learning rate [default: 0.00001]
- `--sealed-probes <FILE>...`: Files of sealed probes (JSON Lines of {"question", "reference"?, "name"?}): a training record containing one is refused.
- `--no-release`: Gate the checkpoint but do not release it: it is not kept.

### splinter improve

Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs: more samples, another episode protocol, a trained candidate, or no change

```text
splinter improve [OPTIONS] <EXPERIMENT>...
```

Arguments:

- `<EXPERIMENT>...` (required): Improvement experiments, each a TOML file: the capability (a task set), the incumbent, the interventions to consider (more samples, another episode protocol, a trained candidate, no change) and what a success is worth. Each intervention is measured against the incumbent on the same tasks and the one with the greatest net value is chosen, or none; nothing is applied.

Options:

- `--compare-fixed`: Beside the choices, what applying one intervention to every experiment would have come to.

### splinter session

Take in recorded agent sessions, secrets removed

```text
splinter session [OPTIONS] <COMMAND>
```

Subcommands:

- [`add`](#splinter-session-add) - Take recorded agent sessions in as sources
- [`list`](#splinter-session-list) - List the stored sessions

#### splinter session add

Take recorded agent sessions in as sources: ATIF files, or directories searched for *.atif.json. Each is validated, refused with the reason when it cannot be rendered as training conversations, stripped of secrets (keys, tokens, passwords, private keys) and stored once by content; the same file again changes nothing. Exits 1 when any file was refused

```text
splinter session add [OPTIONS] <PATH>...
```

Arguments:

- `<PATH>...` (required): ATIF files, or directories of them.

Options:

- `--max-bytes <BYTES>`: The largest session file taken, in bytes. Default: `67108864`.

#### splinter session list

List the stored sessions

```text
splinter session list [OPTIONS]
```

### splinter claims

What a person taught in sessions: claims proposed by a model, ruled on by code, and the ledger of the rulings

```text
splinter claims [OPTIONS] <COMMAND>
```

Subcommands:

- [`extract`](#splinter-claims-extract) - Have a model propose the claims each stored session teaches
- [`gate`](#splinter-claims-gate) - Rule on a claim set
- [`forget`](#splinter-claims-forget) - Forget a live claim
- [`list`](#splinter-claims-list) - List the stored claim sets
- [`show`](#splinter-claims-show) - Show a claim set
- [`ledger`](#splinter-claims-ledger) - Show the ledger

#### splinter claims extract

Have a model propose the claims each stored session teaches - a correction, a fact, a procedure - with the person's exact words that support each, into a claim set. Exits 1 when a session's reply stayed unusable

```text
splinter claims extract [OPTIONS] <SESSION-ID>...
```

Arguments:

- `<SESSION-ID>...` (required): The sessions, by id or unique prefix, in the order they happened.

Options:

- `--generator <REF>`: The model that reads the sessions. Default: `policy:default`.
- `--passes <N>`: How many times each session is read, in different orders; only a claim every pass proposed is kept, the rest are listed as unconfirmed. Default: `2`.

#### splinter claims gate

Rule on a claim set: quotes verbatim in the person's words, the statement's numbers, names and quoted terms in them, the agent's sentences never evidence. Claims that name the same thing are paired: a later claim that contradicts the earlier supersedes it, one that says the same again reinforces it (recorded, never refused). Every ruling is kept, each refusal with its reason

```text
splinter claims gate [OPTIONS] <CLAIMSET-ID>
```

Arguments:

- `<CLAIMSET-ID>` (required): The claim set, by id or unique prefix.

Options:

- `--judge <REF>`: The model that decides whether a later claim supersedes, reinforces or is apart from an earlier one about the same thing; without one the statements decide.
- `--entail`: Whether the judge also refuses a claim whose cited words do not assert its statement (`--entail=false` turns it off) Values: `true`, `false`.
- `--subjects <FILE>...`: Files (JSON Lines of {"id", "terms"}) saying what the facts the sessions are known to be about (`extra.absorb_subject` of their trajectory) look like in a statement; the report then says whether the claims cover them.

#### splinter claims forget

Forget a live claim: it leaves the live set, so the next `absorb` trains again from the base without it, even when nothing else is new. The ruling and the claim's evidence stay in the ledger

```text
splinter claims forget [OPTIONS] <CLAIM-ID>
```

Arguments:

- `<CLAIM-ID>` (required): The live claim, by id or unique prefix (`claims ledger` lists them)

#### splinter claims list

List the stored claim sets

```text
splinter claims list [OPTIONS]
```

#### splinter claims show

Show a claim set: its proposals per session

```text
splinter claims show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The claim set, by id or unique prefix.

#### splinter claims ledger

Show the ledger: live claims with how often each was said again, superseded ones with by which, and refused proposals with their reasons

```text
splinter claims ledger [OPTIONS]
```

### splinter tasks

Generate and inspect task sets

```text
splinter tasks [OPTIONS] <COMMAND>
```

Subcommands:

- [`history`](#splinter-tasks-history) - Read tasks from a repository's history
- [`variants`](#splinter-tasks-variants) - Write differently worded questions about each task of a task set, into a task set of their own
- [`combine`](#splinter-tasks-combine) - Make harder tasks of the admitted single-fault tasks of a task set
- [`drills`](#splinter-tasks-drills) - Make comprehension drills from a repository's clean snapshot
- [`pick`](#splinter-tasks-pick) - Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking)
- [`check-requests`](#splinter-tasks-check-requests) - Drop the tasks whose request does not state the contract their hidden check enforces (a returned value or a raised error that no failing test's title speaks of, so the request cannot lead to the fix)
- [`harbor`](#splinter-tasks-harbor) - Harbor tasks
- [`list`](#splinter-tasks-list) - List the stored task sets
- [`show`](#splinter-tasks-show) - Show a task set's tasks, or one task

#### splinter tasks history

Read tasks from a repository's history: a commit that changes source and tests together is a task when its parent tree fails the commit's own tests and the commit's tree passes them. Formatting-only, generated, vendored, merge and test-less commits are excluded and counted

```text
splinter tasks history [OPTIONS] <REPOSITORY>
```

Arguments:

- `<REPOSITORY>` (required): The git repository.

Options:

- `--range <RANGE>`: The revision range, as `git rev-list` reads it (`A..B`); a bare revision reads all of its history. Default: `HEAD`.
- `--test-command <PROGRAM ARG...>...`: The command that runs the project's whole test suite in the root of its tree (everything after this flag); detected by rule when not given.
- `--max-commits <N>`: The most commits considered, newest first.
- `--store`: Store the admitted tasks as a task set (no rollout is made)

#### splinter tasks variants

Write differently worded questions about each task of a task set, into a task set of their own: the same facts, to be measured and never trained on

```text
splinter tasks variants [OPTIONS] <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--generator <REF>`: The model that writes the variants (the one that wrote the tasks) Default: `policy:default`.
- `--per-task <N>`: The most variants one task is given. Default: `3`.

#### splinter tasks combine

Make harder tasks of the admitted single-fault tasks of a task set: each holds several independent faults in different files of one snapshot, is graded by every member's hidden tests, and is admitted only when its reference change makes it pass (twice) and no single member's does. A task with a fault in a file the exam reserved is an exam task. The choice is a function of the seed and the tasks alone. They go into task sets of their own: all of them, the exam's, and the rest

```text
splinter tasks combine [OPTIONS] --per-family <N> <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set of single-fault tasks, by id or unique prefix.

Options:

- `--faults <K>`: Faults in each task, in as many different files. Default: `2`.
- `--per-family <N>` (required): The most combined tasks any one file takes part in.
- `--max-tasks <N>`: The most combined tasks made in all. Default: `64`.
- `--seed <N>`: The seed that decides which tasks are combined (drawn and printed when not given)

#### splinter tasks drills

Make comprehension drills from a repository's clean snapshot: small closed-book tasks that show a program calling one function of the repository on inputs its own tests use, and ask what it prints. The reference is what the interpreter prints against the snapshot (never a task's broken tree, never a hidden test); a candidate is admitted only when it runs, prints something informative and prints the same twice. A drill is tagged with its file and function, and one of a file the practice exam reserved is an exam task. They go into task sets of their own: all of them, the exam's, and the rest

```text
splinter tasks drills [OPTIONS] <REPO>
```

Arguments:

- `<REPO>` (required): The git repository: a work tree or a bare repository.

Options:

- `--rev <REV>`: The revision whose snapshot is drilled: anything git resolves to a commit. The tracked files of that commit are the snapshot; uncommitted work is not in it. Default: `HEAD`.
- `--allow-dirty`: Snapshot the committed files of a revision that is checked out with uncommitted changes, instead of refusing it.
- `--per-function <N>`: The most drills of any one function. Default: `3`.
- `--seed <N>`: The seed that chooses among the inputs the tests give (drawn and printed when not given)
- `--file <PATH>...`: A source file (`.py` or `.lua`), relative to the repository, to drill; repeat for more. Every source file that is not a test without any.
- `--for-tasks <TASKSET-ID>`: Drill only the functions that contain a line the reference changes of this task set's workspace tasks change (the tasks a policy fails or solves only sometimes), so that it can be trained on the drills before it attempts them again. The revision must be the one the tasks were made from.

#### splinter tasks pick

Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking): a development holdout, a sample of one task per family, or one set out of several. Naming the sets in another order, or a task in two of them, changes nothing. A task that declares no family is a family of its own

```text
splinter tasks pick [OPTIONS] --seed <N> <TASKSET-ID>...
```

Arguments:

- `<TASKSET-ID>...` (required): The task sets, by id or unique prefix.

Options:

- `--family-share <F>`: Pick this fraction (0 to 1) of the families, every task of each.
- `--per-family <N>`: Pick at most this many tasks of every family.
- `--seed <N>` (required): The seed that decides what is picked.

#### splinter tasks check-requests

Drop the tasks whose request does not state the contract their hidden check enforces (a returned value or a raised error that no failing test's title speaks of, so the request cannot lead to the fix). The kept tasks, unchanged, are written as a task set of their own; the dropped are counted by what they leave unstated. The same rule refuses such tasks when they are generated

```text
splinter tasks check-requests [OPTIONS] <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

#### splinter tasks harbor

Harbor tasks: discover, import and export

```text
splinter tasks harbor [OPTIONS] <COMMAND>
```

Subcommands:

- [`discover`](#splinter-tasks-harbor-discover) - List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in
- [`import`](#splinter-tasks-harbor-import) - Import Harbor tasks (each a task directory) as workspace tasks in one task set
- [`export`](#splinter-tasks-harbor-export) - Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly
- [`episodes`](#splinter-tasks-harbor-episodes) - Run a directory of practice task folders (or one folder) as episodes

##### splinter tasks harbor discover

List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in. Writes nothing

```text
splinter tasks harbor discover [OPTIONS] <DIR>
```

Arguments:

- `<DIR>` (required): The directory (a checkout of a task repository)

##### splinter tasks harbor import

Import Harbor tasks (each a task directory) as workspace tasks in one task set. A task the host cannot honour (a Dockerfile with RUN, a prebuilt image, a Windows or multi-step task) is refused with the reason; the rest are held to the bar of every workspace task

```text
splinter tasks harbor import [OPTIONS] <DIR>...
```

Arguments:

- `<DIR>...` (required): The task directories.

Options:

- `--oracle <ORACLE>`: What to do about a task's solution/solve.sh: `skip` does not run it, `try` runs it and imports without a reference when it does not run on the host, `require` refuses the task unless it runs and its change passes the task's check. Default: `try`. Values: `skip`, `try`, `require`.

##### splinter tasks harbor export

Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly

```text
splinter tasks harbor export [OPTIONS] --out <DIR> <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--out <DIR>` (required): The directory to write into (it must not hold the tasks yet)
- `--image <IMAGE>`: The base image of each Dockerfile. Default: `ubuntu:24.04`.

##### splinter tasks harbor episodes

Run a directory of practice task folders (or one folder) as episodes: the policy attempts each task --attempts times, each in a fresh copy of its starting files through sven's agent, asked for the tokens and log-probabilities it samples; afterwards the task's own tests grade what it left, outside its reach. Every attempt is stored as an experience with a record of its outcome (passed, failed, exposed, verifier error or timeout, infrastructure), reward, usage and provenance. Exam material (`[metadata].bench_sealed`, a held-out tier, the eval set of the collection's split.toml) is refused before anything runs. A rerun skips attempts already recorded and redoes the ones the infrastructure stopped

```text
splinter tasks harbor episodes [OPTIONS] <DIR>
```

Arguments:

- `<DIR>` (required): The directory of task folders, or one task folder.

Options:

- `--policy <REF>`: The policy. Default: `policy:default`.
- `--attempts <N>`: Attempts per task. Default: `1`.
- `--seed <SEED>`: The seed every attempt's draws derive from (drawn and printed when not given; a rerun with the same seed finds its attempts)
- `--jobs <N>`: Attempts run at once. Default: `1`.
- `--deadline <DUR>`: How long one attempt's agent may run (`90s`, `10m`, ...); the solve stage's deadline when not given.
- `--max-tool-calls <N>`: Tool calls one attempt may request.
- `--max-output-tokens <N>`: Output tokens one attempt may generate.
- `--out <FILE>`: Also write every attempt's record to this file, one JSON object per line.

#### splinter tasks list

List the stored task sets

```text
splinter tasks list [OPTIONS]
```

#### splinter tasks show

Show a task set's tasks, or one task

```text
splinter tasks show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): A task set's or a task's id, or a unique prefix of it.

### splinter verify

Check a factory run: every product against its blobs and its view

```text
splinter verify [OPTIONS] <RUN>
```

Arguments:

- `<RUN>` (required): The factory run directory.

### splinter critique

Critique the failed experiences of a set and retry them with it

```text
splinter critique [OPTIONS] <EXPERIENCE-SET>
```

Arguments:

- `<EXPERIENCE-SET>` (required): The experience set, by id or unique prefix.

Options:

- `--critic <REF>`: The model that critiques. Default: `policy:default`.
- `--retry <N>`: Retries per failed experience. Default: `1`.

### splinter judge

Measure a judge against labelled experiences

```text
splinter judge [OPTIONS] <COMMAND>
```

Subcommands:

- [`calibrate`](#splinter-judge-calibrate) - Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by
- [`measure`](#splinter-judge-measure) - Measure a judge on the controls a task set's own references give

#### splinter judge calibrate

Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by

```text
splinter judge calibrate [OPTIONS] --judge <REF> <LABELLED-FILE>
```

Arguments:

- `<LABELLED-FILE>` (required): JSON Lines: `{"experience": "<id>", "label": "pass"|"fail"}`.

Options:

- `--judge <REF>` (required): The judge.

#### splinter judge measure

Measure a judge on the controls a task set's own references give - each task's reference as the right answer, another family's as the wrong one - and show its precision, whether it would be trusted and the controls it got wrong. A calibration that rests on enough controls is kept, and a run reuses it

```text
splinter judge measure [OPTIONS] --judge <REF> <TASK-SET>
```

Arguments:

- `<TASK-SET>` (required): The task set, by id or unique prefix.

Options:

- `--judge <REF>` (required): The judge.
- `--fit`: Measure it as a judge of fit - is a passage a natural reply to its message - and not of whether an answer gives the reference.

### splinter experiences

Inspect experience sets and experiences

```text
splinter experiences [OPTIONS] <COMMAND>
```

Subcommands:

- [`list`](#splinter-experiences-list) - List the stored experience sets
- [`show`](#splinter-experiences-show) - Show an experience set, or one experience with its annotations
- [`replay`](#splinter-experiences-replay) - Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed
- [`ingest-harbor`](#splinter-experiences-ingest-harbor) - Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict
- [`ingest-bench`](#splinter-experiences-ingest-bench) - Admit the trials of exports written by `bench export` as experiences
- [`diagnose`](#splinter-experiences-diagnose) - Why each failed trial of bench exports failed
- [`capabilities`](#splinter-experiences-capabilities) - A policy's competence per capability of the examiner's taxonomy, from bench exports or practice runs
- [`import`](#splinter-experiences-import) - Import experience sets, with their experiences' annotations and trial register, from another state root
- [`ingest-hf`](#splinter-experiences-ingest-hf) - Ingest a bundle `publish` wrote, at a pinned revision of its Hugging Face dataset repository
- [`compare`](#splinter-experiences-compare) - Compare two experience sets of the same tasks
- [`compare-arms`](#splinter-experiences-compare-arms) - Compare graded trials of several arms (Harbor jobs or trials files) with a reference arm

#### splinter experiences list

List the stored experience sets

```text
splinter experiences list [OPTIONS]
```

#### splinter experiences show

Show an experience set, or one experience with its annotations

```text
splinter experiences show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): An experience set's or an experience's id, or a unique prefix.

Options:

- `--graph`: Show the relations around it instead: retries, critiques, preferences.

#### splinter experiences replay

Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed

```text
splinter experiences replay [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): An experience set's or an experience's id, or a unique prefix.

#### splinter experiences ingest-harbor

Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict: a reward of exactly 1 passes, any other fails, no reward or an exception is no verdict

```text
splinter experiences ingest-harbor [OPTIONS] <TASKSET-ID> <TRIAL>...
```

Arguments:

- `<TASKSET-ID>` (required): The task set the trials' tasks were imported into.
- `<TRIAL>...` (required): The trial directories.

#### splinter experiences ingest-bench

Admit the trials of exports written by `bench export` as experiences: every file is checked against the digest in the trial's manifest and the manifest against the export's index, a tampered or partial trial is refused with the reason, the benchmark's grade is the verdict and the ATIF trajectory the record. Ingesting an export again changes nothing; a trial name with other content is refused. The examiner's score record must agree with the grade; a trial it does not count has no verdict, never a fail. A confidential trial, and an export of sealed material (`sealed-export.json` beside it), are refused. Trials come out in one experience set per split: `train` trials can be built into datasets, `eval` trials are examination material no dataset accepts

```text
splinter experiences ingest-bench [OPTIONS] <EXPORT>...
```

Arguments:

- `<EXPORT>...` (required): The export directories (each holds index.jsonl)

Options:

- `--assume-set <SPLIT>`: The split of trials whose export names none (exports made before the benchmark recorded splits). Without it such a trial is refused: a split is never guessed. The register records that the operator stated it, and a trial that names its split must agree. Values: `train`, `eval`.

#### splinter experiences diagnose

Why each failed trial of bench exports failed: one failure class per trial, from its evidence, with the step and excerpt that decided it

```text
splinter experiences diagnose [OPTIONS] <DIR>...
```

Arguments:

- `<DIR>...` (required): The export directories (each holds index.jsonl; skipped.jsonl when the export skipped trials), or practice run directories (each holds report.json and export/)

#### splinter experiences capabilities

A policy's competence per capability of the examiner's taxonomy, from bench exports or practice runs: pass rates with 95% intervals, failure classes, frontier bands and the curriculum's weight, kept in the store with where every number came from

```text
splinter experiences capabilities [OPTIONS] <DIR>...
```

Arguments:

- `<DIR>...` (required): The export directories, or practice run directories (each holds report.json and export/; its material's practice.json labels its tasks)

Options:

- `--labels <PRACTICE_DIR>...`: A practice directory whose practice.json labels tasks the runs' material does not (staged before the examiner named capabilities); a task is labelled from it only when its source digest matches. Repeatable.
- `--policy <NAME>`: The policy the trials of an export without a practice run's report count under.

#### splinter experiences import

Import experience sets, with their experiences' annotations and trial register, from another state root

```text
splinter experiences import [OPTIONS] --from <DIR> <SET>...
```

Arguments:

- `<SET>...` (required): The sets, by id or unique prefix in that root.

Options:

- `--from <DIR>` (required): The state root to read the sets from (only read; import from a copy of a root another process writes)

#### splinter experiences ingest-hf

Ingest a bundle `publish` wrote, at a pinned revision of its Hugging Face dataset repository: every file checked against the manifest's digest, everything stored under its own content address

```text
splinter experiences ingest-hf [OPTIONS] <REPO@REVISION>
```

Arguments:

- `<REPO@REVISION>` (required): The bundle, `OWNER/NAME@REVISION`: a commit, or a branch or tag, which is resolved to the commit it names and recorded as that commit.

Options:

- `--hub-endpoint <URL>`: The Hub to talk to (a local emulator's `http://127.0.0.1:PORT` in a test) Default: `https://huggingface.co`.
- `--token-file <FILE>`: A file holding the access token (name the `hf` CLI's own, `~/.cache/huggingface/token`, to use it); without it, calls are anonymous.

#### splinter experiences compare

Compare two experience sets of the same tasks: the candidate's against the baseline's, paired by task, a family counted once. Exposed and harness-failed attempts are left out and counted

```text
splinter experiences compare [OPTIONS] <CANDIDATE-SET> <BASELINE-SET>
```

Arguments:

- `<CANDIDATE-SET>` (required): The candidate's experience set, by id or unique prefix.
- `<BASELINE-SET>` (required): The baseline's experience set, by id or unique prefix.

Options:

- `--interval-seed <N>`: The seed the intervals are resampled with (drawn and printed when not given; the same sets and seed give the same intervals)
- `--stratum-checkers <N>`: Tests a task's fail-to-pass commands run: more is above.
- `--stratum-footprint <N>`: Files a task's reference patch changes: more is above.
- `--stratum-instruction-chars <N>`: Characters of a task's instruction: more is above.

#### splinter experiences compare-arms

Compare graded trials of several arms (Harbor jobs or trials files) with a reference arm: pass@1 and unbiased pass@k, the paired difference with task and family bootstrap intervals, exact one- and two-sided sign tests, Holm-adjusted; timeouts and infrastructure errors counted apart

```text
splinter experiences compare-arms [OPTIONS] --reference <NAME>
```

Options:

- `--arm <NAME=DIR>...`: An arm and the Harbor job directory of its trials (one subdirectory with a result.json per trial), as NAME=DIR; repeat for each arm.
- `--trials <FILE>...`: A trials file: one JSON object per line, {"task", "arm", "family"?, "attempt"?, "pass" or "reward", "status"?} with status graded (default), timeout or infra_error; repeat for several.
- `--reference <NAME>` (required): The arm every other arm is compared with.
- `--families <FILE>`: A JSON object naming each task's family ({"task": "family"}): the difference is then also resampled by whole families.
- `--k <K>...`: A k to report pass@k at (unbiased, over the tasks with at least k graded attempts); repeat for several. pass@1 is always reported.
- `--resamples <N>`: Resamples per bootstrap interval. Default: `10000`.
- `--level <SHARE>`: The share of resampled means an interval holds. Default: `0.95`.
- `--interval-seed <N>`: The seed the intervals are resampled with (drawn and printed when not given; the same trials and seed give the same intervals)

### splinter dataset

Build datasets from experience sets, and export them

```text
splinter dataset [OPTIONS] <COMMAND>
```

Subcommands:

- [`conversations`](#splinter-dataset-conversations) - Build a policy-gradient dataset from graded conversations with a counterpart (a client, a student)
- [`decision-fixtures`](#splinter-dataset-decision-fixtures) - Write labelled decision fixtures (one example per labelled choice question) from the run records of an ablation benchmark output directory, split by task family and seed so that no task is in two splits
- [`check`](#splinter-dataset-check) - Have the trainer read a stored dataset before any device is claimed
- [`verify-speech`](#splinter-dataset-verify-speech) - Hear every recording of an audio dataset (a Hugging Face audio folder's metadata.jsonl) through a recogniser and check it says its transcript
- [`export`](#splinter-dataset-export) - Copy a dataset and its manifest into a directory

#### splinter dataset conversations

Build a policy-gradient dataset from graded conversations with a counterpart (a client, a student): each line of the input names the case, the model that played the learner, the reward the case's grade earned and the conversation record. Only conversations the admission rules accept are used; the learner's turns are the actions

```text
splinter dataset conversations [OPTIONS] --system-prompt <TEXT> <GRADED.JSONL>
```

Arguments:

- `<GRADED.JSONL>` (required): The input: one JSON object per line (`case`, `policy`, `trial`, `reward`, `record`)

Options:

- `--system-prompt <TEXT>` (required): The role prompt the learner played under, which opens every record.
- `--advantage <KIND>`: How brain is to read an attempt's advantage from the rewards of its case (`--advantage` of `dataset build`)

#### splinter dataset decision-fixtures

Write labelled decision fixtures (one example per labelled choice question) from the run records of an ablation benchmark output directory, split by task family and seed so that no task is in two splits. The output is a JSON-lines file, not a stored dataset

```text
splinter dataset decision-fixtures [OPTIONS] --out <FILE> <RUNS-DIR>
```

Arguments:

- `<RUNS-DIR>` (required): The benchmark output directory (it holds `runs/<task>/<variant>/`)

Options:

- `--out <FILE>` (required): The JSON-lines file to write.
- `--split-seed <N>`: Seed of the split; the same seed gives the same split. Drawn and printed when not given.
- `--validation-fraction <F>`: The share of each family's seeds held for validation. Default: `0.2`.
- `--test-fraction <F>`: The share of each family's seeds held for test. Default: `0.2`.
- `--holdout-suite <FAMILY>...`: A whole task family that is test only (repeatable)

#### splinter dataset check

Have the trainer read a stored dataset before any device is claimed

```text
splinter dataset check [OPTIONS] <DATASET-ID>
```

Arguments:

- `<DATASET-ID>` (required): The dataset, by id or unique prefix.

Options:

- `--from <REF>`: The model the training starts from, as `train --from` names it: its base's tokenizer and chat template read the records. Without it, the policy.
- `--max-block <TOKENS>`: The training row to check against, in tokens: longer examples are counted. Without it, the widest row brain takes.

#### splinter dataset verify-speech

Hear every recording of an audio dataset (a Hugging Face audio folder's metadata.jsonl) through a recogniser and check it says its transcript: a different text, a take read twice or cut short fails, a close but imperfect hearing is for review. The verdict and its evidence are written into each row; exits 1 unless every recording passes

```text
splinter dataset verify-speech [OPTIONS] <METADATA>
```

Arguments:

- `<METADATA>` (required): The audio folder's metadata.jsonl: one row per recording, its `file_name` relative to this file's directory. Any format the brain server reads (WAV at any rate and channel count, Ogg/Opus, WebM/Opus)

Options:

- `--text-column <COLUMN>`: The column holding each recording's transcript. Default: `text`.
- `--recognizer <MODEL>`: The recognition model, as the brain server lists it. Qwen3-ASR hears Swedish and English alike. Default: `brain/qwen3asr`.
- `--pass-at <F>`: At or above this similarity (letters and digits compared, spacing and punctuation ignored) a recording passes. Default: `0.9`.
- `--fail-below <F>`: Below this similarity a recording fails; between the two it is for review. Default: `0.6`.
- `--min-length-ratio <F>`: The shortest hearing, over the transcript's length, that passes: a take cut short is below it. Default: `0.8`.
- `--max-length-ratio <F>`: The longest hearing, over the transcript's length, that passes: a take read twice is above it. Default: `1.25`.
- `--rehear`: Hear every recording again, even one unchanged since this recogniser last heard it.

#### splinter dataset export

Copy a dataset and its manifest into a directory

```text
splinter dataset export [OPTIONS] --out <DIR> <DATASET-ID>
```

Arguments:

- `<DATASET-ID>` (required): The dataset, by id or unique prefix.

Options:

- `--out <DIR>` (required): The directory to copy into.

### splinter goal

Goal-directed learning: what may enter a selection of material

```text
splinter goal [OPTIONS] <COMMAND>
```

Subcommands:

- [`admit`](#splinter-goal-admit) - Admit documents into a selection of learning material

#### splinter goal admit

Admit documents into a selection of learning material. Public or simulated documents (`--open`) enter as they are. Recordings of real people (`--live`) enter only with the person's consent (`--consent`) and after the scrub of the current detectors (`--scrub` runs it): an unscrubbed, unconsented or still-identifying document is refused and reported, and the command then exits with an error. Admitted documents are written to `--out` as Markdown with their header

```text
splinter goal admit [OPTIONS] --out <DIR>
```

Options:

- `--open <FILE>...`: Documents with no person in them (a file each)
- `--live <FILE>...`: Recordings of real people (a file each)
- `--consent`: The people agreed to the use of their recordings as training material.
- `--scrub`: Scrub the live documents (secrets, addresses, accounts, contact data) first.
- `--out <DIR>` (required): The directory the admitted documents are written to.

### splinter release

The releases and the aliases pointing at them (a factory run's `promotion` stage makes a release)

```text
splinter release [OPTIONS] <COMMAND>
```

Subcommands:

- [`list`](#splinter-release-list) - List the releases and the aliases pointing at them

#### splinter release list

List the releases and the aliases pointing at them

```text
splinter release list [OPTIONS]
```

### splinter rollback

Point an alias back at the release its current one was trained from

```text
splinter rollback [OPTIONS] <ALIAS>
```

Arguments:

- `<ALIAS>` (required): The alias, e.g. default.

### splinter eval

Evaluate a factory run's candidates under another plan into a new run (`RUN --plan PLAN --out DIR`); grade a model closed-book on a file of tasks (`--suite FILE`); or show and freeze the anchor suite

```text
splinter eval [OPTIONS] [RUN]
       splinter eval <COMMAND>
```

Arguments:

- `<RUN>`: A factory run directory, whose candidates `--plan` evaluates; none shows the anchor suite, or grades a model with --suite.

Options:

- `--freeze <FILE>...`: Make these files' tasks (repeat the flag for several) the anchor suite's next version. JSON Lines: {"instruction", "reference", "kind"?}.
- `--suite <FILE>`: Grade --policy closed-book on this file's tasks (the anchor format), keeping every item with its verdict; decides nothing.
- `--policy <REF>`: With --suite: the model graded. Default: `policy:default`.
- `--plan <PLAN>`: The evaluation plan to evaluate the run's candidates under, reusing them (a fork of the run into --out with the recipe's plan replaced)
- `--out <DIR>`: With --plan: the new run directory.

Subcommands:

- [`matrix`](#splinter-eval-matrix) - The continual-learning results matrix of a lineage

#### splinter eval matrix

The continual-learning results matrix of a lineage: what each gate measured of each release on each earlier release's suite, with the metrics read from it. A cell no gate measured is shown absent

```text
splinter eval matrix [OPTIONS]
```

Options:

- `--alias <NAME>`: The alias whose releases are the generations. Default: `default`.

### splinter exam

Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them; or, with --exam-set, to a frozen exam of reserved families, with every verdict kept

```text
splinter exam [OPTIONS] <CANDIDATE>
```

Arguments:

- `<CANDIDATE>` (required): The candidate to examine: its id, or a unique prefix of it.

Options:

- `--judge <REF>`: The model that judges (default: the judge role's model)
- `--prompt <GOAL>`: Also ask the base under this goal, as a prompt-only baseline the candidate is compared with (`learn` uses its goal)
- `--retrieve <SOURCE-ID>...`: Show the model the passages of these stored sources that bear on the question, found by meaning and by exact words, instead of all of one.
- `--passages <N>`: How many passages `--retrieve` shows. Default: `6`.
- `--reranker <REF>`: A model that reads the candidates search found beside the question and moves the ones that bear on it forward.
- `--exam-set <EXAM>`: Put the candidate to this frozen exam (its id, or its manifest file) instead: four arms - the base, the base told who the candidate is to be, the candidate, and the candidate under that prompt - compared per task with the family as the unit of evidence, every verdict kept.
- `--resamples <N>`: With --exam-set: answers per task per arm, the first greedy and the rest sampled [default: 3]
- `--pilot-families <N>`: With --exam-set: put only this many of the exam's families to the candidate, as a pilot that estimates the discordance and clustering.
- `--deployed-only`: With --exam-set: ask only the candidate under the prompt it is deployed with. Nothing is compared in the report; the arms of another report of the same exam (the base, the prompted base, the same tasks and families, greedy answers) are what its verdicts are paired with.
- `--no-voice`: With --exam-set: leave out the voice score, which loads each arm on the device to give the writer's own text a likelihood.
- `--interval-seed <N>`: The seed the bootstrap intervals are resampled with (drawn and printed when not given; the same records and seed give the same intervals)

### splinter select

Choose the step of a run kept with its evaluations by what each writes on the dev suite: right answers, no invented specifics, then voice

```text
splinter select [OPTIONS] --exam-set <EXAM> <CANDIDATE>
```

Arguments:

- `<CANDIDATE>` (required): The candidate, trained with --keep-evaluations: its id, or a unique prefix of it.

Options:

- `--exam-set <EXAM>` (required): The dev suite (its id, or its manifest file) every kept evaluation is put to.
- `--judge <REF>`: The model that judges (default: the judge role's model)
- `--adopt`: Make the chosen evaluation a candidate of its own, to be released.

### splinter exam-report

Analyses done after an exam from what its report kept: labels to calibrate the judge by, and memorisation

```text
splinter exam-report [OPTIONS] <COMMAND>
```

Subcommands:

- [`labels-export`](#splinter-exam-report-labels-export) - Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said
- [`labels-import`](#splinter-exam-report-labels-import) - What a person's labels say of the judge
- [`memorisation`](#splinter-exam-report-memorisation) - How much of each arm's answers is the text the candidate was trained to produce again

#### splinter exam-report labels-export

Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said

```text
splinter exam-report labels-export [OPTIONS] --out <FILE> --key <FILE> <REPORT>
```

Arguments:

- `<REPORT>` (required): The exam report (`exam --exam-set ... --json` output)

Options:

- `--out <FILE>` (required): Where the items to label are written (JSON Lines)
- `--key <FILE>` (required): Where the key is written (JSON)
- `--n <N>`: Items to sample. Default: `400`.
- `--seed <N>`: Varies the sample (drawn and printed when not given; the same report and seed give the same sample)

#### splinter exam-report labels-import

What a person's labels say of the judge: agreement, kappa, false right and false wrong rates, length bias

```text
splinter exam-report labels-import [OPTIONS] --key <FILE> <LABELS>
```

Arguments:

- `<LABELS>` (required): The items, each with its `label` filled in `right` or `wrong`.

Options:

- `--key <FILE>` (required): The key `labels-export` wrote.

#### splinter exam-report memorisation

How much of each arm's answers is the text the candidate was trained to produce again: overlap of 8, 13 and 20 word runs and the longest run

```text
splinter exam-report memorisation [OPTIONS] --candidate <CANDIDATE> <REPORT>
```

Arguments:

- `<REPORT>` (required): The exam report.

Options:

- `--candidate <CANDIDATE>` (required): The candidate the report is of.

### splinter exam-set

Make and show frozen exams: families reserved from sources, tasks written from their text alone

```text
splinter exam-set [OPTIONS] <COMMAND>
```

Subcommands:

- [`create`](#splinter-exam-set-create) - Capture the sources, reserve families of them, write the exam from that text alone and freeze it
- [`power`](#splinter-exam-set-power) - Simulate the planned paired test to size an exam before it is paid for
- [`show`](#splinter-exam-set-show) - Show a frozen exam

#### splinter exam-set create

Capture the sources, reserve families of them, write the exam from that text alone and freeze it

```text
splinter exam-set create [OPTIONS] <SOURCE>...
```

Arguments:

- `<SOURCE>...` (required): Files, directories, `cmd:<command>` (quoted as one argument), or stored source ids.

Options:

- `--families <N>`: How many families to reserve. Default: `50`.
- `--tasks-per-family <N>`: Tasks of each family of the final test. Default: `8`.
- `--dev-families <N>`: How many more families to reserve for the dev suite. Default: `50`.
- `--dev-tasks-per-family <N>`: Tasks of each dev family. Default: `4`.
- `--kinds <K,..>...`: Task kinds, comma-separated. Default: `converse,advise,explain`.
- `--generator <REF>`: The model that writes the tasks (default: the policy)
- `--goal <TEXT>`: What the learner is after; steers the tasks.
- `--persona <NAME>`: Who wrote the sources, when they are one person's.
- `--seed <N>`: Varies which families are reserved (drawn and printed when not given; the same seed over the same sources reserves the same families, and exams are comparable across runs only when they share one)
- `--not-trained-by <CANDIDATE>...`: A candidate (repeat the flag for several) whose training, and that of the releases it continues, no reserved family may overlap: the exam is for measuring it.

#### splinter exam-set power

Simulate the planned paired test to size an exam before it is paid for: its power at an assumed effect, and how often it calls a difference that is not there

```text
splinter exam-set power [OPTIONS]
```

Options:

- `--families <N>`: Families of the exam. Default: `50`.
- `--tasks-per-family <N>`: Tasks of each. Default: `8`.
- `--effect <SHARE>`: The lead of the first arm in the share of tasks right (0.10 is ten points) Default: `0.1`.
- `--discordance <SHARE>`: The share of tasks only one of the arms gets right. Default: `0.3`.
- `--icc <RHO>`: The intraclass correlation of the per-task difference in a family. Default: `0.1`.
- `--from-report <FILE>`: Take the discordance and the intraclass correlation of the primary comparison from the report of a pilot (`exam --json` output), not from an assumption.
- `--replicates <N>`: Simulated exams. Default: `400`.
- `--training-runs <N>`: The training runs the result will rest on: fewer than three, or fewer than a hundred families, and it is labelled a pilot. Default: `1`.
- `--seed <N>`: Seeds the simulation (drawn and printed when not given; the same plan and seed give the same answer)

#### splinter exam-set show

Show a frozen exam

```text
splinter exam-set show [OPTIONS] <EXAM>
```

Arguments:

- `<EXAM>` (required): Its id, or its manifest file.

### splinter runs

List, inspect and cancel runs

```text
splinter runs [OPTIONS] <COMMAND>
```

Subcommands:

- [`list`](#splinter-runs-list) - List the recorded runs
- [`show`](#splinter-runs-show) - Show one run's record
- [`cancel`](#splinter-runs-cancel) - Ask a run in progress to stop

#### splinter runs list

List the recorded runs

```text
splinter runs list [OPTIONS]
```

#### splinter runs show

Show one run's record: arguments, stages, status, outputs

```text
splinter runs show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The run's id.

#### splinter runs cancel

Ask a run in progress to stop

```text
splinter runs cancel [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The run's id.

### splinter jobs

Submit work that runs outside this process, list it, reconcile it after a restart, retry a failed attempt, show one job or cancel one; plain `jobs` lists the subcommands

```text
splinter jobs [OPTIONS] [COMMAND]
```

Subcommands:

- [`submit`](#splinter-jobs-submit) - Submit a program as one job and return as soon as its workers are started and detached
- [`list`](#splinter-jobs-list) - List the submitted jobs and the state each was last seen in
- [`reconcile`](#splinter-jobs-reconcile) - Ask each unfinished job's launcher what became of it and record the answer
- [`retry`](#splinter-jobs-retry) - Start the next attempt of every job whose last attempt failed and whose retry policy allows another
- [`show`](#splinter-jobs-show) - Show one job, its state and the specification it runs
- [`cancel`](#splinter-jobs-cancel) - Stop a job's workers and record it cancelled

#### splinter jobs submit

Submit a program as one job and return as soon as its workers are started and detached

```text
splinter jobs submit [OPTIONS] <PROGRAM ARG...>...
```

Arguments:

- `<PROGRAM ARG...>...` (required): The program and its arguments, run directly and never through a shell. `{rank}` and `{world}` are filled in for each worker.

Options:

- `--distributed`: Prepare a distributed training job: write its plan and a fresh owner-only secret into every attempt directory before launching and append `--job-plan <file>`, `--job-rank <rank>` and `--job-secret <file>` to each worker. The plan registers the ranks on the address of the machine rank 0 runs on: this machine's loopback for a local job, the first node's address for a job on the allocation's nodes. `{local_rank}` in the command is a rank's place on its node.
- `--nodes <N>`: How many machines the job is placed on. Default: `1`.
- `--ranks-per-node <N>`: How many workers run on each machine. Default: `1`.
- `--gpus-per-rank <N>`: How many accelerators each worker may use (`0` for none) Default: `0`.
- `--wall-clock <SECONDS>`: The job's wall clock in seconds; a worker that outlives it is stopped and the job fails with exit code 124. `0` for no bound. Default: `0`.
- `--attempts <N>`: Attempts in all, including the first. Default: `1`.

#### splinter jobs list

List the submitted jobs and the state each was last seen in

```text
splinter jobs list [OPTIONS]
```

#### splinter jobs reconcile

Ask each unfinished job's launcher what became of it and record the answer: completed, failed, or lost when nothing can account for it

```text
splinter jobs reconcile [OPTIONS]
```

#### splinter jobs retry

Start the next attempt of every job whose last attempt failed and whose retry policy allows another

```text
splinter jobs retry [OPTIONS]
```

#### splinter jobs show

Show one job, its state and the specification it runs

```text
splinter jobs show [OPTIONS] <JOB>
```

Arguments:

- `<JOB>` (required): The job's digest, or a unique prefix of its hex.

#### splinter jobs cancel

Stop a job's workers and record it cancelled

```text
splinter jobs cancel [OPTIONS] <JOB>
```

Arguments:

- `<JOB>` (required): The job's digest, or a unique prefix of its hex.

### splinter state

Show, maintain, verify, repair, archive and restore the state: the experience database and the files it tracks

```text
splinter state [OPTIONS] <COMMAND>
```

Subcommands:

- [`status`](#splinter-state-status) - What the experience database holds, as files
- [`maintain`](#splinter-state-maintain) - Merge small files, index what is not indexed and retire finished writers
- [`unpin`](#splinter-state-unpin) - Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected
- [`verify`](#splinter-state-verify) - Check the database and every file it tracks, and report each one that is missing or damaged
- [`repair`](#splinter-state-repair) - Recover what verify finds
- [`archive`](#splinter-state-archive) - Pack the database and the files it tracks into one archive
- [`pack`](#splinter-state-pack) - Rewrite the experience database into the packed layout
- [`restore`](#splinter-state-restore) - Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first

#### splinter state status

What the experience database holds, as files

```text
splinter state status [OPTIONS]
```

#### splinter state maintain

Merge small files, index what is not indexed and retire finished writers; nothing stored changes

```text
splinter state maintain [OPTIONS]
```

Options:

- `--collect`: Also delete files nothing reaches that are past their grace period; a snapshot a dataset pinned is never touched.

#### splinter state unpin

Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected

```text
splinter state unpin [OPTIONS] <HOLDER>
```

Arguments:

- `<HOLDER>` (required): The holder's name.

#### splinter state verify

Check the database and every file it tracks, and report each one that is missing or damaged; exits 1 when anything is

```text
splinter state verify [OPTIONS]
```

Options:

- `--deep`: Read every byte instead of only checking sizes.

#### splinter state repair

Recover what verify finds: fill holes from copies (archives or other state roots), rebuild indexes, and with --accept-loss write off what no copy has

```text
splinter state repair [OPTIONS]
```

Options:

- `--from <PATH>...`: A copy to take missing files from: an archive file, or another state root. May be given more than once.
- `--accept-loss`: Give up on what no copy has: withdraw damaged database files and record lost artifacts in the ledger, which `state status` lists.

#### splinter state archive

Pack the database and the files it tracks into one archive; the same state always gives the same bytes

```text
splinter state archive [OPTIONS] <FILE>
```

Arguments:

- `<FILE>` (required): Where to write the archive (a .tar.zst)

Options:

- `--no-artifacts`: Leave the artifacts (adapters, datasets) out.
- `--since <ARCHIVE>`: An earlier archive: carry only what it lacks. Restore the result together with that archive.

#### splinter state pack

Rewrite the experience database into the packed layout: a few large files instead of one or more per experience, every object verified on the way. A packed database is compacted. Stop every process using the state first

```text
splinter state pack [OPTIONS]
```

#### splinter state restore

Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first

```text
splinter state restore [OPTIONS] <FILES>...
```

Arguments:

- `<FILES>...` (required): The archive to restore, then any it builds on.

### splinter lineage

Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it

```text
splinter lineage [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): Any artifact's id - a source, task, experience, dataset, candidate, release, answer, ... - or a unique prefix of it.

Options:

- `--up`: Only where it came from.
- `--down`: Only what came from it.
- `--both`: Both ways (the default)
- `--depth <N>`: The most edges walked from it (default: all the way)

### splinter publish

Publish experience sets, datasets, candidates and signed aggregate verdicts as one tagged, immutable revision of a private Hugging Face dataset repository; sealed and item-level material is refused

```text
splinter publish [OPTIONS] --repo <OWNER/NAME>
```

Options:

- `--repo <OWNER/NAME>` (required): The dataset repository, `owner/name`; created private when missing, and refused unless the Hub states it private.
- `--set <SET>...`: An experience set, by id or unique prefix (repeatable)
- `--dataset <DATASET>...`: A dataset, by id or unique prefix (repeatable)
- `--candidate <CANDIDATE>...`: A trained candidate, by id or unique prefix (repeatable); its adapter, training record and the datasets it was trained on come along.
- `--verdict <FILE>...`: A signed aggregate verdict of the examiner (repeatable)
- `--examiner-key <FILE>`: The examiner's public key the verdicts verify under.
- `--hub-endpoint <URL>`: The Hub to talk to (a local emulator's `http://127.0.0.1:PORT` in a test) Default: `https://huggingface.co`.
- `--token-file <FILE>`: A file holding the access token (name the `hf` CLI's own, `~/.cache/huggingface/token`, to use it); without it, calls are anonymous.

### splinter config

Read and change this machine's settings (connections and operator grants), one validated file

```text
splinter config [OPTIONS] <COMMAND>
```

Subcommands:

- [`show`](#splinter-config-show) - Print the file's path and its effective content
- [`get`](#splinter-config-get) - Print one value
- [`set`](#splinter-config-set) - Set one value, given as YAML (`set connections.hosted.token env:SVEN_KEY`, `set connections.brain '{protocol
- [`reset`](#splinter-config-reset) - Remove one value, or with --all return the file to the empty default
- [`schema`](#splinter-config-schema) - Print the JSON Schema of the file
- [`path`](#splinter-config-path) - Print the file's path
- [`import`](#splinter-config-import) - Move the settings of the older `connections.yaml` and `operators.yaml` beside the file into it, then remove them

#### splinter config show

Print the file's path and its effective content; a missing file shows the empty default

```text
splinter config show [OPTIONS]
```

#### splinter config get

Print one value. KEY is a dotted path, with `[name]` for a name that holds dots: `connections.hosted.url`, `operators[exam.sealed].network`

```text
splinter config get [OPTIONS] <KEY>
```

Arguments:

- `<KEY>` (required): The key to read.

#### splinter config set

Set one value, given as YAML (`set connections.hosted.token env:SVEN_KEY`, `set connections.brain '{protocol: openai, url: "http://127.0.0.1:8990/v1"}'`). The whole file is validated against the schema first: a refused write changes nothing

```text
splinter config set [OPTIONS] <KEY> <VALUE>
```

Arguments:

- `<KEY>` (required): The key to write.
- `<VALUE>` (required): The value, as YAML.

#### splinter config reset

Remove one value, or with --all return the file to the empty default

```text
splinter config reset [OPTIONS] [KEY]
```

Arguments:

- `<KEY>`: The key to remove.

Options:

- `--all`: Return the whole file to the empty default.

#### splinter config schema

Print the JSON Schema of the file

```text
splinter config schema [OPTIONS]
```

#### splinter config path

Print the file's path

```text
splinter config path [OPTIONS]
```

#### splinter config import

Move the settings of the older `connections.yaml` and `operators.yaml` beside the file into it, then remove them

```text
splinter config import [OPTIONS]
```
