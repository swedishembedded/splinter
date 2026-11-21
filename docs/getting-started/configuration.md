<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Configuration

A run needs no flags. Splinter finds what this machine holds: brain's model
store (the one `brain pull` fills), the `brain` on `PATH` it serves models
from, the GPUs, and the
strongest chat model in the store that fits a card, which then helps the
policy learn (see [Roles](#roles)). Every choice is a flag, valid on every
command; nothing is read from environment variables. The flags are listed
under [global options](../reference/cli.md#global-options).

## State

Runtime state (sources, tasks, experiences, datasets, candidates, releases, answers, runs) lives under
`~/.splinter`. Override it with `--state DIR`. State left in `~/.sven/splinter` by an
earlier version is still used until `~/.splinter` exists; move it there to adopt the new location. Layout and
maintenance are in [Storage](../reference/storage.md).

## Jobs

`--job-owner NAME` is the controller name recorded with each job this process
submits and with the claim it takes on one (default: this machine's host name).
`--job-attempt-root DIR` writes a job's attempt directories under `DIR`
(default: `jobs` under the state root).

`--job-launcher KIND` selects what starts a job's workers. `local` (the
default) runs every rank as a detached process on this machine. `slurm`
submits the job to Slurm's batch system, which needs `--slurm-sbatch PATH`,
`--slurm-squeue PATH` and `--slurm-scancel PATH`: the programs that submit a
job, report its state and stop it. `--slurm-arg ARG` passes an argument to
every submission before the batch script, for a partition or an account, and
repeats for more than one. Every one is a flag; the environment is not read,
and a program the Slurm launcher needs but does not have is refused by name
before anything is submitted.

## Cluster allocations

Inside a Slurm allocation, Splinter reads the job from the scheduler's
variables: its id (`SLURM_JOB_ID`), its nodes (`SLURM_JOB_NODELIST`, the
compressed host list such as `n[424,442-445]`), the GPUs each node was granted
(`SLURM_GPUS_ON_NODE`, else the indices of `SLURM_JOB_GPUS`) and the node it
runs on (`SLURMD_NODENAME`). A value that does not parse is refused by name;
it is never read as "no allocation". `splinter doctor` shows what was
detected and which nodes a run uses:

```bash
splinter doctor --cluster-nodes n442
# ok   cluster: job 5600435: nodes n424,n442,n445,n467 (4 GPUs each), this node n424; using n442 (chosen by --cluster-nodes, 4 GPUs): a brain server per node
```

| Flag | Meaning |
|---|---|
| `--cluster auto\|off` | `auto` (the default) uses an allocation of more than one node, or the nodes `--cluster-nodes` names; `off` keeps everything on this machine, as outside an allocation. An allocation of one node with no nodes named is a single-machine run. |
| `--cluster-nodes HOSTLIST` | The nodes this process may use, in host-list syntax (default: every node of the allocation). Other sessions may own some nodes of a shared allocation; name yours. A node outside the allocation is refused. |
| `--cluster-exec-prefix WORDS` | What a program started on a node runs under, split at spaces. A step starts on the node's host, not in a container this process runs in. |
| `--slurm-srun PATH` | The `srun` steps are started with (default: the first on `PATH`); `--slurm-scancel PATH` stops one. |
| `--slurm-step-arg ARG` | An argument every step passes to `srun` (`--mem=64G`); repeats. |
| `--slurm-single-node-step-arg ARG` | An argument a single-node step passes besides; on an HPE Slingshot fabric `--network=single_node_vni`, without which a single-node step has no fabric access; repeats. |

## Machine and models

| Flag | Meaning |
|---|---|
| `--models-dir DIR` | Brain's model store (default: the store `brain pull` fills, `~/.local/share/brain/models`). |
| `--policy-base CHECKPOINT` | The policy's base checkpoint, a path or a name in the store (default `Qwen/Qwen3-0.6B`). |
| `--brain-bin PATH` | The `brain` binary Splinter starts a server from, and a release is served on (default: the first `brain` on `PATH`). |
| `--brain-endpoint URL`, `--brain-key-file FILE` | A running brain server to request every local model from, instead of starting one (see [Serving models from brain](#serving-models-from-brain)). |
| `--assistant-model REF` | The stronger model that plans, writes tasks, teaches, judges and reads sentences when the command names none (default: none, the policy plays its roles; see [Roles](#roles)). |
| `--judge-model REF` | The model that judges what only a judge can decide (default: the assistant). |
| `--front-door-model REF` | The model that reads sentences (default: the assistant). |
| `--default-budget DURATION` | How long a `learn` may take when it names no `--budget` (a sentence names none); unbounded when not given. |
| `--thinking` | Lets local models reason before they answer. Off, a reasoning model is asked with its reasoning block closed, so every reply is the answer and comes at once; training renders its answers the same way. |
| `--min-calibration-controls N` | The fewest controls a judge is measured on before its verdicts count (default 16). |
| `--allow-remote` | Permits models reached over the network. |
| `--docker PATH`, `--workspace-image IMAGE` | A docker client and a digest-pinned image: an agent's commands then run in a container with no network. |
| `--workspace-scratch DIR` | Where trees are laid out for repository tasks. |
| `--cargo-home DIR` | A cargo home the commands of a Cargo project build offline from. |

`--bf16-base` (on `learn`, `practice` and `absorb`) and `train --base-precision` hold a large policy's
base at bf16 so it trains on one card. `train --base-precision` also takes `int8`, `nf4` and `fp8`
(block-scaled E4M3, half of bf16: an fp8 checkpoint's own format, so nothing is requantised); brain refuses a
tier the base cannot hold, naming the formats it can.

## Model references



Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`--policy-base`, else `Qwen/Qwen3-0.6B` in brain's model store) with the champion's adapter, the release `default` points at; the base alone before any release |
| `policy:<alias>` | the base with the adapter of the release `<alias>` points at (`release --alias`) |
| `local:<checkpoint>[+<adapter>][@<tokens>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+`. The context is the largest the checkpoint supports (its `max_position_embeddings`); `@<tokens>` limits it, as fitting a large model on a card requires (a 14B judge on a 24 GiB card: `local:Qwen/Qwen3-14B@4096`) |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A policy alias is resolved once per command, when it is first used: a
run keeps the release it started with however the alias moves meanwhile,
and records it (`learn`'s first stage, a candidate's `parent`).

The policy's context is limited with the global `--policy-context-tokens N`
(a policy too large for the card at its maximum), any other local model's by
its reference.

A remote reference is refused unless `--allow-remote` is given; that is the
only way Splinter uses the network. Its key is the one sven holds for the
provider: store it once with `sven auth set openrouter` (or the provider's id).

## Serving models from brain

Splinter reaches brain only through its API. A local model (`local:...`, `policy:...`) is never loaded in
the process: it is requested from a `brain serve --openai` by the id that server serves it under, and the
attempts of `practice`, `learn`, `generation run`, `solve` and the release gate run against it at once,
which brain's continuous batching decodes together.

Without flags, Splinter starts such a server for the length of the command: the `brain` on `PATH` (or
`--brain-bin`), over the model store, with the adapters Splinter trains served from a directory under the
state root and the training workload on. `--device` and `--backend` are brain's own flags, handed to that
server; `--attempt-concurrency N` is its `--max-batch` and the number of attempts asked of it at once (16
when not given). With no `brain` binary and no endpoint, a command that needs a local model is refused
with what to name.

Given a running server, Splinter starts nothing:

```text
brain --device gpu3 --backend cuda serve --openai 127.0.0.1:8805 --api-key-file KEY \
      --checkpoint ~/models/Qwen/Qwen3.8-27B-FP8 --max-batch 48
splinter --brain-endpoint http://127.0.0.1:8805/v1 --brain-key-file KEY \
         --attempt-concurrency 48 --policy-base Qwen/Qwen3.8-27B-FP8 solve ...
```

`--brain-endpoint URL` repeats for more endpoints; an attempt is pinned to one by a stable hash of its
seed, so its conversation stays cached on one server. `--brain-key-file` names the file holding the key
(the one word of `brain serve --api-key-file`). `--attempt-concurrency N` is how many attempts are in
flight in all; set it to the endpoints' `--max-batch`, because brain does not advertise it (16 per endpoint
when not given).

A base is requested as `<org>/<model>`, the last two components of its checkpoint path, which is how
brain's model store names it. An adapter is found on the endpoint by the digest of its file. A reference
the endpoint does not list is refused by name, with what it does list. `--brain-adapter-manifest FILE` names
the release manifest the endpoint follows (`brain serve --adapter-manifest FILE`): an adapter the endpoint
does not list, a release candidate in the gate for example, is offered by publishing it there and waited
for. A manifest offers one adapter at a time, so the gate's two arms are asked one after the other. With
endpoints, the gate starts no `brain` of its own: its serve check is that each endpoint lists the candidate
under the digest the release records. `--brain-adapter-dir DIR` instead names the directory the endpoint
serves (`brain serve --serve-adapters DIR`): an adapter is offered by writing it there under its digest, and
none is withdrawn, so a champion and its candidate are served at once, as a campaign's agent examination
needs.

### Training on a brain endpoint

With endpoints, `train` and the training stage of `learn`, `practice` and `generation run` do not load the
base in the process either: a supervised fine-tune and a policy-gradient run (self-distillation with
teacher contexts included) are jobs of the brain service, which schedules them on the cards it serves
from, taking them from idle serving replicas and giving them back.

```text
brain serve --openai 0.0.0.0:8805 --api-key-file KEY --models-dir ~/models \
      --training --artifact-store /shared/artifacts
splinter --brain-endpoint http://gpu-node:8805/v1 --brain-key-file KEY \
         --brain-artifact-store /shared/artifacts --policy-base Qwen/Qwen3.8-27B-FP8 train ...
```

`--brain-artifact-store DIR` is the directory the service was started with as `--artifact-store`: the
datasets (and the adapter a run continues) are published there as files named by their BLAKE3 digest, the
job names them by digest, and the trained adapter and its training record come back the same way, so no
dataset travels through the API and the directory may be on a filesystem several nodes share. The base is
requested by the id brain's store gives it, as for inference. Every dataset is read first by the service's
own trainer (`check_dataset`), which names a record it refuses by line and sizes the training row to the
longest example, before a run is asked for. A run the service has no room for (every card that could
hold it is serving a request) is asked again, of every endpoint, until it is placed or the run is
cancelled; a request the service refuses outright is reported with its reason. Supervised, policy-gradient
and preference (DPO) runs, full-parameter runs and adapter merges are all jobs of the service; a checkpoint
a full-parameter run or a merge makes is installed in the server's model store as `local/<name>`. Without
endpoints, Splinter starts a server of its own (`--brain-bin`) and trains through it the same way: nothing
trains in Splinter's process.

## Roles

Stages ask for a model by role, never by name: `policy` (the model being trained), `teacher`,
`generator`, `planner`, `judge`, `critic` and `router`. A role a command does not name
(`--generator`, `--teacher`, `--judge`, `--planner`) is played by the assistant
(`--assistant-model`), the judge by `--judge-model` and the router by `--front-door-model` when they
are named; the policy plays the critic, and every role the assistant would play when there is none.

With no assistant named, the policy plays the assistant's roles. Name a stronger model with
`--assistant-model` (for example `local:Qwen/Qwen3-14B@8192` beside a 0.6B policy): Splinter does not
pick one itself, because which stored model fits which card at which context is brain's to compute and
its API does not answer it.

`splinter doctor` checks that the model each role would get resolves to weights on disk, and warns
when the policy is left to teach, judge, plan, write tasks or read sentences for itself: that run
can start, but rarely trains anything.

## Request limits

`--remote-concurrency` is how many requests to a model reached over an API may be in flight at once (4 by
default); a model on the local device is asked one at a time. `--temperature` is the sampling temperature (0 to
2) asked of a model reached through sven's providers, the provider's own default when not given.
`--solve-deadline` is how many seconds one task's solve may take on the wall clock (300 by default); raise it
when many tasks share one served model, so a task is not failed for waiting in the queue.
