<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Splinter reaches brain only through its API

Decided 2026-11-06. Splinter holds no model in its own process: every inference and every training
run is a request to a brain service, which owns the devices, batches concurrent clients and
schedules serving and training on the same cards.

## Why

Measured in one campaign on four GH200 cards:

- The in-process policy (`splinter-model`'s `LocalQwen`) builds brain's dense Qwen3 decoder, so a
  Qwen3.8 (qwen35) policy cannot be loaded at all; the served brain runs the same checkpoint.
- The in-process policy decodes one attempt at a time: `generation run` on an 8B policy finished 8 of
  260 attempts in 45 minutes, where a served brain runs 64 agent conversations at once at
  ~1500-1800 decode tokens/s over four cards.
- An in-process model and a served brain compete for the same card memory without either knowing.

## Design

1. **One brain service per node owns its GPUs.** Inference runs with replicas and continuous
   batching; training runs as a job the residency manager schedules on the same devices (a training
   job takes cards from serving replicas and gives them back).
2. **Splinter is a client.** Inference over the OpenAI-compatible HTTP surface (what sven already
   uses), so concurrent attempts are concurrent requests. Training as a `Subscribe` job (progress
   frames, `Cancel`), returning the trained adapter.
3. **No bulk data through the API.** On one machine, datasets and adapters move as file descriptors
   over D-Bus (memfd / `SCM_RIGHTS`, brain's existing blob protocol). Across machines, by content
   digest in a content-addressed artifact store on the cluster's shared filesystem: the client writes
   `<store>/<blake3>`, sends the digest, and brain resolves it from the store it was configured with
   (a host-resolved parameter, never a caller-chosen path), verifies it and answers with the digest of
   what it wrote. Both transports address artifacts by digest; the fd is the local fast path.
4. **Scale-out without new plumbing.** Splinter's configuration names brain endpoints (URLs or bus
   names), never a process it starts. On a Slurm cluster each node's brain is one job step that
   publishes its endpoint in a connection record on the shared filesystem; splinter's Slurm launcher
   starts them. More nodes are more endpoints; brain's replica routing keeps a conversation on the
   replica holding its prefix.

## Phases

1. Inference through a brain endpoint in `practice`, `learn`, `generation run` and the release gate,
   attempts concurrent; the in-process inference path is removed once parity holds.
2. A served chat fine-tune / policy-gradient job in brain (dataset and adapter by fd or digest,
   scheduled by residency); `splinter train` calls it; the in-process model is gone.
3. The endpoint registry and Slurm launch of brain workers.

## Phase 1 as built

`--brain-endpoint URL` (repeatable) with `--brain-key-file`, `--attempt-concurrency` and
`--brain-adapter-manifest` make every `local:` and `policy:` reference a request to the endpoint
(`splinter_model::endpoint`). A base is found by `<org>/<model>`, an adapter by the digest of its file
in the model list, and a reference the endpoint does not list is refused by name. Attempts of the
solve stage and of the gate's exam probes run `--attempt-concurrency` at a time (default 16 per
endpoint: brain does not advertise its `--max-batch` in the model list); an attempt is pinned to an
endpoint by a stable hash of its seed. The gate starts no `brain` of its own with endpoints: its
serve check is that each endpoint lists the candidate under its digest, and a candidate is offered
by publishing the release manifest the endpoint follows (`brain serve --adapter-manifest`), one
adapter at a time, so the two arms are graded one after the other.

What is not done:

- The in-process inference path (`LocalQwen`, residency-held bases, exams and closed-book answers
  through `answer`/`exam`, the embedder) is still the path without endpoints, and `exam`,
  held-out scoring inside training, `ask` and the scripted specs run on it. It goes when the
  endpoint is the only path and those callers ask through it.
- Phase 2 (training): a served chat fine-tune / policy-gradient job; the dataset and the adapter
  move by fd over D-Bus or by blake3 digest in a content-addressed store the job is given. Until
  then training loads the base in this process, so a Qwen3.8 generation cannot train, only be
  measured.
- Phase 3: the endpoint registry and Slurm launch of workers. Endpoints are named on the command
  line, a conversation is pinned to one by hash (not by the replica holding its prefix), and the
  key travels as one shared file.
- A closed-book solve (no workspace) is pinned by model, not by attempt: only workspace attempts
  carry a per-attempt seed to hash.
- Per-response thinking budgets are not sent on the served path of a local reference.
