<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Architecture

Splinter is the learning campaign built on two other projects. It decides what is learned, which
experience qualifies as training material and whether a result is released. It does not run
agents and does not compute with models.

| Project | Owns |
|---|---|
| [sven](https://swedishembedded.com/docs/sven) | Running agents: tools, sessions, delegation, permissions, and the authoritative record (trajectory) of what an agent did |
| [brain](https://swedishembedded.com/docs/brain) | Model computation: inference, training algorithms, evaluation arithmetic, checkpoints, adapters and serving |
| Splinter | Intent, task acquisition, evidence admission, dataset curation, run scheduling, lineage and release decisions |

Neither sven nor brain depends on Splinter.

## What Splinter implements

- Turning a request into sources, tasks and a recorded run, and the controller that runs it.
- Knowledge intake: sources, sections, claims and probes, with provenance.
- The verifiers a policy cannot reach, the held-out, retention and anchor suites, the exam and
  the release gate. See [Verification](verification.md) and [Release policy](release-policy.md).
- Deciding which experience becomes training data, and building versioned datasets.
- Releases: immutable adapters or checkpoints with manifests, the `default` alias, and lineage
  from an answer back to its sources.
- Owning the runtime's resources: one resident copy of the base model shared by the policy,
  the candidate and the champion (different LoRA adapters on one base), released before a stage
  that loads its own copy, such as training or the gate's `brain serve` check.

## What it consumes from sven

Splinter runs every agent through sven's public SDK (`sven-sdk`) and holds sven's trajectory
format, ATIF, in its vocabulary. A task is solved through sven in exactly the environment it
records: closed-book with no tools, a runtime with a single code tool, or a change to a
repository made in a fresh copy of its snapshot with sven's file, search and shell tools.
The same typed-call mechanism asks a model for task proposals, claims and judgments. The
`splinter-agent` crate is the only code in Splinter that runs sven; `splinter-model` implements
sven's model-provider interface so that sven's agents can run the in-process model.

## What it consumes from brain

Inference, LoRA and full fine-tuning, DPO, adapter merging, held-out scoring, and `brain serve`
all belong to brain. The `splinter-model` crate is the only one that touches it: it supplies the
in-process model provider sven's agents use, residency of the base, training runs, and the release
manifest that `brain serve --adapter-manifest` follows.

## Layers

Crates are in tiers; a crate depends only on lower tiers, apart from the reviewed same-tier
exceptions the repository's `architecture.toml` lists. A gate enforces it.

```text
sample, surface   splinter (CLI) and samples/*: depend on the SDK and nothing else
sdk               splinter-sdk: the one way to embed Splinter
workflow          splinter-pipelines: stages and the pipelines that order them
application       splinter-orchestrator: what a run is carried out with
adapters          splinter-agent (sven) and splinter-model (brain)
domain            splinter-knowledge: intake and task construction
measurement       splinter-eval, splinter-data: what is measured, what is trained on
storage           splinter-store, splinter-sandbox: state on disk, processes
foundation        splinter-core (the vocabulary), splinter-expdb (the database)
```

Models are asked for by **role**, never by name: policy, teacher, generator, planner, judge, critic
and router. Configuration decides who plays each; with nothing configured, the policy plays all
of them. See [Configuration](../getting-started/configuration.md).

## Where things run

| What | Where |
|---|---|
| Task generation, admission rules and verifiers of closed-book kinds | In Splinter (`splinter-knowledge`, `splinter-eval`), by code |
| Executable checks and repository tests | In the sandbox: on the host under kernel confinement, or in a container when one is configured |
| The agent's attempt | Through sven, in the environment the task records |
| Model inference and training | In brain: in the learning process for training and the agent's model, and in a `brain serve` process for the release's serve check |
| The judge | A model other than the teacher, generator and policy, calibrated on controls before its verdicts count |
| The release gate and the exam | In Splinter, on suites the policy under training never saw; the serve check starts a real `brain serve` |

## After release: no Splinter needed

A release is an ordinary brain adapter with a manifest. It is served by plain `brain serve` on
the base it was trained on, and plain sven can use it through its brain provider. See [Serve a
release without Splinter](../guides/releases.md#serve-a-release-without-splinter).

## Storage

Everything Splinter knows lives in one experience database under the state root, with adapters and
datasets kept as plain files beside it. See [Storage](../reference/storage.md).
