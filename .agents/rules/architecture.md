<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Architecture rules

Read this before you design anything, add a crate or a dependency, or decide
which crate owns a change. `architecture.toml` is authoritative for the tiers,
the forbidden pairs and the reviewed exceptions; `make check/gates/architecture`
enforces it against `cargo metadata`, and `scripts/gates/test-architecture.py`
is the gate's own specification. What each crate owns is stated in its
`src/lib.rs` documentation.

## Ownership

| System | Owns | Must not own |
|---|---|---|
| **sven** | Running agents: tools, sessions, delegation, permissions, the authoritative trajectory of what an agent did | What to learn, training, release decisions |
| **brain** | Model computation: inference, training algorithms, evaluation arithmetic, checkpoints, adapters, serving | Which experiences qualify, campaign scheduling, release authority |
| **Splinter** | The learning campaign: intent, task acquisition, evidence admission, dataset curation, run scheduling, lineage, release decisions | Re-implementations of anything sven or brain provide |

Neither sven nor brain depends on Splinter.

**sven** is consumed through its public SDK, `sven-sdk`, by `splinter-agent`
(which runs agents and typed calls) and `splinter-model` (which implements
sven's model provider for the in-process model). The vocabulary holds sven's
trajectory format, `atif`, which `sven-sdk` re-exports. Needing any other sven
crate is a gap in sven's SDK: fix it there and record it in
`.agents/roadmap/splinter.md`, never reach further in.

**brain** is reached by `splinter-model` alone. It may use any crate of brain's
repository: it instantiates the SDK's higher-level primitives first (the
`brain` crate: devices, residency, models) and reaches into another brain crate
only when the state it needs sits there. Nothing in Splinter re-implements what
brain computes.

**The rule that decides a boundary:** each project proves its own capabilities
with its own standalone samples, and Splinter proves they compose. If moving
something into Splinter leaves sven or brain unable to demonstrate a capability
without Splinter, the boundary is drawn wrong. A reusable mechanism (a training
objective, a replay sampler, a verifier interface) stays in the project that
owns the computation; Splinter keeps the decision about when and on what to use
it.

## Tiers

Dependencies flow down; a crate depends only on lower tiers, except for the
same-tier pairs `architecture.toml` lists with their reasons.

```text
sample, surface   splinter (CLI) and samples/*: depend on the SDK and nothing else
sdk               splinter-sdk: the one way to embed Splinter
workflow          splinter-pipelines: stages and the pipelines that order them
application       splinter-orchestrator: what a run is carried out with
adapters          splinter-agent (sven) and splinter-model (brain and sven's provider)
domain            splinter-knowledge: intake and task construction
measurement       splinter-eval and splinter-data: what is measured, what is trained on
storage           splinter-store and splinter-sandbox: state on disk, processes
foundation        splinter-core (the vocabulary) and splinter-expdb (the database)
```

## Where a change goes

| If you are adding... | It belongs in... |
|---|---|
| A new noun, an id, or an invariant of one | `splinter-core` |
| A new kind of durable state, or a way to read it | `splinter-store` (or the crate whose types it embeds, when it cannot sit below them) |
| A new place code runs | `splinter-sandbox` |
| A new verifier, a measurement, a release-gate check | `splinter-eval` |
| A new training projection, or a change to the dataset format | `splinter-data` |
| A new source parser, chunker, or task-construction rule | `splinter-knowledge` |
| Anything that runs a sven agent or a typed sven call | `splinter-agent` |
| Anything that touches brain: inference, training, residency, what a trainer can read | `splinter-model` |
| A setting, a model role, a service a process shares, run recording | `splinter-orchestrator` |
| A stage, a pipeline, the order stages run in | `splinter-pipelines` |
| A public type or entry point for callers | `splinter-sdk` |
| Command-line grammar or output | `splinter` |
| A controlled learning sample | `samples/<name>`, on the SDK alone |

Two questions before adding anything:

1. *Could a completely different learning pipeline use this?* If yes, it
   belongs below the pipelines. If no, it belongs in the pipeline.
2. *Would sven or brain still make sense without Splinter if this lived
   there?* If moving it upstream keeps them whole, it probably belongs there.

## Laws

- **One implementation.** Never copy code out of sven or brain; depend on it.
- **A mechanism never decides when it is used.** A verifier is eval's; which
  verifier grades a kind is a pipeline's. A trainer is the model adapter's;
  choosing supervised or preference training is a pipeline's. A typed call is
  the agent's; asking it is a stage's.
- **No learning algorithm in the CLI, the SDK or the orchestrator.** They
  compose and present; algorithms live in the project that owns the
  computation, and the decision to use one lives in a pipeline.
- **A model is asked for by role.** No stage names a model; it names a role
  (`policy`, `teacher`, `generator`, `planner`, `judge`, `critic`, `router`)
  and configuration decides who plays it (`splinter_core::role` holds the
  rule). No code that generates tasks calls brain.
- **An experience is immutable.** Verdicts, rewards, step labels and dataset
  membership are facts derived about it. Never rewrite an experience to encode
  a later interpretation: add an annotation, or project a new view.
- **No crate per training algorithm, and no experience format per pipeline.**
  Experience is collected once and reinterpreted; algorithms are brain's.
- **The stage engine records, stages do not.** A stage does its work and
  reports what it did; cancel, budget, skipping, timing and the run record are
  the engine's.
- **Configuration is a value.** Only `splinter-orchestrator`'s `config` module
  reads or writes the environment; everything below takes settings as
  arguments (`make check/gates/env-reads`).

## Changing one thing touches several

- **A training objective:** the objective and its line format (`data`); what
  the backend can train from a file (`model` capabilities and its dataset
  check); the stage that picks the objective (`pipelines`); the SDK's exposure
  and the docs.
- **A task kind:** the kind catalogue and its admission rules (`knowledge`),
  the verifier that grades it and the kinds' names (`eval`, `core`), a
  fixture, an evaluation suite, the docs.
- **A model role:** the role and its fallback rule (`core::role`), the
  configuration that feeds it (`orchestrator`), the stage that asks for it,
  `status` output, the docs.
- **A pipeline:** its stages and order (`pipelines`, on the stage engine), its
  request and report, the SDK method, the CLI verb if it has one, a scripted
  integration spec, and the README's tables of what Splinter learns from and
  produces when either changes.
- **A kind of task the gate probes:** the suite that holds its tasks and how a
  model is attempted at them (`pipelines` release), the numbers the report
  keeps and the rule that decides them (`eval` gate), the exam or anchor that
  supplies the tasks, the serve check's comparison of a served run with the
  in-process one, the CLI's report of the gate, the docs.
- **A field of an experience:** the vocabulary (`core`), its canonical form and
  golden test (`crates/store/tests/identity.rs`), ingestion, every projection
  that reads it, and a note on compatibility with what is already stored: an
  address must not change for state that is already there.
