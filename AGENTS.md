# AGENTS.md - Splinter

Splinter is the end-to-end learning system built on top of **sven** (agent
execution) and **brain** (inference and training). You tell it, in plain
language, what to learn - a document, a command-line tool, a task - and it
acquires that capability, measures that it did, and releases a better
version of its own model. The released model is then deployable without
Splinter: on plain `brain serve`, or through sven talking to brain.

## Repository scope

**In scope** - the learning campaign and nothing it can get from its
dependencies:
- Turning a request ("learn this document", "learn this tool") into tasks,
  sources and a campaign; the controller that runs it.
- Knowledge intake: sources, sections, facts and probes, with provenance.
- Deciding which experience qualifies as training material, and curating
  versioned datasets from it.
- Sample design and measurement: verifiers the policy cannot reach,
  held-out, retention and anchor suites, release gates.
- Releases: immutable adapters or full checkpoints with manifests, the
  default alias, lineage from an answer back to its sources.
- Owning the embedded runtime's resources (one resident base, phases).

**Out of scope** - use the dependency, or fix it there:
- Running agents, tools, sessions, delegation, trajectories: **sven**.
- Model computation - inference, training algorithms, evaluation arithmetic,
  checkpoints, adapter formats, serving: **brain**.
- Anything that re-implements either of those locally.

**Naming other projects.** Splinter depends on sven and brain, so it names
them. It names no other project: describe an orchestrator or host
generically. `make check` enforces this (`check-repo-scope`).

## Ownership - what belongs where

| System | Owns | Must not own |
|---|---|---|
| **sven** | Running agents: tools, sessions, delegation, permissions, the authoritative trajectory of what an agent did | What to learn, training, release decisions |
| **brain** | Model computation: inference, training algorithms, evaluation arithmetic, checkpoints, adapters, serving | Which experiences qualify, campaign scheduling, release authority |
| **Splinter** | The learning campaign: intent, task acquisition, evidence admission, dataset curation, run scheduling, lineage, release decisions | Re-implementations of anything sven or brain provide |

Neither sven nor brain depends on Splinter. Splinter consumes both through
their public SDKs (`sven-sdk`, `brain`), and holds sven's trajectory format
(`atif`, which `sven-sdk` re-exports) in its vocabulary; every import of a brain or sven
crate other than those two is a gap in that project's SDK, tracked in
`.agents/roadmap/splinter.md`, not a licence to reach further in.

**The rule that decides a boundary:** each project proves its own
capabilities with its own standalone samples, and Splinter proves they
compose. If moving something into Splinter leaves sven or brain unable to
demonstrate a capability without Splinter, the boundary is drawn wrong. A
reusable mechanism (a training objective, a replay sampler, a verifier
interface) stays in the project that owns the computation; Splinter keeps
the decision about when and on what to use it.

## Conventions

- **One implementation.** Never copy code out of sven or brain. If Splinter
  needs something they have, depend on it; if their SDK does not expose it,
  fix the SDK in that repository.
- **Evidence over claims.** A learning run is judged by a measurement the
  policy under training cannot influence: held-out probes scored against
  source material, retention of everything learned before, and an anchor
  suite for general behaviour. "The model said it learned" is not evidence;
  neither is a training loss.
- **Unmeasured is not zero.** A number that was not measured is absent, never
  reported as `0`.
- **Zero warnings.** `cargo clippy -D warnings` clean, including warnings the
  change did not cause.
- **ASCII punctuation.** No em/en dashes, curly quotes or ellipsis
  characters anywhere; the text-hygiene hook repairs them.
- **Every source file** starts with the SPDX line and the copyright line
  (`scripts/spdx/check.py --fix` adds them).
- **Linear history.** Rebase, never merge; the pre-push hook refuses merges.
- **No trailers.** `Co-Authored-By:`/`Claude-Session:` lines are stripped at
  commit and refused at push.

## Where things go

| What | Where |
|---|---|
| A non-obvious defect or a gate that lied, with the number that proved it | `.agents/knowledge/<NNN>-<slug>.md`, listed in `.agents/knowledge/index.md` |
| Outstanding work | `.agents/roadmap/` |
| Research and design notes that inform the roadmap | `.agents/research/` |
| Research papers (finished, citable writing) | `papers/<topic>/`, LaTeX, builds with `make` |
| User-facing documentation | `README.md` |

Code never cites a `docs/` or `.agents/` path (a gate enforces it): state the
fact inline instead.

## Research papers

`papers/` holds research papers, not plans. A paper is a proper, high-quality
paper in the style of an arXiv submission (abstract, method, evaluation,
results, limitations, references; buildable to PDF), and it contains only
findings that are proven and verified. Roadmaps, intentions, open questions
and "next steps" belong in `.agents/roadmap/` and `.agents/research/`; a paper
that reads like one of them is wrong.

- **Route by scope.** Every paper states its scope in its introduction and
  `papers/README.md` lists them. Put a new finding in the paper whose scope it
  falls in; if none fits, start a new paper directory and add its scope there.
- **Keep papers current.** A change that produces a new verified finding, or
  invalidates a stated one (a corrected defect, a re-measured result, a
  changed method), updates the affected paper in the same change: text,
  tables, figures and the numbers they rest on.
- **Verified means reproducible.** Every number traces to a measurement that
  can be rerun with the commands the paper documents, comes with its sample
  size and uncertainty, and is labelled as a pilot when the evaluation was
  underpowered or not frozen in advance. A result that was not measured is
  absent, never zero; a result that did not hold is reported as such.
- **No speculation as findings.** Claims the evidence does not support stay
  out or are stated as limitations. Every citation is checked to exist.
- **The paper must build.** `make` in the paper directory produces the PDF
  with no undefined references or citations before the change is committed.

## Architecture

Crates in tiers; a crate depends only on crates in a lower tier, except for
the reviewed same-tier exceptions `architecture.toml` lists. That file is
authoritative: it names every crate's tier, the pairs that must never be
reachable from one another, and the exceptions, each with its reason.
`scripts/gates/check-architecture.sh` enforces it against `cargo metadata`,
and `scripts/gates/test-architecture.py` is the gate's own specification.

Dependencies flow down. Read the tiers from the bottom:

```text
sample, surface        splinter (CLI) and samples/*: depend on the SDK and nothing else
sdk                    splinter-sdk: the one way to embed Splinter
workflow               splinter-pipelines: stages and the pipelines that order them
application            splinter-orchestrator: what a run is carried out with
adapters               splinter-agent (sven) and splinter-model (brain): the only code that
                       talks to either
domain                 splinter-knowledge: intake and task construction
measurement            splinter-eval and splinter-data: what is measured, what is trained on
storage                splinter-store and splinter-sandbox: state on disk, processes
foundation             splinter-core (the vocabulary) and splinter-expdb (the database)
```

| Crate | Tier | Owns | Depends on |
|---|---|---|---|
| `splinter-expdb` | foundation | the experience graph database, with no internal dependencies; `splinter-store` stores experiences in it: immutable content-addressed segments, blob packs and index runs on plain files; manifests as the only transaction layer; per-writer ids so ingest needs no coordination or file locks; the graph of tasks, states, decisions, transitions, counterfactual sets and episode families; modality-neutral episodes of streams on explicit clock domains with events, actions and correspondences; evaluations, derivations and skills kept apart from raw experience; pinned snapshots, compaction and garbage collection, whose storage protocol is model-checked in TLA+; queries and training views (imitation, preference, group-relative, transitions, step labels, world-model windows, contrastive pairs, masked prediction, action chunks) as reproducible projections | - |
| `splinter-core` | foundation | the vocabulary, with no I/O: the digests (blake3 content addresses; sha256 for what a tool reports) and the canonical JSON they are computed over, the UTC clock, the source (its parts' content addressed, spans resolved to exact bytes), the task, the immutable content-addressed experience with sven's trajectory format, the annotations made about an experience (verdicts ranked by strength, step labels, relations), the chat wire shapes and the one system prompt every model run on a task answers under and every chat record shows, the names tasks travel under, the rule that an instruction must stand on its own without the material a student is not shown, how a command names a model (`policy:`, `local:`, `remote:`), and the roles models play in a run with the one rule that decides who plays each, longitudinal records (a `timeline-v1` line with interventions) and the participant-safe history made from one: raw identifiers replaced by opaque keyed keys, every item tied to its file line and terms, one content address per history, each with the invariants it keeps | - |
| `splinter-store` | storage | durable state: the injected `StateRoot` and its layout, time-ordered ids, the decision rule that turns an experience's verdicts into one decision and a reward, the run record every command that writes pipeline state keeps (stages, status, outputs, the cross-process cancel request, kept as events and signals in the experience database); the source store - sources and their parts' content stored once per digest; the task store - generated tasks under their address and named task sets recording how each was generated; and the experience store, kept in the experience database through one shared `Workspace`: immutable content-addressed experiences (each also an attempt in the graph), annotations as ranked evaluations of it, experience sets, the derived reward; longitudinal datasets kept as one immutable content-addressed episode per participant, imported from a record file idempotently and in bounded memory; the lineage of datasets, training runs and releases back to the attempts and episodes; maintenance of the database; documents (canonical-JSON records checked against their address on every read), pointers (a name's whole history, moved by compare-and-set), artifacts (the plain files tools need, tracked by the database) and the loss ledger; the deterministic state archive, verified restore, verify and repair | core, expdb |
| `splinter-sandbox` | storage | where a solver's code runs: the one bounded process runner (explicit environment, process-group kill at a timeout, output caps, rlimits), the runtime registry and what a runtime resolves to (version, executable digest), the process sandbox (bounded, not isolation) and the container sandbox (image pinned by digest, no network, read-only root), and environments whose snapshot digest pins everything that determines their behaviour | core |
| `splinter-eval` | measurement | measurement: the release gate's four checks and how each is decided (over a sign test it is handed), the predictive gate over metrics on paired held-out units, pass@k and where a pass rate and a teacher's verified answer put a task (always, frontier, taught, never), and verifiers by strength (executable checks run in the task's sandbox, mutation-validated generated tests, formal exact match, agreement between answers, judge calibration) with the composite that annotates all their verdicts, a critique verified by the outcome of the retry it led to, and two models compared on the items both were graded on | core, sandbox, store |
| `splinter-model` | adapters | the model being trained, and the only crate that touches brain: the in-process provider behind sven's `ModelProvider`, `ModelSelection` and its identity, the residency (one resident copy per base checkpoint, shared by every model on it, each model's adapter attached before its generation, released for training and serving), the model loaded for a command (quiescing the device when dropped), LoRA fine-tuning with replay, continuation and held-out scoring, DPO preference fine-tuning scored by brain's preference score, which dataset formats brain trains and the check of a dataset file against its parser, and brain's paired sign test as the gate's significance | core, data, eval |
| `splinter-agent` | adapters | a task solved through sven in exactly the environment it records (closed-book with no tools, or a runtime with the one tool `run_code`), bounded by sven's run options, under Splinter's system prompt in place of sven's; the open-book prompt a teacher's solve and `ask --open-book` are sent; the judge verifier, a different model grading closed-book through that same solve; the critic, a model saying closed-book what is wrong with a failed attempt from its evidence summaries, never the teacher's material; the retry with that critique, the bounded critique-and-retry loop that records the chain as relations, the typed sven call every other crate asks a model through (`TypedCall`, the empty toolset and the bounds set in one place), and the typed call that asks a model for task proposals | core, eval, knowledge, sandbox, store |
| `splinter-knowledge` | domain | intake: capturing documents, repository trees and command runs as sources, addressable sections, the text gates a generated task is held to, the concepts a task exercises (declared, else the source sections its evidence falls in, else its kind), the grounding material a teacher is shown of a task (the sections its evidence falls in, its passages and hints, never its reference), the denoise task generator, and model task generators: the task kind catalogue as data, a generator model's proposals (asked for through a `TaskProposer` the knowledge crate is handed) admitted by code (grounded, self-contained, naming their subject, executed where computed, deduplicated, never contradicting one another), and the rehearsal tasks a base model answers for its own answers to be replayed (sums and format requests built by code with their references, the brief a model writes general requests from, and the admission that keeps every anchor task and its near copies out) | core, eval, sandbox, store |
| `splinter-data` | measurement | the holdout rule (which records are held out, a group never divided), the split of a dataset file by it, and the split of the records to train on by the same rule into the records fitted and the monitoring records a run scores as it trains; projections of the experience store into training records for one objective (final answers, steps, critiques, preferences, verdicts, decisions, retrieval, outcomes, denoise, raw text, a model's own answers as given), every conversation starting with the system turn every solve runs under, what a student may see of privileged material (a teacher's solve included), the dataset each objective is written as with its manifest (the writer takes the trainer's check as a parameter and knows no backend), the projection of longitudinal episodes into participant-safe `timeline-v1` records at a prediction point, written with a manifest naming episodes, source files and combined terms, the splits of a record dataset (by participant group, by temporal cutoff, leaving one source out) with the leakage gates every split and written dataset passes, the dataset store naming each by its manifest, and the seeded sample of earlier records replayed | core, store |
| `splinter-orchestrator` | application | what a run is carried out with: `Config` (the only reader of the environment), the runtime a process shares and the context one command works in (its own pin of each policy alias), model references resolved under the configuration and the network opt-in, the roles models play and who plays each, the recorded run and its cross-process cancel, the stage engine that runs a pipeline's stages in order over one state (the engine checks for a cancel and the budget, skips a stage that does not apply, stops where a stage says to, and records every stage with its time and its failure), the release store (immutable releases, aliases moved by compare-and-set) and the answers `ask` gave, and the error every command reports | agent, core, data, eval, knowledge, model, sandbox, store |
| `splinter-pipelines` | workflow | the stages and the pipeline: the router (a sentence read by an `IntentClassifier` - by default the router-role model through sven's typed call - and routed by code), and each stage as a command - sources, tasks, solve, verify, critique, dataset, rehearse (the base's own answers to general tasks, never an anchor task), train (from the champion, with replay and the rehearsal mixed in at their shares and monitored on), release (the four-check gate, immutable releases, aliases, rollback), eval, reserve (the exam's families taken out of the sources up front) and the exam set and powered exam built on it - with `learn` composing them into one recorded run; and the curriculum: pass@k with a teacher solving open-book what the policy never solves, the tasks worth training on (failed at least sometimes, with a verified answer), concept mastery across releases from closed-book solves, concepts a failed retention check queues for new tasks, and diversity quotas on a training set | agent, core, data, eval, knowledge, model, orchestrator, sandbox, store |
| `splinter-sdk` | sdk | the one way to embed Splinter: a `Splinter` handle that owns the runtime, the trainer and the progress receiver, hands each command a context of its own and runs it as a recorded run, with the commands and the layers a caller works with re-exported under stable names | agent, core, data, eval, knowledge, model, orchestrator, pipelines, sandbox, store |
| `splinter` | surface | the binary: `clap` grammar and output, nothing else | sdk |

Beside the crates: `samples/<name>` holds one controlled learning
sample per directory (a standalone program with a README stating what
was measured), `tasks/<family>` the frozen task families (workspace, hidden
world, verifier, audit), and `scripts/` the local override, lock pinning,
hooks and gates.

### Where does it go

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
| A controlled learning experiment | `samples/<name>`, on the SDK alone |

### Laws

- **A mechanism never decides when it is used.** A verifier is eval's; which
  verifier grades a kind is a pipeline's. A trainer is the model adapter's;
  choosing supervised or preference training is a pipeline's. A typed call is
  the agent's; asking it is a stage's.
- **No learning algorithm in the CLI, the SDK or the orchestrator.** They
  compose and present; algorithms live in the project that owns the
  computation, and the decision to use one lives in a pipeline.
- **A model is asked for by role.** No stage names a model; it names a role
  (`policy`, `teacher`, `generator`, `planner`, `judge`, `critic`, `router`)
  and configuration decides who plays it. No code that generates tasks calls
  brain.
- **An experience is immutable.** Verdicts, rewards, step labels and dataset
  membership are facts derived about it. Never rewrite an experience to
  encode a later interpretation: add an annotation, or project a new view.
- **No crate per training algorithm, and no experience format per
  pipeline.** Experience is collected once and reinterpreted; algorithms are
  brain's.
- **The stage engine records, stages do not.** A stage does its work and
  reports what it did; cancel, budget, skipping, timing and the run record
  are the engine's.

### Two questions before adding anything

1. *Could a completely different learning pipeline use this?* If yes, it
   belongs below the pipelines. If no, it belongs in the pipeline.
2. *Would sven or brain still make sense without Splinter if this lived
   there?* If moving it upstream keeps them whole, it probably belongs there.

### Changing one thing touches several

- **A training objective:** the objective and its line format (`data`); what
  the backend can train from a file (`model` capabilities and its dataset
  check); the stage that picks the objective (`pipelines`); the SDK's
  exposure and the docs.
- **A task kind:** the kind catalogue and its admission rules (`knowledge`),
  the verifier that grades it and the kinds' names (`eval`, `core`), a
  fixture, an evaluation suite, the docs.
- **A model role:** the role and its fallback rule (`core::role`), the
  configuration that feeds it (`orchestrator`), the stage that asks for it,
  `status` output, the docs.
- **A pipeline:** its stages and order (`pipelines`, on the stage engine),
  its request and report, the SDK method, the CLI verb if it has one, a
  scripted integration spec, the README's capability table.
- **A field of an experience:** the vocabulary (`core`), its canonical form
  and golden test (`store/tests/identity.rs`), ingestion, every projection
  that reads it, and a note on compatibility with what is already stored: an
  address must not change for state that is already there.


## Quality rules

Each is enforced by a gate in `make check` and at commit where it can be:

- **Layering** - dependencies point down the tiers, forbidden pairs stay
  unreachable at any depth, an exception that no longer applies is removed,
  a declared dependency is used, only `splinter-model` depends on brain, only
  the two adapters depend on sven, and the command line and every sample
  depend on the SDK and nothing else (`check-architecture`, whose own
  specification is `scripts/gates/test-architecture.py`).
- **Configuration is a value** - only `splinter-orchestrator`'s `config` module
  reads or writes the environment; everything below takes its settings as
  arguments, and tests build their own `StateRoot` instead of mutating
  process state (`check-env-reads`).
- **Small files** - no source file over 800 lines; a file that must stay
  larger for a while is recorded in `architecture.toml` and may only shrink
  (`check-architecture`).
- **Documented** - every public item of a library crate has a doc comment
  (`#![warn(missing_docs)]` under clippy `-D warnings`).
- **No numbers without reproduction** - no unreviewed performance figure in
  a README, comment or string (`check-no-perf-numbers`).
- **Scripts that work** - every script parses and is referenced
  (`check-scripts`).
- **One definition per default** - a default (a model, a limit, a step
  count) is a named constant in the crate that owns it; the CLI shows it,
  it does not restate it.
- **Errors carry context** - a failure names what failed and on which input;
  `unwrap`/`expect` outside tests only for an invariant the code states.

## Commands

```bash
make hooks/install   # once per clone: pre-commit, commit-msg and pre-push hooks
make local           # build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR)
make lock            # pin Cargo.lock to those checkouts' HEADs
make build test      # release profile throughout
make check           # every gate: hygiene, headers, scope, layering, env reads, perf numbers,
                     # scripts, fmt, clippy -D warnings, lock sources, history
```

Never run `cargo fmt --all`: it follows the local path overrides into the
sven and brain checkouts. `make fmt` formats Splinter's own files only.
