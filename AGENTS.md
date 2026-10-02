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
- Experiment design and measurement: verifiers the policy cannot reach,
  held-out, retention and anchor suites, release gates.
- Releases: immutable adapters with manifests, the default alias, lineage
  from an answer back to its sources.
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
| **Splinter** | The learning campaign: intent, task acquisition, evidence admission, dataset curation, experiment scheduling, lineage, release decisions | Re-implementations of anything sven or brain provide |

Neither sven nor brain depends on Splinter. Splinter consumes both through
their public SDKs (`sven-sdk`, `brain`); every import of a brain or sven
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
| User-facing documentation | `README.md` |

Code never cites a `docs/` or `.agents/` path (a gate enforces it): state the
fact inline instead.

## Architecture

Ten crates in layers; a crate depends only on the ones below it.
`architecture.toml` is the authoritative edge list and
`scripts/gates/check-architecture.sh` enforces it against `cargo metadata`.

| Crate | Owns | Depends on |
|---|---|---|
| `splinter-expdb` | the experience graph database, with no internal dependencies; `splinter-record` stores experiences in it: immutable content-addressed segments, blob packs and index runs on plain files; manifests as the only transaction layer; per-writer ids so ingest needs no coordination or file locks; the graph of tasks, states, decisions, transitions, counterfactual sets and episode families; modality-neutral episodes of streams on explicit clock domains with events, actions and correspondences; evaluations, derivations and skills kept apart from raw experience; pinned snapshots, compaction and garbage collection, whose storage protocol is model-checked in TLA+; queries and training views (imitation, preference, group-relative, transitions, step labels, world-model windows, contrastive pairs, masked prediction, action chunks) as reproducible projections | - |
| `splinter-record` | durable state: the injected `StateRoot` and its layout, atomic and write-once writes, time-ordered ids, the UTC clock, the digests (blake3 content addresses; sha256 for what a tool reports), the run record every command that writes pipeline state keeps (stages, status, outputs, the cross-process cancel request, kept as events and signals in the experience database); the source store - sources and their parts' content stored once per digest, spans resolved to exact bytes; the task store - generated tasks under their address and named task sets recording how each was generated; and the experience store, kept in the experience database through one shared `Workspace`: immutable content-addressed experiences (each also an attempt in the graph), annotations as ranked evaluations of it, experience sets, the derived reward; the lineage of datasets, training runs and releases back to the attempts; maintenance of the database; where releases and frozen suites live | splinter-expdb |
| `splinter-sandbox` | where a solver's code runs: the one bounded process runner (explicit environment, process-group kill at a timeout, output caps, rlimits), the runtime registry and what a runtime resolves to (version, executable digest), the process sandbox (bounded, not isolation) and the container sandbox (image pinned by digest, no network, read-only root), and environments whose snapshot digest pins everything that determines their behaviour | store |
| `splinter-lab` | measurement: the holdout rule, verdicts that cannot be self-awarded, served-model identity, wire capture, dataset derivation from verified episodes, task families, pass@k and where a pass rate and a teacher's verified answer put a task (always, frontier, taught, never), the one system prompt every model run on a task answers under and every chat record shows, and verifiers by strength (executable checks run in the task's sandbox, mutation-validated generated tests, formal exact match, agreement between answers, judge calibration) with the composite that annotates all their verdicts, a critique verified by the outcome of the retry it led to, and two models compared on the items both were graded on | store, sandbox |
| `splinter-policy` | the model being trained, and the only crate that touches brain: the in-process provider behind sven's `ModelProvider`, `ModelSelection` and its identity, the residency (one resident copy per base checkpoint, shared by every model on it, each model's adapter attached before its generation, released for training and serving), the model loaded for a command (quiescing the device when dropped), LoRA fine-tuning with replay, continuation and held-out scoring, DPO preference fine-tuning scored by brain's preference score, and brain's paired sign test | lab |
| `splinter-agent` | a task solved through sven in exactly the environment it records (closed-book with no tools, or a runtime with the one tool `run_code`), bounded by sven's run options, under Splinter's system prompt in place of sven's; the open-book prompt a teacher's solve and `ask --open-book` are sent; the judge verifier, a different model grading closed-book through that same solve; the critic, a model saying closed-book what is wrong with a failed attempt from its evidence summaries, never the teacher's material; the retry with that critique, and the bounded critique-and-retry loop that records the chain as relations | store, sandbox, lab (views in its specs only) |
| `splinter-knowledge` | intake: capturing documents, repository trees and command runs as sources, addressable sections, the text gates a generated task is held to, the concepts a task exercises (declared, else the source sections its evidence falls in, else its kind), the grounding material a teacher is shown of a task (the sections its evidence falls in, its passages and hints, never its reference), the denoise task generator, and model task generators: the task kind catalogue as data, and a generator model's proposals admitted by code (grounded, self-contained, naming their subject, executed where computed, deduplicated, never contradicting one another) | store, sandbox, lab, policy, agent, views |
| `splinter-views` | projections of the experience store into training records for one objective (final answers, steps, critiques, preferences, verdicts, decisions, retrieval, outcomes, denoise, raw text), every conversation starting with the system turn every solve runs under, what a student may see of privileged material (a teacher's solve included), the dataset each objective is written as with its manifest, the dataset store naming each by its manifest, and the seeded sample of earlier records replayed | store, lab, policy |
| `splinter-campaign` | the controller: `Config` (the only reader of the environment), the context a command works in, model references (`policy:`, `local:`, `remote:`) and the network opt-in, the front door (a sentence classified by the policy through sven's typed call, routed by code), and each stage as a command - sources, tasks, solve, verify, critique, dataset, train (from the champion, with replay), release (the four-check gate, immutable releases, aliases, rollback) and eval - with `learn` composing them into one recorded run; and the curriculum: pass@k with a teacher solving open-book what the policy never solves, the tasks worth training on (failed at least sometimes, with a verified answer), concept mastery across releases from closed-book solves, concepts a failed retention check queues for new tasks, and diversity quotas on a training set | all of the above |
| `splinter` | the binary: `clap` grammar and output, nothing else | campaign and the types it prints |

Beside the crates: `experiments/<name>` holds one controlled learning
experiment per directory (a standalone program with a README stating what
was measured), `tasks/<family>` the frozen task families (workspace, hidden
world, verifier, audit), and `scripts/` the local override, lock pinning,
hooks and gates.

## Quality rules

Each is enforced by a gate in `make check` and at commit where it can be:

- **Layering** - edges only as `architecture.toml` lists; only
  `splinter-policy` depends on brain (`check-architecture`).
- **Configuration is a value** - only `splinter-campaign`'s `config` module
  reads or writes the environment; everything below takes its settings as
  arguments, and tests build their own `StateRoot` instead of mutating
  process state (`check-env-reads`).
- **Small files** - no source file over 800 lines (`check-architecture`).
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
