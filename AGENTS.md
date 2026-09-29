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

## Layout

| Path | What |
|---|---|
| `crates/splinter` | the `splinter` binary: runs, traces, fact extraction, training, promotion, the in-process brain model provider |
| `crates/lab` | `splinter-lab`: the measurement harness every experiment shares - served-model identity, verdicts that cannot be self-awarded, wire capture, dataset derivation |
| `experiments/<name>` | one controlled learning experiment per directory, each with a README stating what was measured |
| `tasks/<family>` | frozen task families: workspace, hidden world, verifier, audit |
| `scripts/` | local-override and lock pinning, hooks, gates |

## Commands

```bash
make hooks/install   # once per clone: pre-commit, commit-msg and pre-push hooks
make local           # build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR, TARGET_DIR)
make lock            # pin Cargo.lock to those checkouts' HEADs
make build test      # release profile throughout
make check           # every gate: hygiene, headers, fmt, clippy -D warnings, lock sources, history
```

Never run `cargo fmt --all`: it follows the local path overrides into the
sven and brain checkouts. `make fmt` formats Splinter's own files only.
