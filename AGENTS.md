# AGENTS.md - Splinter

Splinter is the end-to-end learning system built on top of **sven** (agent
execution) and **brain** (inference and training). You tell it, in plain
language, what to learn - a document, a command-line tool, a task - and it
acquires that capability, measures that it did, and releases a better
version of its own model. The released model is then deployable without
Splinter: on plain `brain serve`, or through sven talking to brain.

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
| User-facing documentation | `README.md` |

Code never cites a `docs/` or `.agents/` path (a gate enforces it): state the
fact inline instead.

## Commands

```bash
make hooks/install   # once per clone: pre-commit, commit-msg and pre-push hooks
make check           # every gate over the whole tree
```
