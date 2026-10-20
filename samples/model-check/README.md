<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# model-check - does knowing a model checker help an agent fix concurrency bugs?

**Status: the checking harness exists and checks itself (`make samples/model-check/audit`, no model, no
GPU, no network). One family (a bounded queue) is complete; the agent phases below are not built.**

## The question

An agent that has been taught what a model checker is, how to run it and how to read a counterexample
should solve more unseen bug-fix tasks on small concurrent state machines than the same agent without
that knowledge, with the checker available to both. The full registered design (phases P0-P3, arms,
decision rule) is kept with the campaign plan; this sample holds the part that needs no model:
the tasks and the deterministic check that judges them.

## A family

| File (under `specs/<family>/`) | What it is |
|---|---|
| `spec.toml` | names the module, the TLA+ file and configuration, TLC's distinct-state count, and a symptom in words per invariant |
| `Queue.tla`, `Queue.cfg` | the correct specification with small constants; TLC must find no error and exactly the recorded states |
| `queue.lua` | the reference implementation: a pure Lua 5.1 module (`init`, `actions`, `invariants`) over plain data |
| `queue_test.lua` | in-tree unit tests the agent would also see |
| `mutations.toml` | one-line changes (`find`/`replace` on the reference text) and the property each must violate |

## The refinement check

Deterministic, in two independent halves (`src/check.rs`):

1. `lua/explore.lua` enumerates every reachable state breadth first and checks the implementation's
   invariants, which carry the specification's names.
2. `lua/tracegen.lua` writes the explored behaviours as `<Spec>Trace.tla`: every transition ends one
   trace, and the trace specification follows a chosen trace step by step through the specification's
   own `Next`. TLC checks it; a step the specification does not allow leaves a state with no
   successor, which TLC reports as a deadlock (exit 11).

A mutant is admitted as a task when the reference passes both halves and the mutant fails the way its
`violates` field says: an invariant name, `trace` (every invariant holds, only TLC rejects it) or
`coverage` (safe, but reaches fewer states than the specification).

## Commands

```text
make deps/lua LUA_DIR=<dir>                 # Lua 5.1.5 into <dir>/bin/lua
make deps/tlaplus TLAPLUS_DIR=<dir>         # TLC and a JRE, sha256-checked
make samples/model-check/audit  LUA=<dir>/bin/lua TLAPLUS_DIR=<dir>
make samples/model-check/report LUA=<dir>/bin/lua TLAPLUS_DIR=<dir>
```

`audit` runs TLC on every specification, the explorer and TLC on every reference implementation and
mutant, and the Lua unit tests, and prints one line per check. The Lua tests also run alone:
`lua5.1 samples/model-check/lua/tests/explore_test.lua` (and `tracegen_test.lua`).

## Results

Measured on one aarch64 machine, 56 cores visible, TLC 2.19 on Temurin 21
(audit: 12 checks, 0 failed, three runs in a row).

Queue specification (`Cap = 2`, producers `{1, 2}`, 2 items each): TLC 53 states generated, 35
distinct; the reference implementation reaches the same 35 states with 52 transitions.

| mutant | declared | explorer alone | TLC on the trace module | states |
|---|---|---|---|---|
| boundary-capacity | Bounded | Bounded violated | rejected | 47 |
| dropped-receive-count | Conservation | Conservation violated | rejected | 35 |
| wrong-producer-counted | Conservation | Conservation violated | rejected | 35 |
| lifo-dequeue | PerProducerFifo | PerProducerFifo violated | rejected | 45 |
| item-number-from-zero | PerProducerFifo | PerProducerFifo violated | rejected | 35 |
| overtake-other-producer | trace | no violation | rejected | 35 |
| one-item-fewer | coverage | no violation | accepted | 10 |

"Rejected" is the deadlock above: TLC rejects the mutant's behaviours although, for the first five,
the explorer already named an invariant. The sixth mutant holds every invariant and is caught only by
validating its behaviours against the specification, which is what the trace half is for.

## Limits

One family. The agent phases (base, adapter, knowledge step, final comparison) are not implemented, and
no task statements or hidden checks for an agent exist yet. Components were drafted by a local 27B model
driven by sven and reviewed and corrected line by line; the review verdicts are kept outside the repository.
