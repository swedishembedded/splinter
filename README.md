<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Splinter

A learning agent with its own model. Tell it what to learn, in plain
language; it learns it, measures that it did, and releases a better version
of itself.

```
splinter "Learn all the facts about the STM32 reference manual from <path-to-manual>"
splinter "Learn what we can do with brain command line options"
splinter "Explain brain command line flags without interacting with brain"
```

Splinter is built on two standalone projects and replaces neither:

- **sven** runs the agent: tools, sessions, delegation, and a trustworthy
  record of what the agent did.
- **brain** runs the model: inference, training, evaluation and serving.

What Splinter releases is an ordinary brain adapter with a manifest. It runs
on plain `brain serve`, and plain sven can use it through its brain
provider; neither needs Splinter installed.

## Status

Early. The command line (`crates/splinter/README.md`) runs the learning
pipeline one stage per verb - sources, tasks, solve, verify, critique,
dataset, train, release - and `learn` runs them all as one recorded run:

```bash
splinter eval --suite anchor --freeze general.jsonl   # once: the anchor suite
splinter learn docs/manual.md --goal "the console and power limits"
splinter status
splinter "what baud rate does the console run at?"
```

Every stage stores what it makes under the state root by content address,
so any stage can be rerun or inspected alone (`splinter runs show`,
`splinter experiences show --graph`). Every model runs locally unless a
`remote:` model is named with `--allow-remote`.

A trained candidate continues the current release (the champion) with a
replay of what earlier releases learned, and becomes the policy only if
the release gate measures that it improved on the new material's held-out
questions, kept what earlier releases learned, held a frozen anchor suite
of general tasks, and runs on plain `brain serve` with the same answers.
Each release is an immutable adapter with a manifest of every number the
gate measured; `splinter rollback default` returns to the previous one.

### Not yet

- `lineage` - tracing an answer back to its sources.

## Building

sven and brain are git dependencies at the revisions pinned in `Cargo.lock`:

```bash
make build     # release build of every package
make test      # every test; no model, GPU or network needed
make check     # all gates: text hygiene, headers, fmt, clippy -D warnings, lock sources
```

To build against local checkouts instead - to change sven or brain and
Splinter together - write the gitignored local override once:

```bash
make local SVEN_DIR=<sven-checkout> BRAIN_DIR=<brain-checkout>
```

Builds then compile the checkouts' working trees, offline, without touching
`Cargo.toml` or `Cargo.lock`. `make lock` re-pins the lock to those
checkouts' HEADs; push those commits before sharing the lock. Delete
`.cargo/config.toml` to go back to the remotes.

Runtime state (sources, tasks, experiences, datasets, candidates,
releases, runs) lives under `~/.sven/splinter`, Sven's home in a namespace
of its own (override: `--state DIR` or `SPLINTER_STATE`).

## License

Apache-2.0. See `LICENSE`.
