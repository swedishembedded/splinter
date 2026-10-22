<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Build

```bash
make build                      # release build of every package
make build/crate/splinter       # one package; `make build/crate` lists them
make deps/local SVEN_DIR=../sven BRAIN_DIR=../brain   # build against local checkouts
make hooks/install              # pre-commit, commit-msg and pre-push hooks, once per clone
make help                       # every target, from the Makefile's own comments
```

Everything builds in the release profile: brain's kernels are unusably slow unoptimised, and one profile means one
set of compiled dependencies. A build goes to the repository's `target/`; `make deps/local` changes only where sven
and brain come from. Targets are named `<concept>/<subconcept>/<action>`, and every prefix is itself a target that lists its
children; a gate enforces the naming.

The dependency layering is enforced: a crate may depend only on a lower tier, only `splinter-agent` and `splinter-model`
depend on sven, and only `splinter-model` depends on brain. Read [architecture](../concepts/architecture.md) and the repository's `architecture.toml` before adding a
crate or a dependency. A source file is limited in size by a gate; split a module before it reaches the limit.

Sources of truth:

- The command-line grammar is `clap` in `crates/splinter`; the command reference is generated from it, never edited.
- The configuration is read in one place, the orchestrator's config module; no other crate reads the environment.
