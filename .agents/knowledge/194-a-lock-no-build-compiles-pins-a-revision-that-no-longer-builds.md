<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 194. A lock that no build compiles pins a revision that no longer builds

**Lesson:** every build, test and gate here runs through `make deps/local`, whose path overrides compile the
local sven and brain checkouts and never the revisions `Cargo.lock` pins. `check/gates/lock-sources` checks only
that the pins name the remotes. So splinter code can follow sven's API for days while the lock still pins a sven
that lacks it, and every gate stays green. A fresh clone, or anyone following the manual, then cannot build. When
splinter starts using a new sven or brain API, push that commit and run `make deps/lock` against it in the same
change. Prove the lock with one build that has no `.cargo/config.toml`.

Found 2026-11-20 by the documentation lane: a clean worktree of origin/main (`27fa148fb`) pinned sven
`0d7d6643` and brain `79687164`. `cargo build -p splinter` failed in `splinter-agent` and `splinter-model` on sven
API they already used: `config::CommandConfinement`, `model::time_notice`, `model::SamplingParams`,
`RecordedState::observed` and `ToolsConfig::launcher`. sven `0d7d6643` was 33 commits behind sven main.

The fix was `make deps/lock` against clean checkouts of sven `1db749d9` and brain `b2f46a22c` (origin main, both
reachable at the URLs the manifest names). With no overrides, the workspace then built in the release profile, and
`make check` and the `splinter-model`, `splinter-agent` and `splinter-pipelines` tests passed. No source change was
needed.

A gate that compares the overrides' HEADs with the pins would catch the drift cheaply. But it would fail every lane
whose local sven has moved ahead, until that lane pins and pushes, and a gate that changes the workflow of lanes
running in parallel is a decision for the owner. Building without overrides proves the pin. It costs a full build,
so it belongs before a release, not in every gate run.
