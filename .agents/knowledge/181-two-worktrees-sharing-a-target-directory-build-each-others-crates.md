<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 181. Two worktrees sharing a target directory build each other's crates

**Lesson:** two checkouts of this workspace that share one `CARGO_TARGET_DIR` do not get separate artifacts for
Splinter's own crates. A build in one checkout can link the other checkout's compiled crate and call it fresh, so a
clean `cargo clippy` or `cargo test` proves nothing about the sources in front of you. Before a gate run in a shared
target directory, remove the workspace crates' artifacts (`cargo clean --release -p splinter-core -p ... -p splinter`,
third-party crates stay) or touch the crates you changed; better, give each concurrently active worktree its own
target directory when the disk allows it.

Found 2026-11-19 (lane H), while lane Q built its worktree into the same `containers/targets/splinter`. A
`cargo test --release -p splinter-pipelines` in lane H's worktree failed with two errors its sources could not
produce: `E0063 missing fields briefing and role in initializer of ForkAt` at `crates/pipelines/src/bench/repair/fork.rs:125`
(lane Q had added those fields to `splinter-agent` in its own worktree) and `E0599 no method named adopt found for
DatasetStore` (the method was in lane H's `crates/data/src/store.rs`; the `splinter-data` linked was another
checkout's). A `cargo clippy` of the same package minutes earlier had passed. `cargo clean --release` of the 14
workspace packages (734 files, 933 MiB) made the next build compile every Splinter crate from lane H's sources, and
both errors were gone. Lane T saw the same with a stale `splinter-data` the day before and fixed it by touching the
changed crate.

What is consistent with both observations: cargo names a path crate's outputs and records its dep-info relative to
the workspace root, so the two worktrees' `splinter-agent` share one output name, and freshness is judged from the
mtimes of the sources in the checkout that builds, which can be older than the other checkout's output.
