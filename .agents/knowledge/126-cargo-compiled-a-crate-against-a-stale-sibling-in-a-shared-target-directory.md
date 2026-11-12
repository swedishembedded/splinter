<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 126. Cargo compiled a crate against a stale sibling in a shared target directory

Two worktrees built into one `CARGO_TARGET_DIR`. After new modules were added to `splinter-knowledge` in a worktree, `cargo test -p splinter-pipelines` compiled the pipelines against the old knowledge artifact and failed with seven `unresolved import splinter_knowledge::goal::source` errors, although `cargo build -p splinter-knowledge` of the new sources succeeded. Touching `crates/knowledge/src/lib.rs` and the changed files forced the rebuild and the errors went. Rule: when a dependency's new items "do not exist", suspect the artifact before the code (touch its sources), and give each worktree its own target directory when disk allows. Evidence: the failed runs of 2026-11-12 in worktree goal2.
