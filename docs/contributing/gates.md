<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Gates

`make check` runs every gate over the whole tree: `check/gates`, `check/fmt`, `check/clippy` and `check/papers`.
The same gates run per commit in the hooks, so a clone without hooks installed is still held to them. Each gate is one
script, `scripts/gates/check-<name>.sh`, and one target, `make check/gates/<name>`; the script's header says what it
enforces and why.

| Gate | Enforces |
|---|---|
| `text-hygiene` | ASCII punctuation, whitespace, line endings, parseable files |
| `spdx` | Every tracked file carries its SPDX and copyright header (Markdown uses CC-BY-4.0 comments) |
| `architecture` | `architecture.toml` against the real dependency graph |
| `env-reads` | Only the orchestrator's config module reads the environment |
| `large-files` | No model artifacts or oversized files in git |
| `linear-history` | The working branch's history is linear |
| `lock-sources` | `Cargo.lock` pins sven and brain to their remotes |
| `make-targets` | Target naming and self-documentation |
| `manifests` | Every Cargo manifest parses |
| `no-doc-citations` | Code does not cite a `docs/` or `.agents/` file path; state the fact inline or link by URL |
| `no-machine-paths` | No absolute machine paths |
| `no-perf-numbers` | No unreviewed performance numbers in READMEs, comments or strings; mark a reviewed one with `perf-number: reason` |
| `repo-scope` | Splinter names only sven and brain |
| `scripts` | Every script works and is used |
| `seed-literals` | Random choices take a seed the run states |
| `test-skips` | No test skips itself silently |

Clippy runs with warnings denied, including warnings the change did not cause. Never disable a check, weaken an
assertion or delete a failing test to make a change pass; fix the cause. A gate that lied is recorded with the number that
proved it in the repository's knowledge notes.
