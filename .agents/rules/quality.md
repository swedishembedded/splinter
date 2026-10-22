<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Quality rules

Read this before you write code, a test, a measurement or a commit. Each rule
names the gate that enforces it; `make check` runs every gate over the tree and
the hooks (`make hooks/install`) run them per commit. Each gate script's header
under `scripts/gates/` says what it enforces and why, and
`docs/contributing/gates.md` lists them.

## Evidence

- **Evidence over claims.** A learning run is judged by a measurement the
  policy under training cannot influence: held-out probes scored against source
  material, retention of everything learned before, and an anchor suite for
  general behaviour. "The model said it learned" is not evidence; neither is a
  training loss.
- **Unmeasured is not zero.** A number that was not measured is absent, never
  reported as `0`.
- **No numbers without reproduction.** No unreviewed performance figure in a
  README, comment or string; a reviewed one carries `perf-number: <reason>`
  (`check/gates/no-perf-numbers`).
- **No fixed default seeds.** A literal seed in production code is a salt, a
  world constant, a frozen artifact's seed, a greedy path's or a test double's,
  and is listed with that reason in `scripts/gates/seed-literals.allow.d/`;
  every other default seed is drawn, printed and recorded. A seed literal in a
  test says why determinism is necessary with `// seed: <reason>` (`# seed:` in
  Python), else it is dropped and the test passes for any seed
  (`check/gates/seed-literals`).
- **A defect that cost a measurement is written down** as
  `.agents/knowledge/<NNN>-<slug>.md` with the number that proved it, and
  listed in `.agents/knowledge/index.md`, in the same change.

## Tests

- **No test skips itself silently.** A test that cannot run (no git, no docker
  daemon, no weights, running as root, an unset environment variable) fails
  with the reason, or carries `#[ignore = "reason"]` so every run shows it as
  ignored. It never prints "skipped" and returns, which reports a pass for a
  check that never happened. A skip is allowed only when the person asking for
  the change explicitly asked for that test to be skippable, and the test says
  so where it skips. This holds for every crate and sample, and for ignored
  hardware tests: run with `--ignored`, a missing prerequisite fails them
  (`check/gates/test-skips`).
- **Tests build their own state.** A test builds its own `StateRoot` instead of
  mutating process state (`check/gates/env-reads`).

## Code

- **Zero warnings.** `cargo clippy -D warnings` is clean, including warnings
  the change did not cause (`check/clippy`; the commit hook lints the packages
  that own the staged files).
- **Documented.** Every public item of a library crate has a doc comment
  (`#![warn(missing_docs)]` under clippy `-D warnings`).
- **Small files.** No Rust source file over the `max_file_lines` limit in
  `architecture.toml`; a file that must stay larger for a while is recorded
  there and may only shrink (`check/gates/architecture`).
- **One definition per default.** A default (a model, a limit, a step count) is
  a named constant in the crate that owns it; the CLI shows it, it does not
  restate it.
- **Errors carry context.** A failure names what failed and on which input;
  `unwrap`/`expect` outside tests only for an invariant the code states.
- **Code never cites a `docs/` or `.agents/` path.** State the fact inline; a
  test that reads a page to hold it to the program marks the line with
  `doc-reference: <why>` (`check/gates/no-doc-citations`).
- **Scripts that work.** Every script parses and is referenced from somewhere
  else in the repository (`check/gates/scripts`).
- **Formatting.** `make fmt` formats Splinter's own files only. Never run
  `cargo fmt --all`: it follows the local path overrides into the sven and
  brain checkouts.

## Model work on GPUs

- **Batched, on every card.** Inference over many prompts (exams, gates,
  sampling, generation) keeps the server's decode slots full by asking as many
  requests at once as it has slots, never one at a time; training runs with the
  largest micro-batch the card holds. Independent work is split across every
  available GPU. A sequential path exists only where the order of answers is
  the thing measured, and it says so.
- **Profiled before it runs long.** A training run or a large inference job is
  first measured on the hardware it will run on: where its time goes (prefill,
  decode, sampling, data movement, the host), how full the card is, and how
  throughput scales with batch. A slow path is fixed (here, or upstream in
  brain) before the long run, not after it.

## Text and history

- **Headers.** Every source file starts with the SPDX line and the copyright
  line; Markdown uses the CC-BY-4.0 comment form (`check/gates/spdx`;
  `python3 scripts/spdx/check.py --fix <file>` inserts them).
- **ASCII punctuation.** No em or en dashes, curly quotes or ellipsis
  characters; the text-hygiene hook repairs them (`check/gates/text-hygiene`).
- **Linear history.** Rebase, never merge; the pre-push hook and
  `check/gates/linear-history` refuse a merge commit.
- **No trailers.** `Co-Authored-By:` and `Claude-Session:` lines are stripped by
  the commit-msg hook and refused at push.
