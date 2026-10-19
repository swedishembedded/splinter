<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# splinter - the command line

The `splinter` binary: the `clap` grammar and the output of every command, and nothing else. It
depends on `splinter-sdk` only; the pipelines, the verifiers and the release gate are in the crates below it.

One verb per pipeline stage. The thing a verb acts on is a positional argument; every stage writes what it makes
under the state root by content address, so any stage can be rerun or inspected alone; `--json` works on every
command. `splinter <command> --help` prints the reference for one command, and
[the command reference](https://swedishembedded.com/docs/splinter/reference/cli) lists all of them, generated from the
grammar in `src/cli.rs` and `src/cli/` so it cannot disagree with the program.

```bash
splinter doctor                         # can a run start here?
splinter learn ./manual.md              # sources to tasks to a candidate, gated, as one recorded run
splinter "learn the manual in ./docs"   # the front door: a sentence becomes one of the commands
splinter ask "What does the manual say about the watchdog?"
```

| To do this | Read |
|---|---|
| Learn a document, a directory or a command's output | [Learn from documents](https://swedishembedded.com/docs/splinter/guides/documents) |
| Learn a repository from its own tests | [Learn a repository](https://swedishembedded.com/docs/splinter/guides/repositories) |
| Learn from your own agent sessions | [Agent sessions](https://swedishembedded.com/docs/splinter/guides/agent-sessions) |
| Build datasets and train | [Training](https://swedishembedded.com/docs/splinter/guides/training) |
| Evaluate, release, serve, roll back | [Evaluation](https://swedishembedded.com/docs/splinter/guides/evaluation) and [Releases](https://swedishembedded.com/docs/splinter/guides/releases) |
| Output formats and exit status | [Output and exit status](https://swedishembedded.com/docs/splinter/reference/output) |
| State, archive and recovery | [Storage](https://swedishembedded.com/docs/splinter/reference/storage) |

The command reference is regenerated with `splinter cli-reference` (a hidden command); `--check FILE` fails when a
committed copy is stale and `--check-invocations FILE...` fails on a `splinter` command in a Markdown shell block that
the grammar rejects. `make docs/check` runs both.
