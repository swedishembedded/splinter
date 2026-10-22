<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Configuration

A run needs no flags. Splinter finds what this machine holds: brain's model
store (the one `brain pull` fills), the `brain` on `PATH`, the GPUs, and the
strongest chat model in the store that fits a card, which then helps the
policy learn (see [Roles](#roles)). Every choice is a flag, valid on every
command; nothing is read from environment variables. The flags are listed
under [global options](../reference/cli.md#global-options).

## State

Runtime state (sources, tasks, experiences, datasets, candidates, releases, answers, runs) lives under
`~/.splinter`. Override it with `--state DIR`. State left in `~/.sven/splinter` by an
earlier version is still used until `~/.splinter` exists; move it there to adopt the new location. Layout and
maintenance are in [Storage](../reference/storage.md).

## Machine and models

| Flag | Meaning |
|---|---|
| `--models-dir DIR` | Brain's model store (default: the store `brain pull` fills, `~/.local/share/brain/models`). |
| `--policy-base CHECKPOINT` | The policy's base checkpoint, a path or a name in the store (default `Qwen/Qwen3-0.6B`). |
| `--brain-bin PATH` | The `brain` binary a release is served on (default: the first `brain` on `PATH`). |
| `--assistant-model REF` | The stronger model that plans, writes tasks, teaches, judges and reads sentences when the command names none (default: discovered, see [Roles](#roles)). |
| `--judge-model REF` | The model that judges what only a judge can decide (default: the assistant). |
| `--front-door-model REF` | The model that reads sentences (default: the assistant). |
| `--default-budget DURATION` | How long a `learn` may take when it names no `--budget` (a sentence names none); unbounded when not given. |
| `--thinking` | Lets local models reason before they answer. Off, a reasoning model is asked with its reasoning block closed, so every reply is the answer and comes at once; training renders its answers the same way. |
| `--min-calibration-controls N` | The fewest controls a judge is measured on before its verdicts count (default 16). |
| `--allow-remote` | Permits models reached over the network. |
| `--docker PATH`, `--workspace-image IMAGE` | A docker client and a digest-pinned image: an agent's commands then run in a container with no network. |
| `--workspace-scratch DIR` | Where trees are laid out for repository tasks. |
| `--cargo-home DIR` | A cargo home the commands of a Cargo project build offline from. |

`--bf16-base` (on `learn`, `practice` and `absorb`) and `train --base-precision` hold a large policy's
base at bf16 so it trains on one card.

## Model references



Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`--policy-base`, else `Qwen/Qwen3-0.6B` in brain's model store) with the champion's adapter, the release `default` points at; the base alone before any release |
| `policy:<alias>` | the base with the adapter of the release `<alias>` points at (`release --alias`) |
| `local:<checkpoint>[+<adapter>][@<tokens>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+`. The context is the largest the checkpoint supports (its `max_position_embeddings`); `@<tokens>` limits it, as fitting a large model on a card requires (a 14B judge on a 24 GiB card: `local:Qwen/Qwen3-14B@4096`) |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A policy alias is resolved once per command, when it is first used: a
run keeps the release it started with however the alias moves meanwhile,
and records it (`learn`'s first stage, a candidate's `parent`).

The policy's context is limited with the global `--policy-context-tokens N`
(a policy too large for the card at its maximum), any other local model's by
its reference.

A remote reference is refused unless `--allow-remote` is given; that is the
only way Splinter uses the network. Its key is the one sven holds for the
provider: store it once with `sven auth set openrouter` (or the provider's id).

## Roles

Stages ask for a model by role, never by name: `policy` (the model being trained), `teacher`,
`generator`, `planner`, `judge`, `critic` and `router`. A role a command does not name
(`--generator`, `--teacher`, `--judge`, `--planner`) is played by the assistant
(`--assistant-model`), the judge by `--judge-model` and the router by `--front-door-model` when they
are named; the policy plays the critic, and every role the assistant would play when there is none.

With no assistant named, Splinter picks one from the model store: the largest chat model that is
larger than the policy and fits this machine's largest card, at the largest context that fits
(for example `local:Qwen/Qwen3-14B@8192` beside a 0.6B policy on a 24 GiB card). A machine with no
GPU gets none: a larger model on the CPU backend is too slow to help. `--assistant-model
policy:default` keeps every role on the policy.

`splinter doctor` checks that the model each role would get resolves to weights on disk, and warns
when the policy is left to teach, judge, plan, write tasks or read sentences for itself: that run
can start, but rarely trains anything.

## Request limits

`--remote-concurrency` is how many requests to a model reached over an API may be in flight at once (4 by
default); a model on the local device is asked one at a time. `--temperature` is the sampling temperature (0 to
2) asked of a model reached through sven's providers, the provider's own default when not given.
`--solve-deadline` is how many seconds one task's solve may take on the wall clock (300 by default); raise it
when many tasks share one served model, so a task is not failed for waiting in the queue.
