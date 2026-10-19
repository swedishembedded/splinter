<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Configuration

A run needs no flags. Configuration names what a machine knows, through the environment and a few
global command-line options. Every command takes the options listed under [global
options](../reference/cli.md#global-options).

## State

Runtime state (sources, tasks, experiences, datasets, candidates, releases, answers, runs) lives under
`~/.splinter`. Override it with `--state DIR` or `SPLINTER_STATE`. State left in `~/.sven/splinter` by an
earlier version is still used until `~/.splinter` exists; move it there to adopt the new location. Layout and
maintenance are in [Storage](../reference/storage.md).

## Environment

| Variable | Meaning |
|---|---|
| `SPLINTER_STATE` | The state root (default `~/.splinter`). |
| `BRAIN_MODELS_DIR` | Brain's model store (default `~/.local/share/brain/models`). |
| `BRAIN_QWEN_WEIGHTS` | The policy's base checkpoint (default `Qwen/Qwen3-0.6B` in the model store). |
| `SPLINTER_BRAIN_BIN` | The `brain` binary a release is served on (default: the first `brain` on `PATH`). |
| `SPLINTER_ASSISTANT_MODEL` | The stronger model that plans, writes tasks and teaches when the command names none. |
| `SPLINTER_JUDGE_MODEL` | The model that judges what only a judge can decide (default: the assistant). |
| `SPLINTER_FRONT_DOOR_MODEL` | A model that reads sentences, in place of the policy; a larger model reads them better. |
| `SPLINTER_BUDGET` | How long a `learn` may take when it names no `--budget` (a sentence names none). |
| `SPLINTER_THINKING` | `1` lets local models reason before they answer. Off, a reasoning model is asked with its reasoning block closed, so every reply is the answer and comes at once; training renders its answers the same way. |
| `SPLINTER_BF16_BASE` | `1` holds a large policy's base at bf16 so it trains on one card (the flag is `--bf16-base`). |
| `SPLINTER_MIN_CALIBRATION_CONTROLS` | The fewest controls a judge is measured on before its verdicts count (default 16). |
| `SPLINTER_ALLOW_REMOTE` | `1` or `true` permits models reached over the network (the flag is `--allow-remote`). |
| `AGENT_OPENROUTER_KEY`, `BRAIN_API_KEY` | Keys for OpenRouter models and for any other provider. |
| `SPLINTER_DOCKER`, `SPLINTER_WORKSPACE_IMAGE` | A docker client and a digest-pinned image: an agent's commands then run in a container with no network. |
| `SPLINTER_WORKSPACE_SCRATCH` | Where trees are laid out for repository tasks. |
| `SPLINTER_CARGO_HOME` | A cargo home the commands of a Cargo project build offline from. |

## Model references



Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`BRAIN_QWEN_WEIGHTS`, else `Qwen/Qwen3-0.6B` in brain's model store) with the champion's adapter, the release `default` points at; the base alone before any release |
| `policy:<alias>` | the base with the adapter of the release `<alias>` points at (`release --alias`) |
| `local:<checkpoint>[+<adapter>][@<tokens>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+`. The context is the largest the checkpoint supports (its `max_position_embeddings`); `@<tokens>` limits it, as fitting a large model on a card requires (a 14B judge on a 24 GiB card: `local:Qwen/Qwen3-14B@4096`) |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A policy alias is resolved once per command, when it is first used: a
run keeps the release it started with however the alias moves meanwhile,
and records it (`learn`'s first stage, a candidate's `parent`).

The policy's context is limited with the global `--policy-context-tokens N`
(a policy too large for the card at its maximum), any other local model's by
its reference.

A remote reference is refused unless `--allow-remote` is given or
`SPLINTER_ALLOW_REMOTE=1` is set; that is the only way Splinter uses the
network. OpenRouter models take their key from `AGENT_OPENROUTER_KEY`, any
other provider from `BRAIN_API_KEY`.

## Roles

Stages ask for a model by role, never by name: `policy` (the model being trained), `teacher`,
`generator`, `planner`, `judge`, `critic` and `router`. With nothing configured the policy plays every
role; `--generator`, `--teacher`, `--judge` and `--planner` and the variables above name others.
`splinter doctor` checks that the model each role would get resolves to weights on disk.

## Request limits

`--remote-concurrency` is how many requests to a model reached over an API may be in flight at once (4 by
default); a model on the local device is asked one at a time. `--temperature` is the sampling temperature (0 to
2) asked of a model reached through sven's providers, the provider's own default when not given.
`--solve-deadline` is how many seconds one task's solve may take on the wall clock (300 by default); raise it
when many tasks share one served model, so a task is not failed for waiting in the queue.
