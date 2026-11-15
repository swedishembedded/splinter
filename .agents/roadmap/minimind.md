<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# minimind: random weights to a small agent, in one resumable command

Scope: `samples/minimind`, one idempotent, resumable command that trains a tiny model from random
weights to an agent solving elementary workspace tool-use tasks: synthetic world -> tokenizer and
random-init checkpoint -> pretraining -> probes -> expert demonstrations through real sven -> full
chat fine-tune -> the model's own verified episodes -> expert iteration -> bench's sealed paired
verdict -> report. Brain is reached only over its HTTP API; sven runs the agent against an
OpenAI-compatible endpoint; bench examines. The sample keeps only what is specific to the demo.

Decisions (2026-11-15): architecture is configuration (`models.<label>.arch` plus its own config,
handed to brain unread), so the same sequence runs on several architectures over shared data
stages; Ouro first. Configuration is YAML (`minimind.yaml`). No downloads by default; a fetched
corpus is a pinned, checksummed, ledger stage. Demonstrations are the exact requests sven sends,
captured on the wire, and only verifier-accepted episodes become records.

## State

| Piece | Where | State |
|---|---|---|
| Stage ledger (key = hash of config slice, inputs, revisions; begin/commit/invalidate pack; attempt ids never reused; outputs re-hashed on reuse; one process per run dir) | `crates/store/src/stage_ledger.rs`, shared pack lock `pack.rs` | done, specs in `crates/store/tests/it/stage_ledger.rs` |
| Agent episode under sven's compact profile (`compact/1`, three tools, first request under 4 KB measured by its spec) and the policy endpoint (an OpenAI-compatible endpoint answered by a program, every request kept; also a recording proxy to a served model) | `crates/agent/src/workspace_episode.rs`, `policy_endpoint.rs` | done; sven pinned to 6f4b1716 |

## Open

- sven: before the compact profile, an embedded engine with `agent.system_prompt` set still sent
  sven's own 25 KB system prompt, including skills discovered in the host user's home directory
  (measured 2026-11-15, 25535 bytes); the compact profile replaces it. Whether the default
  profile of an embedded engine should read the host's skills is sven's to decide.
- `sven_sdk::rollout` does not exist on sven main; episodes are collected by splinter through
  `crates/agent` with durable attempt ids from the stage ledger.

- The rest of the sample and its generic pieces (agent episode under a small-context profile,
  wire records, brain from-scratch calls, bench examination) land in following commits.
