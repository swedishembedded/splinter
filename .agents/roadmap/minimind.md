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

## Open

- The rest of the sample and its generic pieces (agent episode under a small-context profile,
  wire records, brain from-scratch calls, bench examination) land in following commits.
