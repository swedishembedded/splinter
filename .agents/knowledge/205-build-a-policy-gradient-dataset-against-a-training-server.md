<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 205. Build a policy-gradient dataset against a training server

**Lesson:** `splinter dataset build --max-block` measures every record with brain's trainer only when the endpoint
serves training (`brain serve --training --artifact-store`). Against an inference-only server, it falls back to a
character floor. The floor overcounts long records and does not run the trainer's checks, so a record the trainer
refuses stays in the dataset and `train` stops at its first step. The fallback is recorded in `block_measure` as
`floor` with the reason. The text report prints it as a WARNING, but `--json` carries it only in that field. A
pipeline that reads JSON must check `block_measure`, or build against the server that will train.

Seen 2026-11-21 (ledger item 19, round R19), on the same 640 practice trials and the same flags:

| | PG view (`--sampled-only`, max block 32,768) | sft-step view (max block 16,384) |
|---|---|---|
| Inference-only endpoint (floor) | 1,738 records, 269 left out as `over_block` | 1,595 records |
| Training endpoint (trainer measure) | 2,006 records, 1 left out as `over_block` | 1,601 records, 41 left out |
| Refused at train start (floor dataset) | 1 record: recorded ids that do not read back as the message | 6 records: the template renders the history differently with and without the final message |

With the trainer measure, the refused records were left out as `over_block` with the first refusal named.

Related (fixed 2026-11-21, splinter ab8c48544): the policy-gradient summary of `splinter train --json` reported
`steps_skipped_guard: 0` while brain had skipped 16 steps. Count skips from the per-step `skipped` field on runs
trained before that commit.
