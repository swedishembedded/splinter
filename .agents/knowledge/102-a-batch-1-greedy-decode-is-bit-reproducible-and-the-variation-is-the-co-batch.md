<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 102. A batch-1 greedy decode is bit-reproducible, and the variation is the co-batch

Knowledge 101 concluded that the serving kernels are numerically nondeterministic even at batch
1, from a task-derived probe in which three sequential identical requests gave three distinct
tool calls. That conclusion was an artifact of the instrument: the probe hashed the whole
response payload, and two fields of it are random per response - the tool-call id,
`format!("call_{:016x}", crate::rng::random_seed())` in splinter's sibling brain
(`crates/data/src/qwen_chat.rs`), and the chat completion id (a UUID). Neither is rendered into
the prompt: the chat template emits a tool call's name and arguments only. Every repeat
therefore "differed" without the model differing.

Remeasured against the completion's own `token_ids` and per-token `logprobs` (both available
with `return_token_ids` and `logprobs`) on the frozen evaluation build, the same task-derived
request (real coding system prompt, first user request, eight tool schemas, 2444 prompt
tokens), greedy with the requested seed echoed by the server:

| requests | distinct completion token sequences | first divergent token |
|---|---|---|
| 5 sequential (batch 1) | **1** | none |
| 6 sequential (batch 1) | 1 | none |
| 2 concurrent | 1 | none |
| 6 concurrent | **2** | 7 |
| 8 concurrent | **2** | 7 |

(96- and 12-token completions agree; the 12-token runs exclude mid-flight row churn.) Over the
5 sequential requests the chosen-token log-probabilities were **bitwise identical** (max spread
0.000e+00), so the prefill, every decode step and the sampler reproduce exactly at one row.

Why a co-batch changes a row - two mechanisms, both measured in the source:

- `matmul_i8_gemv` is "bit-identical to matmul_i8_gemv_reg below 5 rows ... int8 tensor cores
  (mma.sync m16n8k32) from 5 rows" (`crates/kernels-cuda/src/lib.rs`, `I8_GEMV_MMA_MIN_ROWS = 5`);
  the decode path passes all `bsz` rows through one matmul and the int8 head gets `m = bsz`,
  so from five rows on the arithmetic changes for every row.
- the fused decode attention sizes its split-key count from the batch and from the batch-wide
  `cap` (the longest neighbour): `flash_splits(cap, batch, n_kv_heads)` in
  `crates/model/src/kv_tier.rs`, so the softmax grouping changes with the co-batch.

**The cache is not a cause (13:25).** `runs/g8/det-cache.py` sends one never-seen prompt once
(cold), twice more (cache hits) and then six times at once, at the record's own length and at
about four times it: cold and warm are identical at both lengths, warm repeats are identical,
and six concurrent copies give three and five distinct sequences. So a cache hit's KV is the
cold prefill's KV.

**One request in flight is not enough for an episode (13:04).** The same adapter on the same
eight tasks, greedy at `ARM_CONCURRENCY=1`, passed 3 of 8 and then 5 of 8 with two verdicts
flipped (the runs took 481 s and 781 s); a concurrency-8 run of the same arm passed 5 and
differed from the first on two tasks. So a bit-identical *request* does not make an *episode*
reproducible and `ARM_CONCURRENCY=1` alone is not a deterministic instrument. What remains is
on the episode side - the observations a coding agent sees (command output, timings, paths,
listing order) or the prefill partition of a growing conversation - and it is not yet
identified.

Consequences for evaluation, which is where this cost real measurements: an arm run at
`--remote-concurrency 8` draws a fresh batch composition on every run and at every step, and
even one request in flight leaves an episode's own variance, so a single greedy arm's task
delta under a few tasks is not signal (knowledge 101's 30/31/26 spread is this, not kernel
randomness). The batched half can be made exact without giving up batching: group the int8
projections (the prefill path already does this in `exact_group_rows()`, the decode path had
none) and pin the split count to the context rather than to the batch - `--exact-decode` in
brain - with the token-id repeat test above as its acceptance test.

Open attribution: the six concurrent requests also arrive together, so their prefills can share
a round (rounds above 32 rows take the chunk tape, gated to 1e-4 rather than bits). The
discriminator is to prefill once so the later prefills are cache hits and then send six
concurrent requests: identical output means the decode row count is the cause.

Evidence: `runs/g8/det-margins-a2.out` (5 sequential, token ids and log-probabilities),
`runs/g8/det-batch-frozen.out` and `runs/g8/det-batch-small.out` (the concurrency matrix),
`runs/g8/det-arm-c1.out` (an arm at concurrency 1 against the same arm at 8), the derived record
`runs/g8/determinism-record.json`, and the write-up `runs/g8/determinism-2026-10-25b.md`.
The paper carries it as claim REP2 (`papers/splinter-brain-sven-flywheel`).
