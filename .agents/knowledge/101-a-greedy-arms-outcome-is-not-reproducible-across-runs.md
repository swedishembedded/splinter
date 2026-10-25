<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 101. A greedy arm's outcome is not reproducible across runs

The same adapter (decisive seed 13), the same D46 base set (`blake3:1a5cfb99`) and greedy
decoding gave **30/46** on card 0, then **31/46** and **26/46** on card 1 in two later runs. The
three reports name the same 46 task ids; their verdicts differ on 9, 11 and 12 tasks in the three
pairwise comparisons. The first divergence of a flipped task is in the model's own tool arguments
at step 1, with an identical step-0 prompt. This is run-to-run variation of one greedy arm, not a
comparison of different adapters.

A serving probe (`determinism-probe.py`) sent the same prompt greedily several times in sequence
and several times at once. The stored output records one server returning 1 distinct completion
for both the sequential and the six concurrent requests. The session also observed a second
server return 3 distinct completions for six identical concurrent requests, but that output was
not kept, and a later re-run at a larger token budget on both live servers was identical; the
mechanism - batch or prefix-cache state is plausible - is therefore **not conclusively isolated**.
What is solid is the consequence for measurement: single greedy arms carry roughly a 2.5-task
standard deviation on D46 (the sample standard deviation of 30, 31 and 26 is 2.65), so a paired
p-value computed over one run per arm overstates its precision. The registered three-decoding-seed
protocol is the right instrument; the development probes in knowledge 098, 099 and 100 are single
runs and should be read with this variation in mind.

Mitigations to test are batch-1 evaluation (`ARM_CONCURRENCY=1`) or bit-exact per-row batched
kernels. A batch-1 against batch-8 matrix (fresh servers, two runs each, three sequential and
six concurrent identical requests) was deterministic in all four runs on the probe's synthetic
prompt, while the seeded concurrent test that produced three distinct completions used a
different prompt and a server that had already served evaluation traffic. The variation is
therefore prompt-dependent, and the simple probe is not a reproduction. A **task-derived**
probe (the real coding system prompt, first request and eight tool schemas from a stored
record, greedy, fixed seed) reproduced it exactly: three sequential identical requests gave
three distinct tool calls, six concurrent gave six, and the server echoed the requested seed
(`brain.seed`) on every call. Since sequential requests differ with the seed honoured, the
cause is run-to-run numerical nondeterminism in the serving kernels at near-tie decisions, not
the sampler and not batch composition alone. Batch-1 is therefore **not** the fix. The
remedies are deterministic/bit-exact kernels (pinned reduction order, no atomics or split-k
in the sensitive path) or multiple decoding replicates per claim, which the registered
protocol already prescribes. Until one is measured, a difference of a few tasks between two single greedy arms is
inside the noise this entry measures; the large same-seed supervision gap and the rehearsal
contrast are larger than it, but they are still single-run estimates.

Evidence: `runs/g8/pilot-s13-d46-decisive-s13.verify.json` (30 passed),
`runs/g8/pilot-rep-decisive-s13.verify.json` (31),
`runs/g8/pilot-rep2-decisive-s13.verify.json` (26), their comparisons
`runs/g8/pilot-compare-rep-decisive-s13.json` and `-rep2-.json`,
`runs/g8/logs-queue-rep.out` and `logs-queue-rep2.out` (same model, seed and base set),
`runs/g8/determinism-probe.py` and `runs/g8/determinism-probe.out`.
