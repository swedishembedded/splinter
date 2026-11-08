<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 106. A round stopped asking for tasks at its first barren request

A round wants 12 admitted training tasks over at most 6 requests, but the ask-again loop ended at
the first later request that admitted none (a declined reply counts as one malformed proposal), and
each request was shown only the tasks that same round had admitted. Across 197 rounds of the
Qwen3-4B Lua self-generation runs (4 replicates, 3 generations) a round admitted 5.4 tasks on
average, 31 admitted none, and 75 ended with at most 10 proposals (one request of eight and at most
a declined reply after it); 23 reached 12. Refusals were mostly repeats (`repeats_another_task` 737
of 1884): the generator was not told what earlier rounds of the skill had admitted, nor what it had
been refused for repeating, and a round with fewer than two records cannot train.

A round now keeps asking until `--min-tasks-per-round` tasks are admitted or
`--max-proposal-batches` requests are spent, even past a request that admitted nothing; every
request is shown the skill's admitted and repeated instructions from this and earlier rounds (never
its probes), and every request's proposed, admitted and refused counts are in the round record.
Probe writing keeps its stop at the first barren request.

Evidence: `~/project/shared/containers/rsi1/lua/r*/gen*.json` (splinter `b688e3ee7`).
