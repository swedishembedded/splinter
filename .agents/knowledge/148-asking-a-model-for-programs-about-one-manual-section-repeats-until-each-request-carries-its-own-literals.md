<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 148. Asking a model for programs about one manual section repeats until each request carries its own literals

**Lesson:** A self-proposing loop saturates because the same section and goal make Qwen3-4B (temperature 1.2, top-k 20) write
the same programs; varying the instruction's angle did not help, but asking each request to use its own seeded word and number
did.

## Measured (Qwen3-4B proposing `predict-lua` programs from three Lua 5.1 manual pages, brain serve, 64 requests in flight)

| Requests | Windows of three sections (RSI2) | One section each | + 12 angles | + angles + seeded literals |
|---|---|---|---|---|
| 320 | about 240 programs at saturation | 235 kept, 207 of 442 admitted repeats (at 384) | 186 kept, 131 repeats | 357 kept, 52 repeats |

With literals, 448 requests yielded 450 distinct programs and a second pass of 704 requests 604 more (repeats 266 of 870
admitted); a second round from the first round's champion yielded about 540 per 704 requests with 333 repeats of 873, so the
pool is finite: proposing for a later round costs more per program. The base answered 17 percent of those proposals right
(102 of 600), so most proposals are informative.

## What this does and does not establish

Diversity per request is a property of the prompt, not of the sampler settings the library fixes. Not established: that more
distinct programs transfer better; the pilot (launch record of RSI3) shows a 300-program round moving the dev set as much as 600.
