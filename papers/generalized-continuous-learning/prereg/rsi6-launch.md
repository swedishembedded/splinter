<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# RSI6 launch record

Committed on 2026-11-16 before any RSI6 lineage ran and before any candidate was examined; completes `rsi6.md`.

## Builds (one frozen copy each for the whole run)

- Serving and training, learner and examiner: brain `c37047f52` (origin/main of 2026-11-16, includes the device-side sampling
  of `4a6efd39c`; the build of the learner-side execution-trace pilot and the multi-family smoke run), binary
  `containers/bin/brain-c37047f52`, SHA-256 `5cab550feaa17328b65e775bb9d2b58b36637fd5bc19b8dad8737150de575bc0`.
- Learner: splinter `58874dba9e378f6f7aede3b0b8ffee953f08795e` (main; task families `lua51`, `python3`, `zero-index`,
  `plus-concat`), built against the pinned sven `6f4b1716` and brain `796871647` crates; binary `containers/bin/lineage-58874dba9`,
  SHA-256 `6016354de42d48c273152bd5b101e8e7ff7ba607b3bb4000520a92b9c2b391ca`.
- Lineage script `~/project/shared/rsi6/run-lineage.sh ARM SERIES CARD SEED [FROM]`, SHA-256
  `8badc5f8aeef28f89aff1ccb9215eabca7ad159355a2874489101255f6cf2994`: one process per generation under the card's lock,
  proposer state in a private memory-backed directory, request cap 40 000 per round, frontier target 2000, learner gate
  target effect 0.09 (RSI5's deviations 1, 2 and 4 applied from the start).
- Interpreters (pinned by SHA-256 in the learner's family specs and in the exam's provenance): Lua 5.1.5
  `43413003...25c7`, Python 3.14.4 `d1b87c36...eee9`, `zero-index` `4c91a008...e007`, `plus-concat` `dd32b653...b8a7`
  (`resources/lua51-dialects/SHA256SUMS`).
- Examiner: bench `fa4571d0`; group plan `rsi6-H` (SHA-256 `47b583f2e4ba4c883187b0cb556fa640fbe8dde18bab486a90c1fb98bc06de3f`):
  groups base, acq1-acq4, vol4; comparisons step1-step4, acquisition-over-volume, retention1-retention3, acq4-over-base;
  `weight = "family_set"` (each language weighted equally), bootstrap 10 000 over templates within languages and lineages.

## Exam H

Drawn by the examiner on 2026-11-16 before any lineage ran: 4020 items in 402 families (Lua 5.1 template set v2 1360,
Python 3 730, `lua51-dialect-a` = `zero-index` 800, `lua51-dialect-b` = `plus-concat` 1130), 10 per family, SHA-256
`c6cbda27469b4240b1e023aa5d256f7d4e6fb0f2c8bbf5649bd240ba746268ac`. Seed and items are held by the examiner only.

## Seeds and cards

- Acquisition: `a0` 8939798283933211487, `a1` 794745168020339165, `a2` 5956095000078893585, `a3` 2491191022992895262.
- Volume control: `c0` 1313972778602269514, `c1` 3737194048575806709, `c2` 5921694071382973231, `c3` 3583089160515051464.
- Learner development generator seed 4348667186414741079.
- Three GH200 cards (the fourth serves another project): gpu0 `a0`, `c0`, `a3`; gpu1 `a1`, `c1`, `c3`; gpu2 `a2`, `c2`. Lineages
  sharing a card take its lock one generation at a time.

## Leases

The run spans more than one GPU lease. A lineage stopped by the end of a lease is resumed in the next with the same script,
seed and settings from the first generation it has not handed off (`FROM`), reusing its journals; its in-memory proposer
state is rebuilt. Each resume is listed beside the outcome.
