<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# RSI7 launch record

Committed on 2026-11-17 before any RSI7 lineage ran and before any candidate was examined; completes `rsi7.md`.

## Builds (one frozen copy each for the whole run)

- Serving and training, learner and examiner: brain `c37047f52`, binary `containers/bin/brain-c37047f52`, SHA-256
  `5cab550feaa17328b65e775bb9d2b58b36637fd5bc19b8dad8737150de575bc0` (RSI6's build).
- Learner: splinter `c705cc4b7392efb9b49115389cbfe7c37c037785` (main; task families and the semantic output rule
  `--admit-output semantic` of `f2cda04a9`: no fractional number with more than 6 significant digits, no integer of more than 7
  digits, no address or object repr, no two or more dunder names, no interpreter version; refusals journalled in `refused.jsonl`),
  built against the pinned sven `0d7d6643` and brain `796871647` crates; binary `containers/bin/lineage-c705cc4b7`, SHA-256
  `86e8558624227f3a6b04690912dc75df1ee7ac5840e3dc15579ed85871690e56`.
- Lineage script `~/project/shared/rsi7/run-lineage.sh ARM SERIES CARD SEED [FROM]`, SHA-256
  `9b3c42a3f6e9f36977e27c10e8c274bd41e4919512e9995a60bfb6da6e8472db`: RSI6's script with the frontier target 1500, the semantic
  output rule, the learner commits above and its own development-generator seed.
- Examiner: bench `c491698b` (semantic exams); group plan `rsi7-I` (SHA-256 `ecaa707e...`; as `rsi6-H`: groups base,
  acq1-acq4, vol4; comparisons step1-step4, acquisition-over-volume, retention1-retention3, acq4-over-base; languages weighted
  equally); secondary plan `rsi7-I-steps` (`fcf151aa...`); examination script `examiner-private/rsi7/run-I.sh` (`940af664...`).

## Exam I

Drawn by the examiner on 2026-11-17 before any lineage ran, with `--semantic`: 3930 items in 393 families (Lua 5.1 1350, Python 3
670, `lua51-dialect-a` = `zero-index` 790, `lua51-dialect-b` = `plus-concat` 1120), 10 per family, 9 templates dropped as unable
to fill their share with semantic references (listed in the provenance), SHA-256
`91269964806dbdb2b3e2b5e32380d7650cec15b7c2f54513dd5fdb1addf09699`. Seed and items are held by the examiner only.

## Seeds and cards

- Acquisition: `a0` 2686826188015578942, `a1` 4462193450708056891, `a2` 2019155494163254621, `a3` 1215751275817771848.
- Volume control: `c0` 8171511350891784203, `c1` 3248978048455331431, `c2` 5291601049214148305, `c3` 8426585028971804896.
- Learner development generator seed 4264438271202500486.
- gpu0 `a0`, `c0`, `a3`; gpu1 `a1`, `c1`, `c3`; gpu2 `a2`, `c2` (gpu3 serves another project). Lineages sharing a card
  take its lock one generation at a time. A lineage stopped by a lease end or an infrastructure failure resumes from its first
  generation not handed off with the same script, seed and settings; each resume is listed beside the outcome.
