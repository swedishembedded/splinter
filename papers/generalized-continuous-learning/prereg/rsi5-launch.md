<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# RSI5 launch record

Committed on 2026-11-16 before any RSI5 lineage ran and before any candidate was examined; completes `rsi5.md`.

## Builds

- Serving and training (learner and examiner alike, one frozen copy for the whole run): brain `e70131163` (origin/main of
  2026-11-16), binary `containers/bin/brain-e70131163`, SHA-256
  `01e7c73d27ce805c091c96ac25628c2300f3f6719403a64bba7fa2294eaec876`.
- Learner: splinter `793bb9cef89770bfc58a5a4282cb75ccd9013113` (main; includes the measured-length frontier rule `019239f10`
  and the opt-in reasoning options, unused here), built against the pinned sven `6f4b1716` and brain `796871647` crates;
  binary `containers/bin/lineage-793bb9cef`, SHA-256 `2342bac33058384462cccc2d09f3ecb451c542d0b58eb86e8fd1ceb8e05e49be`.
- Lineage script `~/project/shared/rsi5/run-lineage.sh`, SHA-256
  `b7b133471ba53b76246902793dd77e05c9edd8b5ee630f69c930253462e39947`; one process per generation under the card's lock.
- Examiner: bench `c883edf14d2293c93372061777bdaf5b601d25eb`; group plan `rsi5-G` (SHA-256
  `30fde85afc387df3e88eaa1836c5bec03000f1182a5c5737787ce200ead91b04`): steps 1-4, generation 4 over base, strata by base answer
  and by family set, bootstrap 10000 over templates and lineages, 95% intervals.

## Exam G

Drawn by the examiner on 2026-11-16 with template set `v2` and 15 items per family: 2040 items in 136 families (1170 items
of the 78 families used for exam F, 870 of 58 families never used in any exam), SHA-256
`cd9f1920846b6939ae8622312185e06cf2858cb91fc59fd63b43d5b5cccb6bd8`. The seed and the items are held by the examiner only.

## Cards

Three GH200 cards (the fourth serves another project): lineage `g0` on gpu0, `g1` on gpu1, `g2` and `g3` share gpu2, taking
the card's lock one generation at a time. Examination runs after every lineage has handed off its last candidate.
