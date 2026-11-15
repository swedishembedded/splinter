<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of two RSI4 follow-ups: a fresh exam (F) and a matched-data control (M)

Written on 2026-11-15, after RSI4's outcome (`rsi4.md`) and before any run of either follow-up. The questions, arms, exams and
criteria below are fixed by this commit; builds, digests, item counts and the exam's template count go into `rsi4-followups-launch.md`,
committed before any candidate is examined. Later changes are dated deviations beside the outcome, which is recorded whatever it is.

## Why

RSI4 met its registered criterion (per-lineage one-sided exact sign tests over items). Two questions remain that the registered result
does not answer, raised by an independent review and an external assessment:

1. **Robustness of the exam result.** The exam's 889 items come from 25 program templates; the examiner's family-clustered 95%
   intervals include zero for the generation-3 step of three lineages, and the same exam was used, with designs changing, across
   RSI1 to RSI4. Does RSI4's improvement replicate on an exam drawn afresh, never used before, with more templates?
2. **Whether the improvement is recursive.** Each RSI4 generation trains from the base on more verified data (300, 900, 2100 records)
   proposed by the previous champion. Is generation 3 better than the same amount of interpreter-verified frontier data proposed by
   the base and trained once from the base? If not, the gain comes from the amount of verified data, not from an improved model
   proposing better material.

## F - re-examination on a fresh exam

- **Exam:** drawn by the examiner from its own program generator with a new seed it alone holds, never shown to the learner, never
  published, never used before; at least 50 program templates and at least 1500 items; items are **not** selected by how the base
  answers them (the base's own pass rate on F is reported). Exact or normalised duplicates of any program any RSI4 candidate or M
  candidate trained on are excluded before grading, as before.
- **Models:** the base and the twelve RSI4 champions (four lineages, three generations), exactly the adapters handed over in RSI4,
  served on the examiner's pinned build at f32 KV under the same direct-completion protocol.
- **Primary criterion (F1):** a step (generation 1 vs base, 2 vs 1, 3 vs 2) holds when wins exceed losses and the one-sided exact
  sign test over items gives p < 0.05; a lineage replicates when all three steps hold; **F1 is met when at least three of four
  lineages replicate.**
- **Co-primary, cluster-robust (F2):** the same steps judged by the one-sided 95% template-clustered interval of the paired gain
  (lower bound above zero); reported per step and per lineage, met under the same 3-of-4 rule.

## M - matched-data control

- **Arms:** four control lineages, one per new seed, each a single generation from the base: the **base** proposes programs with
  RSI4's proposer and settings until it holds 2100 frontier programs (4 samples at temperature 1.0, a program enters when at least one
  prediction is wrong), the interpreter labels them, and a fresh LoRA is trained from the base on those 2100 records for 2 epochs
  (525 steps), every other training setting as RSI4's. Built with RSI4's binaries.
- **Comparison:** each control is examined on F (and, for continuity, on the RSI4 exam) and compared with RSI4's generation-3
  champions.
- **Criterion (M1):** for each of the four (RSI4 lineage, control) pairs matched by index, generation 3 vs control by the one-sided
  exact sign test over F's items; **"the improvement is recursive" is supported when at least three of four pairs favour generation 3
  at p < 0.05**; the opposite direction is reported the same way. Template-clustered intervals are reported beside it.
- **Seeds:** m0 8785991200924142667, m1 7404742185133534162, m2 811289239880634778, m3 2657151457107420430; generator seed
  6881244634785368405.

## Separation

As RSI4: the learner never reads F or any verdict; the examiner receives each M candidate with the full list of programs it trained
on through the one-way hand-off and returns signed aggregate verdicts only.

## What is not claimed

Nothing until the verdicts evaluate F1, F2 and M1; each is recorded whatever it is.
