<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI5: four generations of environment-supervised acquisition, each judged on a fresh sealed exam

Written on 2026-11-16, before any RSI5 lineage, exam draw or examination. The question, arms, exam and criteria below are fixed
by this commit; builds, digests, item counts and template counts go into `rsi5-launch.md`, committed before any candidate is
examined. Later changes are dated deviations recorded beside the outcome, which is reported whatever it is.

## Question

Does the learning loop of Splinter, Sven and Brain improve a 4B model at every generation, four generations in a row, on a
fresh exam drawn and graded by the separate examiner, and by a large total margin?

## What informed the design (learner-side only)

- The companion paper's RSI4 runs: generation 1 trained on 300 records at learning rate 3e-5 and did not improve on a fresh
  exam (F); generations 2 and 3 did.
- A learner-side pilot on 2026-11-15 (learner data only: lineage proposals and the lineages' own development items; no exam,
  no verdict). On 400 development programs, answer-only training on 4000 interpreter-labelled programs at learning rate 1e-4
  raised greedy direct-answer accuracy from 0.535 to 0.695, and a second round on 4000 fresh programs to 0.750. Training on the
  shortest interpreter-verified reasoning trace of the base model lowered thinking-on accuracy (0.402 to 0.31-0.34) at every
  learning rate tried. RSI5 therefore uses answer-only interpreter labels, not reasoning traces.

## Arms

- **Lineages:** four, `g0`..`g3`, seeds `g0` 8950147810279709875, `g1` 420195494785436082, `g2` 6979610617474167843,
  `g3` 9213004823522191878; lineage generator seed 3141303514277995914 (the learner's own development generator).
- **Base:** Qwen/Qwen3-4B.
- **Each generation** (the S4 cycle of the companion paper): the champion proposes Lua 5.1 programs from manual sections 5.4
  to 5.6; the pinned Lua 5.1.5 interpreter labels them; a program enters the frontier when at least one of 4 predictions by the
  champion at temperature 1.0 is wrong; frontier records longer than the training row are skipped by the measured-length rule;
  a fresh LoRA is trained **from the base** on the cumulative frontier pool for 2 epochs.
- **Pool sizes:** 1000, 2000, 4000, 8000 records after generations 1, 2, 3, 4 (new frontier programs per generation 1000,
  1000, 2000, 4000), so every step multiplies the pool by two. Request caps 10 times the generation's new target.
- **Training:** rank 16, learning rate 1e-4, 8 records per optimiser step, weight decay 0, max_block 1024, thinking off,
  replay share 0 (the cumulative pool contains every earlier record). Steps follow from the pool: 250, 500, 1000, 2000.
- **Learner gate:** the lineage's development and retention gate as in RSI4 (dev generator 100, dev proposals 300, one-sided
  sign test at alpha 0.025, retention margin 0.05). A rejected candidate is recorded and the lineage stops; it is not retried.
- **Hand-off:** each advanced champion goes to the examiner with the full list of programs it trained on, through the one-way
  hand-off; the learner receives signed aggregate verdicts only after the last generation of every lineage has been handed off.

## Exam G

- Drawn by the examiner from its own program generator, template set `v2` (the 78 families used for exam F plus 58 families
  never used in any exam), with a new seed only the examiner holds, never shown to the learner, never published, never used
  before; 15 items per family. Items are not selected by how any model answers them.
- Exact or normalised duplicates of any program any examined candidate trained on are excluded before grading.
- Protocol: the examiner's registered direct-completion protocol (one greedy completion, thinking off, no system prompt,
  last fenced block, f32 KV), every candidate served on one pinned build.

## Criteria

- **G1 (primary, every generation):** for each step (generation 1 vs base, 2 vs 1, 3 vs 2, 4 vs 3), the difference between
  the mean share of G's items answered right by the four lineages' candidates after and before the step, with a 95% interval
  from a bootstrap that resamples templates and lineages jointly. A step holds when the interval's lower bound is above zero.
  **G1 is met when all four steps hold.**
- **G2 (primary, strength):** generation 4 vs base by the same statistic; **met when the interval's lower bound is above 0.05**
  (five percentage points of G's items).
- **G3 (secondary, unseen families):** G2's statistic restricted to the `v2` families never used in any exam; reported with its
  interval, met when the lower bound is above zero.
- Reported beside them, without criteria: per-lineage one-sided exact sign tests over items for every step (item-level
  comparisons between independently trained adapters carry adapter-to-adapter variation, see the companion paper's seed null);
  items a generation loses that its predecessor answered; the worst per-template change; exam F (thinking off) for every
  candidate; the base and the generation-4 candidates on G with thinking on (descriptive); GPU-hours per generation (proposing,
  predicting, training, gating).

## Separation

The learner never reads G, F or any item-level result. The examiner receives candidates and training lists through the
hand-off and returns signed aggregate verdicts only.

## What is not claimed

Nothing until the verdicts evaluate G1, G2 and G3. No claim about recursion (an evolving proposer vs a frozen one) follows from
RSI5: it has no matched-data control.
