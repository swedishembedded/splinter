<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI6: continual acquisition of task families, each generation judged on a fresh sealed exam

Written on 2026-11-16, before any RSI6 lineage, exam draw or examination. The question, arms, exam and criteria below are fixed
by this commit; builds, digests, item counts and template counts go into `rsi6-launch.md`, committed before any candidate is
examined. Later changes are dated deviations recorded beside the outcome, which is reported whatever it is.

## Question

RSI5 showed one task family learned from one source of programs saturates: generation-to-generation gains on a fresh exam fell
from +5.1 to +0.3 points as the pool doubled. Does the same loop improve at every generation when each generation acquires a
new task family while keeping the earlier ones, judged on one fresh sealed exam spanning all families, with retention of
earlier families and a control that adds data but no new family?

## Families (each exactly verified by its pinned interpreter)

1. `lua51`: Lua 5.1.5 output prediction (RSI5's family).
2. `python3`: Python 3 output prediction (the node's Python 3.14.4, run isolated).
3. `zero-index`: a counterfactual Lua dialect (0-based string positions, `#s` is the length less one), with its generated manual.
4. `plus-concat`: a counterfactual Lua dialect (`..` removed, `+` concatenates two strings), with its generated manual.

No pretrained model has seen the two dialects; the base model's learner-side development accuracy on them in the smoke run
was 0.24 and 0.38 against 0.54 for Lua 5.1.

## Arms

- **Acquisition lineages** `a0`..`a3`: generation k finds a frontier of 2000 programs of family k (the champion proposes
  from family k's manual, the family's interpreter labels, a program enters when at least one of 4 predictions at
  temperature 1.0 is wrong) and trains a fresh LoRA from the base on the union of the frontiers of families 1..k (2000k
  records).
- **Volume-control lineages** `c0`..`c3`: identical, except that every generation finds 2000 more `lua51` programs, so
  generation k trains on 2000k `lua51` records: the same amount of verified data, no new family.
- **Training:** rank 16, learning rate 1e-4, 8 records per step, 2 epochs, weight decay 0, max_block 1024, thinking off,
  the measured-length rule for over-long records.
- **Learner gate:** RSI5's (one-sided sign test on the learner's pooled development items of all four families). A rejected
  candidate is recorded, is not handed off, and the lineage continues from its champion; for the criteria, a lineage's
  generation-k model is then its last handed-off champion (no improvement is credited for that step).
- **Proposer novelty and frontier yield** are logged per family and generation as leading indicators (reported, no criterion).

## Exam H

Drawn by the examiner from its own generators before any lineage runs, with a new seed only the examiner holds: Lua 5.1 (template
set v2), Python 3, and both dialects, 10 items per template family, one combined collection with a per-language stratum. Items
are not selected by any model's answers. Programs any examined candidate trained on are excluded before grading. Protocol: the
registered direct-completion protocol (greedy, thinking off, f32 KV), one pinned build for every candidate.

## Criteria (the examiner's bootstrap over templates and lineages, 95% intervals)

- **H1 (primary, every generation):** for the acquisition lineages, each step (generation k vs k-1, k = 1..4, generation 0
  being the base) improves the mean over the four languages of the share right on H (each language weighted equally): lower
  bound above zero. Met when all four steps hold.
- **H2 (primary, acquisition beats volume):** at generation 4, the acquisition lineages' language-mean exceeds the
  volume-control lineages': lower bound above zero.
- **H3 (secondary, retention):** for each family j < 4, the acquisition lineages' share right on family j's stratum after
  generation 4 is no more than 2 points below its value after generation j (upper bound of the drop below 0.02).
- Reported without criteria: each language's stratum at every generation (the skill-by-generation matrix), per-lineage sign
  tests, proposer novelty and yield, GPU-hours per generation.

## Separation

As RSI5: the learner never reads H or any item-level result; signed aggregate verdicts only, after the last hand-off.

## What is not claimed

Nothing until the verdicts evaluate H1, H2 and H3.
