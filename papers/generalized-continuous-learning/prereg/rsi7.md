<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI7: continual acquisition of task families from semantic data, each generation judged on a fresh sealed semantic exam

Written on 2026-11-16, before any RSI6 lineage, exam draw or examination. The question, arms, exam and criteria below are fixed
by this commit; builds, digests, item counts and template counts go into `rsi7-launch.md`, committed before any candidate is
examined. Later changes are dated deviations recorded beside the outcome, which is reported whatever it is.

## Question

As RSI6 (stopped after 11 of 32 candidates, see `rsi6.md`), with one change on each side: does the loop improve at every generation
when each generation acquires a new task family **and every training record and exam item tests the language's semantics rather
than memorised digits or interpreter internals**? In RSI6's learner records about a quarter of the Lua frontier (and 12-22% of the
Python and dialect frontiers) had answers needing long float digits, huge integers, object reprs or internals listings; the
frontier rule over-selects them because the model gets them wrong.

## Families (each exactly verified by its pinned interpreter)

1. `lua51`: Lua 5.1.5 output prediction (RSI5's family).
2. `python3`: Python 3 output prediction (the node's Python 3.14.4, run isolated).
3. `zero-index`: a counterfactual Lua dialect (0-based string positions, `#s` is the length less one), with its generated manual.
4. `plus-concat`: a counterfactual Lua dialect (`..` removed, `+` concatenates two strings), with its generated manual.

No pretrained model has seen the two dialects; the base model's learner-side development accuracy on them in the smoke run
was 0.24 and 0.38 against 0.54 for Lua 5.1.

## Arms

- **Acquisition lineages** `a0`..`a3`: generation k finds a frontier of 1500 programs of family k (the champion proposes
  from family k's manual, the family's interpreter labels, a program enters when at least one of 4 predictions at
  temperature 1.0 is wrong) and trains a fresh LoRA from the base on the union of the frontiers of families 1..k (1500k
  records).
- **Volume-control lineages** `c0`..`c3`: identical, except that every generation finds 1500 more `lua51` programs, so
  generation k trains on 1500k `lua51` records: the same amount of verified data, no new family.
- **Training:** rank 16, learning rate 1e-4, 8 records per step, 2 epochs, weight decay 0, max_block 1024, thinking off,
  the measured-length rule for over-long records.
- **Learner gate:** RSI5's (one-sided sign test on the learner's pooled development items of all four families). A rejected
  candidate is recorded, is not handed off, and the lineage continues from its champion; for the criteria, a lineage's
  generation-k model is then its last handed-off champion (no improvement is credited for that step).
- **Semantic admission (learner):** every family's programs are admitted only when their interpreter output passes the
  learner's semantic rule (no fractional number with more than 6 significant digits, no integer of more than 7 digits, no object or
  address repr, no listing of two or more dunder names, no interpreter version); refused programs are journalled with the reason.
- **Proposer novelty and frontier yield** are logged per family and generation as leading indicators (reported, no criterion).

## Exam I

Drawn by the examiner from its own generators before any lineage runs, with a new seed only the examiner holds: Lua 5.1 (template
set v2), Python 3, and both dialects, 10 items per template family, **semantic references only** (the examiner's own rule,
`--semantic`, written independently of the learner's), one combined collection with a per-language stratum. Templates that cannot
fill their share with semantic references are dropped and listed in the exam's provenance. Programs any examined candidate trained
on are excluded before grading. Protocol: the registered direct-completion protocol (greedy, thinking off, f32 KV), one pinned build.

## Criteria (the examiner's bootstrap over templates and lineages, 95% intervals)

- **I1 (primary, every generation):** for the acquisition lineages, each step (generation k vs k-1, k = 1..4, generation 0
  being the base) improves the mean over the four languages of the share right on I (each language weighted equally): lower
  bound above zero. Met when all four steps hold.
- **I2 (primary, acquisition beats volume):** at generation 4, the acquisition lineages' language-mean exceeds the
  volume-control lineages': lower bound above zero.
- **I3 (secondary, retention):** for each family j < 4, the acquisition lineages' share right on family j's stratum after
  generation 4 is no more than 2 points below its value after generation j (upper bound of the drop below 0.02).
- Reported without criteria: each language's stratum at every generation (the skill-by-generation matrix), per-lineage sign
  tests, proposer novelty and yield, GPU-hours per generation.

## Separation

As RSI5: the learner never reads I or any item-level result; signed aggregate verdicts only, after the last hand-off.

## What is not claimed

Nothing until the verdicts evaluate I1, I2 and I3.
