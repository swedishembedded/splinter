<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI3: self-taught generations from the environment's own answers, judged per lineage by bench

Written on 2026-11-13, before any run of RSI3. The design, exam, criterion and analysis below are fixed by this commit.
The builds, digests and the settings resolved from the defaults named here are recorded in a launch record
(`rsi3-launch.md`) committed before generation 1; anything changed after that is reported as a dated deviation beside the
outcome, which is reported whatever it is.

## Why a third registration

RSI2 (on-policy policy gradient, outcome entry in `rsi2.md`) raised the learner's own dev exam but moved the sealed exam
of items the base never answers by 2-4 items at generation 1 in every lineage; no lineage passed. RSI1's arm H, whose
update trains on the Lua interpreter's output for programs the model itself proposed (including the ones it mispredicted),
moved the same exam by 16-59 items at generation 1, but its per-lineage replication (arm HL) met the criterion in 1 of 4
lineages: later generations stalled (no round accepted) or traded items. RSI3 keeps arm H's source of new information and
replaces the parts that failed: underpowered acceptance, no replay across generations, a fixed proposal pool, and a
learner that ran the model in its own process.

## Design

- **Policy:** Qwen/Qwen3-4B, the same base for all lineages; thinking off.
- **Lineages:** four, differing only in the learner's seed; three generations each from the base.
- **Generation:** started from the previous champion (generation 1 from the base). The current policy proposes Lua 5.1
  programs from the Lua 5.1 reference manual and predicts what each prints; the Lua interpreter runs each program and
  its output is the label. The generation trains on its own proposals labelled by the interpreter (mispredicted and
  correct ones; the mix and weighting are fixed in the launch record), plus a replay share of earlier generations'
  accepted data. No stronger model, teacher or human-written answer enters training.
- **Engine:** brain is reached only through its API (serving and training jobs); every policy decision is served.
- **Acceptance inside the loop:** a dev set and a retention set built only from the learner's own proposals and
  generator (never the exam), sized by power-first planning; the generation's champion is its last accepted policy
  (none accepted: the predecessor stays champion and that step fails).

## Exam (sealed, bench only)

Unchanged from RSI2: bench's sealed 889-item Lua exam (items the base never answered under the neutral protocol), held,
served and graded only by bench on its pinned build, direct-completion protocol (one greedy completion, no system
prompt, thinking off, last fenced block, whitespace-normalised token compare). The learner never reads it. Each
generation hands bench its candidate and the full list of programs it trained on. **Bench excludes from a step's
paired comparison every exam item that matches (exactly or after removing comments and whitespace) any program either
compared model was trained on**, and reports how many it excluded; the step is decided on the remaining items.

## Criterion

As RSI2: for each lineage and each step (generation 1 vs base, 2 vs 1, 3 vs 2), wins and losses over the compared
items, one-sided exact sign test; a step holds when wins exceed losses and p < 0.05; a lineage succeeds when all three
steps hold; **RSI3 meets its criterion when at least three of the four lineages succeed.** Pooled per-step tests,
exclusion counts, dev and retention trajectories and GPU-hours are reported beside it.

## What is not claimed

Nothing until bench's verdicts evaluate the criterion; the outcome is recorded whatever it is.

## Outcome (2026-11-13): criterion not met, two of four lineages improved at every generation

Bench examined every champion on its pinned build (brain d11e91340, direct-completion protocol, sealed 889-item exam,
training-overlap exclusion on) and returned signed aggregate verdicts; its signed plan verdict reads `plan_met: false`.
Items right by generation (base 0 in every lineage), with each step's wins/losses and one-sided sign-test p:

| Lineage | gen 1 vs base | gen 2 vs gen 1 | gen 3 vs gen 2 | succeeds |
|---|---|---|---|---|
| r0 | 58 (58/0, 3.5e-18) | 122 (78/14, 3.1e-12) | 123 (29/28, 0.50) | no |
| r1 | 72 (72/0, 2.1e-22) | 101 (41/12, 4.1e-5) | 91 (18/28, 0.95) | no |
| r2 | 67 (67/0, 6.8e-21) | 86 (38/19, 0.0082) | 121 (48/13, 3.8e-6) | yes |
| r3 | 91 (91/0, 4.0e-28) | 118 (42/15, 0.00023) | 142 (47/23, 0.0028) | yes |

Pooled over the four lineages: gen 1 vs base 288/0 (p 2.0e-87), gen 2 vs gen 1 199/60 (p 6.9e-19), gen 3 vs gen 2
142/92 (p 0.00065). No exam item matched any training program in any comparison (0 excluded) and no integrity flag was
raised. The criterion (at least three of four lineages succeed) is not met: every lineage improved at generations 1 and
2; at generation 3 r0 held flat and r1 fell (its own dev gate had accepted it). Deviations: r2's held-out proposal cap
raised before its first round (see `rsi3-launch.md`); r3's generation 3 failed once on a transient connection close and
was resumed from its journals with the same build and settings. Learning-signal decay is visible in the learner's
records: generation 1 trained on 568 mispredicted programs of 600 proposals, generation 2 on 229 of 554. RSI4
(`rsi4.md`) tests cumulative retraining from the base on frontier-selected programs.
