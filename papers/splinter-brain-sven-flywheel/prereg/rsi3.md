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
