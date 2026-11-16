<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Compact, correct reasoning roadmap

Scope: teaching a model to reason before it answers, with traces that are concise yet sufficient, verified by an executable
oracle, and rewritten by a separate compactor that keeps only what led to the correct answer; a follow-on paper to "The Frontier
Must Grow".

**Status: planned. Nothing below has been started.** Update this file in the same commit that starts, finishes or changes an
item.

## Why

The self-taught generations of the flywheel paper (S1-S4) train and examine with thinking off: every record supervises only the
final output (`uaple`, `candle\t1`), and the model never calls a tool or the interpreter. On the fresh exam F the untrained base
answers 8.3% directly and at least 47.3% with thinking on (8192-token budget, 372 empty replies, so a lower bound); generation 3
reaches 16.7% directly. The headroom is in reasoning, and nothing in S1-S4 teaches or measures it.

## Idea

Let the model reason freely with thinking on, including false starts; keep only attempts whose final answer the interpreter
confirms; a separate **compactor** agent rewrites each successful trace into the shortest trace that still leads to the answer
(keeping the steps that produced it, dropping detours and errors); verify the compacted trace; train with thinking on on
(prompt, compacted trace, answer). The model may reason badly at first and still learn from the parts that worked, and its traces
should get shorter and more reliable over generations.

## Pilot evidence (2026-11-15, learner-side, not examined)

Shortest-correct-trace self-training (STaR with the interpreter as verifier, base Qwen3-4B, 4096-token budget, 4 samples,
2186 verified traces) LOWERED thinking-on development accuracy from 0.402 to 0.31-0.34 at lr 3e-5, 1e-4 and 2e-4, while the
trainer's held-out loss improved (0.236 -> 0.109); replies with no fenced answer rose (493 -> 661 of 1600). Answer-only
interpreter labels raised thinking-on accuracy to 0.776 (about half of the base's thinking-on misses were format failures).
Traces sampled from the answer-trained adapter gave no gain over it. Unproven hypothesis: the shortest correct base trace
rewards lucky shortcuts and teaches under-verification. Any compactor (CR1-CR5) must beat answer-only training, not only raw
traces, and selection by length alone is suspect. Pilot files: `~/project/shared/tmp/lsp` (harness `lsp.py`).

## Questions (each one registered before its run, judged by the separate examiner on a sealed exam)

- CR1: does training on compacted verified traces beat training on raw verified traces (STaR / rejection sampling) at equal
  compute, on thinking-on accuracy and on tokens per solved item?
- CR2: does it beat answer-only training (the S4 recipe) on thinking-on accuracy, and does answer-only training hurt or help
  thinking-on accuracy (measure the existing S4 adapters with thinking on first: descriptive, cheap)?
- CR3: does the compactor preserve correctness: compacted traces that no longer lead to the answer, or that contain claims the
  interpreter contradicts, must be measured and rejected, not trained on.
- CR4: does verification of intermediate steps matter: a compactor checked only by the final answer vs one whose kept steps are
  checked against interpreter execution traces (Lua debug hooks: line, variable values)?
- CR5: does it compound over generations (the S4 loop with thinking on and the compactor in the loop), with the matched-data
  control of the S4 follow-ups as the template for separating volume from improvement?

## Design notes

- Compactor: a separate model role (initially the 27B or the same 4B with a distinct prompt); it never sees the exam; its output
  is verified, never trusted. A faithfulness check replays the compacted trace: the policy conditioned on it must reach the same
  answer, and every value it states that the interpreter can check (execution-trace states) must match.
- Arms at equal compute: answer-only (S4), raw verified traces, compacted traces, compacted + step-verified traces,
  execution-trace supervision (traces generated from the interpreter's debug hooks, no model reasoning), on-policy RL with the
  interpreter's verdict as reward (brain's policy gradient; needs the numerics gate on the exact path).
- Metrics: thinking-on and thinking-off accuracy on a sealed fresh exam with template-clustered intervals; tokens per solved
  item and trace length distribution; fraction of traces with a step the interpreter contradicts; held-out templates and an
  unseen language feature family for transfer.
- Ownership: splinter (curation, compactor role, verification policy, campaigns); sven (the compactor and policy calls, trace
  records); brain (thinking-on training and serving, policy gradient); the examiner (sealed exam, thinking-on protocol).

## Prerequisites

- [ ] The flywheel paper final (S4 follow-ups reported).
- [ ] Descriptive thinking-on examination of the S4 champions and base on F (examiner protocol variant exists).
- [ ] An execution-trace extractor for Lua 5.1 programs (debug.sethook line and local-variable states), deterministic.
- [ ] A compactor role and its faithfulness check in splinter, with tests on synthetic traces.
- [ ] Registration of CR1-CR4 with power planning from a pilot.
