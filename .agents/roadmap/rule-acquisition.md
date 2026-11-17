<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Rule-acquisition efficiency roadmap (paper 3)

Scope: whether a model that has acquired more invented rule systems acquires a new, never-seen one with fewer verified experiences, examined closed-book on rule axes it never met, and the interpreter-grounded exploration that keeps supplying novel rule-exercising programs.

**Status: planned (2026-11-17).** Design from an advisor review after the external review of "The Frontier Must Grow" (arXiv
2611.20049: low novelty against arXiv 2207.14502, STaR, Absolute Zero 2505.03335; recursion not isolated). Update this file in the
same commit that starts, finishes or changes an item. Related: `generalized-autonomous-learning.md` (E8), RSI7 in
`papers/generalized-continuous-learning/prereg/rsi7.md`.

## Hypothesis

A model's cost to acquire a new rule system falls as the number of rule systems it has already acquired rises; the gain survives
placebo and volume controls, holds on rule axes never mutated in training, and lies in the learning slope, not only the
intercept.

## Prior art (arXiv IDs checked to exist by the advisor) and what is open

- Closest: Meta-RFFT 2502.11525 (multi-task rule-following pre-training helps an unseen task; rules in the prompt, length
  generalisation, no dose-response, no self-generated data). SEAL 2506.10943, Doc-to-LoRA 2602.15902, CaMeLS 2305.15076, MetaICL
  2110.15943, CrossFit 2104.08835, test-time training 2411.07279; Lake and Baroni MLC (Nature 2023).
- Counterfactual or invented languages, evaluation only: 2307.02477, 2510.03415, 2501.02825, 2401.12973, 2311.09635,
  EsoLang-Bench 2603.09678, 2605.15607.
- Proposers trained from learner signals: Absolute Zero 2505.03335, SQLM 2508.03682, R-Zero 2508.05004, SPICE 2510.24684, SOAR
  2601.18778 (grounded progress beats intrinsic learnability), Montessori-Instruct 2410.14208, Self-Evolving Curriculum
  2505.14970, MAGELLAN 2502.07709, PAIRED 2012.02096, POET 1901.01753, PLR 2010.03934, OMNI-EPIC 2405.15568, ACES 2310.10692.
  Proposer collapse is named: 2510.27072, R-Diverse 2602.13103, 2604.03472, 2603.13309.
- Disagreement-driven generation is BALD / query-by-committee (1112.5745, 1906.04161); not novel.
- Efficiency measures: Chollet 1911.01547, ARC-AGI-3 2603.24621, ConTinTin 2203.08512.
- Apparently open: weight-level acquisition efficiency on procedurally generated rule systems with rule-axis holdout,
  closed-book, sealed examiner; dose-response in the number of systems at fixed volume; intercept-vs-slope decomposition;
  interpreter-computed rule-coverage descriptors driving a quality-diversity proposer. Frame as multi-task initialisation, not MAML.

## Threats designed against

Intercept masquerading as rate (closed-book primary); generic skill (placebo-dialect arm, K=0 non-discriminating accuracy);
interpolation in generator space (examiner-only axes); dilution (score discriminating items: dialect output differs from
stock); fresh-from-base retraining cannot carry meta-learning (acquire from merged meta-weights); trained proposer likely worse
(demoted to a secondary 2x2); statistical unit is the held-out dialect (12+ dialects); in-context baselines (4B and 27B with
manual and examples); compute accounting (marginal and total with break-even); parametric interpreter with composable axes.

## Design

- Domains: parametric Lua dialect generator, ~12 semantic axes (index base, length rule, concat operator, truthiness of 0 and
  "", modulo sign, and/or return value, default scope, string.sub bounds, for-bound inclusivity, coercion, mixed comparison,
  precedence swap), 2-3 mutated per dialect; learner implements 8 axes, examiner all 12; held-out: 8 seen-axis, 12 unseen-axis
  (primary), 4 Python-hosted (exploratory); every item tagged discriminating or not.
- Meta-training data without a proposer: re-run the verified stock pools (RSI5, RSI7) under each dialect interpreter; programs
  whose output changes are discriminating records.
- Arms at identical records and steps (~16k): B0 base; V stock Lua; P 16 placebo dialects (renames, no semantic change);
  M_N, N in {1,2,4,8,16}; M_4x4000 (diversity vs volume).
- Acquisition on each held-out dialect: same K verified experiences, K in {0,16,64,256,1024}, fresh rank-16 LoRA on the merged
  meta-weights; secondary 2x2 learner x proposer for autonomous acquisition.
- Exam: closed-book primary (dialect tag only), open-book secondary; stock-answer rate on discriminating items.
- Primary criterion (all): (1) learning-curve area on unseen-axis dialects, closed-book, discriminating items, M_16 above the
  better of V and P, hierarchical bootstrap (dialects, seeds, items) lower bound > 0; (2) data-efficiency ratio >= 2 (lower
  bound > 1) at K=64; (3) positive trend of area vs log2 N. Secondary: slope K=0..256, retention, break-even, in-context lines.
- Power: ~12 held-out dialects, 200 discriminating items each, 2 fit seeds, 2 meta seeds for B0/V/P/M_16; fix from pilot 3.
- Cost: ~140 GPU-h (3-4 days on 3 GH200s) plus 1-2 days of pilots; batched examination of ~1400 adapters needed.

## Pilots (in order, each with a kill rule)

- [ ] 1 Discrimination rate (CPU): parametric generator + interpreter; per axis share of pool programs whose output changes and
  base stock-answer rate; axes under 10% discriminating need targeted generation or are dropped.
- [ ] 2 Base learning curve (~3 GPU-h): K-shot curve on 3 dialects; sets K grid and threshold.
- [ ] 3 Go/no-go (~6 GPU-h): M_6 vs V vs P on 4 unseen-axis dialects; go if closed-book area gap >= 4 points and above placebo;
  open-book-only gain means re-scope ("learning to read rule sheets").
- [ ] 4 Continuation interface (~2 GPU-h): merge + fresh LoRA vs continuing the adapter; forgetting.
- [ ] 5 Exploration (~4 GPU-h, scored on CPU): new discriminating archive cells per request for the strategies below.
- [ ] 6 In-context references: 4B and 27B with manual and examples in context.

## Exploration strategy

Differential, coverage-guided quality-diversity proposing: run each proposal on stock and dialect interpreters (an output
difference proves a changed rule fired; an instrumented interpreter names which); a persistent MAP-Elites archive with cells by
fired-rule set (singles and pairs), construct category and size; archive-conditioned prompts (mutate or combine 2-3 elites of the
target cell); cell choice by learning progress with some uniform exploration; a non-LLM novelty floor by AST mutation and
splicing; committee disagreement across adapters only as a within-cell tiebreak; stop on archive saturation, not request caps.
Compare passive filter, differential filter, archive + learning progress, archive + committee at equal requests and admitted
records, judged on downstream learning-curve area.

## Story

A small model that has acquired more invented rule systems needs measurably fewer verified experiences to acquire the next one,
on held-out systems drawn by an examiner from rule axes it never met, examined closed-book, and the gain grows with the number of
systems acquired while placebo and volume controls show none; the whole loop runs without human data because an
interpreter-grounded, coverage-driven proposer keeps supplying novel rule-exercising programs where passive failure filtering
collapses.
