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

## Deviations (dated; all before any F or M verdict existed)

- 2026-11-15 ~11:30, on an independent referee's review, before any verdict of F or M was produced or read:
  1. **M's primary analysis is lineage-level.** The registered M1 compares adapters of different seeds item by item, which confounds
     the treatment with seed-to-seed variation (RSI4's generation-3 champions range from 192 to 241 items right on the RSI4 exam), and
     its pairing by index is arbitrary. The primary M analysis is now the difference between the mean share of F's items right of the
     four RSI4 generation-3 champions and that of the four M controls, with a 95% interval from a bootstrap that resamples templates and
     lineages (seeds) jointly, and an equivalence margin of 0.02 (two percentage points of F's items): **generation 3 better** (lower
     bound above 0), **equivalent** (interval within plus or minus 0.02), **control better** (upper bound below 0), otherwise
     **inconclusive**. The original M1 is kept and reported as a secondary analysis.
  2. **Arms added.** M-900: four lineages as M with a frontier of 900 base-proposed programs (225 steps, generation 2's amount),
     so that the matched-data controls trace a data-scaling curve beside RSI4's generational curve. S (seed null): each RSI4
     generation-3 champion's training set retrained from the base with a new training seed, measuring the adapter-to-adapter
     variation that item-level comparisons between adapters carry. A control on unfiltered base proposals (2100 records without
     frontier selection) is not run: the registered lineage build cannot draw them.
  3. **F reports more, judged the same way.** Beside F1 and F2: the exact construction of F2's interval (stated in the launch record);
     the lineage-averaged step gain with its template-clustered interval; strata of F by template (families whose name also occurs in
     the RSI4 exam vs new ones) and by the base's answer (base right vs base wrong); per-template aggregate right counts; and a
     similarity audit (for each F item, its nearest program among everything any examined candidate trained on, by normalised
     token-set Jaccard, reported as a distribution).
  4. **Descriptive baselines on F** (no criterion): the base with thinking on; the base's majority answer over 4 samples at
     temperature 1.0; and Qwen/Qwen3.8-27B-FP8 zero-shot under the same protocol.
- 2026-11-15 ~11:30, before any F or M verdict for the lineage concerned existed (m0 had handed off no candidate):
  5. **A frontier program longer than the training row.** Control lineage m0 (`run-control.sh m0 0 8785991200924142667`, RSI4's
     `lineage-86f13672a`) found 2184 frontier programs among 2400 proposals after 3584 requests and selected 2100, and its training
     job was refused before submission: `g1r1-mispredicted.jsonl:394 (prop-e33f535a2d3b8ac2) is 1217 tokens`, longer than the
     row of 1024 tokens (RSI4's `max_block`). The registered driver has no rule for such a record, so the lineage failed outright;
     RSI4 never proposed one. **Rule:** before a round's training job is submitted, each candidate record is measured as the
     trainer measures it (brain's `ExampleMeter` through `splinter_model::measure::RecordMeter`, the same chat line, thinking off,
     unpacked; that record measures 1217 tokens, and the line measured is byte-equal to line 394 of the refused dataset), and the
     round trains on the first 2100 frontier programs, in the order proposed, whose records fit; the skipped ones are recorded by
     id and length in the round record (`frontier.overlong`), and a round with fewer than 2100 that fit fails with that reason.
     With no over-length record the selection, the training request and the round record equal the registered binary's, so the
     rule changes nothing where RSI4's behaviour was defined. **Build:** splinter `05729ef09fa3bb772bc2a37881ee5ac2f3820c8e`
     (tag `rsi4f-lineage-05729ef09`: RSI4's `86f13672a` plus this rule alone; the same rule is main's `019239f10`), against the
     pinned sven `fc99a2f1` and brain `79687164` crates, binary `lineage-05729ef09`, SHA-256
     `37da9a795ebe7afa99d5ef967d70e69e3ee16cc497a8effc4df24f53ca552b02`; script `~/project/shared/rsi4f/run-control-overlong.sh`
     (SHA-256 `e31f94d7cdcd9e759476cdb296f4b72c34a4cda8f43f73ecaf330bba3b8f3a61`), `run-control.sh` with only the binary and the
     stated splinter commit changed. **Scope:** used only for lineages that hit the over-length refusal, resumed from their own
     journals with the same seed, settings and directory (the frontier journal is reused; nothing is proposed again): m0 now, and
     any other M or M-900 lineage that fails the same way later, each listed here when it is resumed.
6. **Proposer state removed mid-run by a housekeeping script (2026-11-15 ~12:17, before any verdict of n0 or n3).** A script
   meant to pack each finished lineage's proposer state into one archive fired early, when n1 finished, and removed the state
   directories of two running lineages: n3 had just started and failed at once ("no task set ... in the store", nothing
   proposed or trained); n0 had finished proposing and was training. Both archives were verified complete (n3: 1636 entries,
   n0: 10605). n0's state was restored from its archive while it trained; n3's state was restored and n3 relaunched with the
   same script (`run-control-900.sh`), seed, binary and settings, starting from its journals. The packing script was stopped
   and is not used again; no other lineage was affected.
