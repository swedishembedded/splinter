<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# generic-learning

Scope: the generic learning loop and its plan E0-E5: the provenance invariant on training targets, min-cost selection, branch values and decisive-step search, start-state practice, the importance-ratio diagnostic, per-action advantages, the problem-state candidates, and the retirement of the prose-reading filters. Items GL-P1 to GL-P7, the pre-registered plan and the decision rule.

Source: the 2026-11-21 training-methods review (read-only advisory; its numbers are the splinter ledger's, except the E2a/E2b exam results and item 24b's in-family transfer result, which were taken from the lane brief and are pending as problem-state PS-2). The same review is recorded in the other three repos' `generic-learning.md`; the full plan E0-E5 is in splinter's.

## The generic loop and its invariants

The loop needs only the verifier, the tools, a token budget and state forks, and routes each task by its measured pass rate under the current policy, never by task content. Every repo holds the same four rules:

1. **Same observation.** A token is a training target only if it was sampled in a context the student will also see. Privileged sources (a teacher, a hinted self, an oracle solution) supply states, not text.
2. **Reasoning length is a budget.** A trial passes only if the verifier passes within `B` completion tokens and `N` turns. Native reasoning mode, one mode and one budget for teacher capture, training data and every exam arm; length is controlled by the budget and by lowest-cost selection among verified passes, never by prompt text.
3. **Evaluation that can see the effect.** At least 60 tasks over at least 20 families, a reference arm between about 0.15 and 0.85, three tiers (unseen instance, held-out variant, held-out family; only the third supports a transfer claim), wall clock an infrastructure backstop.
4. **No string matching.** No filter reads the prose of a trace or the content of a task.

## Capabilities

| Id | Capability | Status | Acceptance | Depends on |
|---|---|---|---|---|
| GL-P1 | The same-observation invariant on training targets, a typed property of a record checked structurally | gap | a record with a token sampled under a hint or knowledge page is refused by `dataset build`; a spec covers it | sven GL-S3 |
| GL-P2 | Min-cost selection: among a task's verified passes keep the lowest total completion tokens within the student's budget | partial: `--max-passes-per-task` ranks by least reasoning per action | ranking by total cost; E2 | bench GL-B1 |
| GL-P3 | Branch values and decisive-step search: `V(s_t)` as the pass rate of k forked continuations, binary search over turns, about `log2(T) x k` short rollouts | gap (forks exist) | shared-prefix preference pairs at the decisive step; E5 | sven GL-S2, bench GL-B3 |
| GL-P4 | Per-action advantages `V(s_{t+1}) - V(s_t)` in the PG view | partial: the view already writes one record per action | advantages carried on the records | GL-P3 |
| GL-P5 | Start-state practice with fork depth as a curriculum variable (path-guided unhinted resampling); the fork moves earlier as the pass rate from `s_t` rises | planned ("signal per GPU-hour" item 3), not built | E4's yield and effect | sven GL-S2, bench GL-B3 |
| GL-P6 | Importance-ratio diagnostic `pi_unhinted(trace) / pi_hinted(trace)`, reported as effective sample size per task; a diagnostic, never a gate | gap | reported per task, calibrated on unhinted passes | brain GL-R2 |
| GL-P7 | E0-E5 entered as problem-state candidates, one variable each; E2a/E2b verdicts recorded (PS-2) | candidates landed 2026-11-21 in `examples/problem-state/problems.toml`; PS-2 open | `campaign problems check` accepts the document | none |

If no continuation passes from any depth, the task is outside the student's support: a measured fact, and the task goes to the teacher.

## Plan E0-E5

All on Qwen3-4B with A2's LoRA recipe, two seeds, family-clustered `experiences compare-arms --families`, Holm over the pre-registered contrasts, target +0.05. The vim champion is v1 (thinking-mode 27B distillation), the only arm with a supported held-out gain. A contrast that cannot reach the target by the pre-computed minimum detectable effect is not run. GPU-hours are the review's estimates.

| # | Experiment (one variable) | Pre-registered success | GPU-h | Problem-state candidate |
|---|---|---|---|---|
| E0 | **Instrument.** Token and turn budgets enforced; budget set offline from existing passing trials (about 1.5x the base's p90). Re-run base and v1 on the 16 validation tasks at two loads | pass rates agree across loads within the A/A band; model-exhaustion share reported | 1-2, after the sven and brain fixes | `overthink.token-turn-budget` |
| E1 | **Leave-families-out on vim train.** Three folds over the 14 rungs; retrain v1's recipe on the other rungs' existing teacher traces; examine on held-out rungs x 8. Variable: none (measures the champion) | pooled held-out-family gain over base at least +0.05, interval above 0. If falsified, vim distillation does not cross families and breadth is the bottleneck | 3-4 | `distill-transfer.leave-families-out` |
| E2 | **Budget-matched selection.** Curation keeps verified teacher passes within the student budget, lowest cost first, from the existing 579 passes. No prompt, no new teacher run | at budget B: gain over v1 at least +0.05, interval above 0; at 4B non-inferior at 0.05; exhaustion share within 10 points of base | 2-3 | `distill-transfer.min-cost-selection` |
| E3 | **Breadth dose-response.** Number of training families (4, 8, 12 rungs) at equal record count, same held-out rungs | monotone, with 12 minus 4 at least +0.05 and interval above 0. Decides whether to fund the environment factory over algorithms | 3-4 | `distill-transfer.family-breadth` |
| E4 | **Path-guided unhinted resampling** on tasks the champion never solves: add these suffix records to the champion's data, trained from base. Control: equal-count extra teacher records | yield at least 30% of never-solved tasks get a verified unhinted suffix; from-scratch pass@4 on them +0.10, held-out families +0.05 over control, retention non-inferior at 0.10 | 4-5 | `4b-data.path-guided-resampling` |
| E5 | **Branch-value step credit** on 0 < p < 1 tasks: shared-prefix pairs at the decisive step against RFT at equal rollout tokens | held-out +0.05 over the RFT control with interval above 0, retention non-inferior. Review's prior about 0.3 | 8-10 | `trajectory-credit.branch-value-pairs` |

E0 to E3 need no new method and fit in one afternoon on four cards.

**Decision rule: E4 and E5 do not start before E1 shows the instrument can detect +0.05.** If E1 is falsified, breadth (E3, bench's environment factory) is funded before any algorithm.

## Stop doing (splinter)

- Any filter that reads the prose of a trace (word lists, answer-presence, command-explained, rewrite passes): retired by `engineering-cleanup.md` SC-9; items 24c and 24d of `experience-to-competence.md` are superseded.
- Training on any token sampled with a hint or knowledge page in context.
- Turning thinking off to control length: knowledge 203 is superseded pending E2.
- Further rounds of teacher-free GRPO or DPO on a few hundred tasks with trajectory-level credit.
- Reading results from the 14-task vim exam or from wall-clock-limited runs; claiming transfer from in-family held-out instances.

Kept as generic: `--distinct-calls` (structural equality of calls) and `--only-actions` (a tool allow-list).

## E0-E3 runs (lane EXP; pre-registered 2026-11-21 17:30, before any trial or training)

**Instrument, shared by every arm of E0-E3.**
- **Builds (frozen):** brain-bcd67e5ad (per-turn thinking budget that always closes, exact reasoning counts) for serving and
  training; sven-fe9cda90 (enforced token and turn caps); bench 3403023a (origin/main, `[budget]` since 023814be) in a
  pinned worktree; splinter-c2a7d877e for datasets, training and `compare-arms` (origin/main differs from it only in
  roadmap, an example and a test).
- **Mode:** thinking on, no appended instruction, brain's per-turn `thinking_budget = 2048` (E2c's), Qwen3 thinking
  sampling T 0.6 / top-p 0.95 / top-k 20, run seed 20261121, sven's brain driver, one bench gateway per arm.
- **Budget B (set offline):** the base's 78 passing thinking-capped trials (lane VIM `xt-base-g1/g3`, `e2c-base-*-lost2`,
  same mode) spent p90 10,076 completion tokens and 6 turns; B = 1.5 x p90 = **15,114 completion tokens and 9 turns**,
  the same for every arm and experiment (`[budget]`, wall clock factor 4, a backstop counted as infrastructure). A
  `budget_exhausted` trial is a model fail; the exhaustion share is reported per arm.
- **Data:** the existing 27B thinking-mode teacher passes behind v1 (VSFT t1-t6 and t34: 1,192 graded, 579 passed),
  curated as v1 (`sft-step --pack --only-actions vim-vim --max-block 16384`); selections read verdicts, rungs and token or
  turn counts only. No new teacher run, no prompt change.
- **Training:** v1's recipe (A2's: Qwen3-4B bf16 base, LoRA r16/alpha 32, lr 2e-4, no weight decay, 4 records per step,
  monitoring every 14 steps, patience 4), steps = round(0.4 x records) (v1: 576 -> 230, about two epochs), seeds 1 and 2.
- **Families** are the 14 difficulty rungs of the train side (`bench_difficulty`); every trials line carries its task's rung
  as `family`.
- **Analysis:** `bench trials` per job, then `splinter experiences compare-arms --trials ... --interval-seed 20261121`,
  two-sided sign test, Holm over each experiment's pre-registered contrasts, task and family-clustered 95% intervals; the
  two seeds of a student are pooled as one arm. Verdicts enter the canonical problem state
  (`~/project/shared/campaigns/experience-to-competence/`) with `update --compare`, experiment ids `exp-e0` to `exp-e3`.
- **Placement:** nodes n467 and n445 (4 GH200 each); each server holds one load, and every arm of a contrast runs in the
  same slot on the same server(s) at equal concurrency.

**E0 (instrument; candidate `overthink.token-turn-budget`).** Base, v1s1 and v1s2 (VSFT adapters) on v1's 16 held-out
validation tasks x 16 attempts, at two loads at once on n467: low (gpu0, 4 trials in flight per arm, 12 in all) and high
(gpu1, 32 per arm, 96 in all). Contrast per arm: high minus low (reference the arm at low load), 3 contrasts, Holm.
**Success:** for every arm the 95% task interval of high minus low contains 0 and the Holm p is at least 0.05 (the A/A
band); the exhaustion share is reported per arm and load. Secondary: v1 minus base at each load.

**E1 (leave-families-out; candidate `distill-transfer.leave-families-out`).** Folds of held-out rungs a {1,4,7,10,13},
b {2,5,8,11,14}, c {3,6,9,12}. Per fold v1's recipe is retrained, seeds 1 and 2, on every teacher pass of the other rungs
(datasets of 402, 343 and 403 records), and base and both seeds are examined on all tasks of the held-out rungs x 8
(58, 65 and 45 tasks; 168 tasks over 14 families pooled). Primary contrast: pooled students minus base. **Success:**
difference at least +0.05, family interval above 0, Holm p below 0.05. If falsified, vim distillation does not cross
families and breadth (E3, bench's environment factory) is funded before any algorithm.

**E2 (budget-matched selection; candidate `distill-transfer.min-cost-selection`).** The single variable is curation:
per fold, of E1's training passes only those within the student budget (at most B completion tokens, N turns, and every
turn's reasoning within the 2,048-token per-turn budget), at most 4 per task, lowest total completion tokens first
(317, 279 and 316 records); trained as E1 (seeds 1 and 2). Examined in E1's slot on the same held-out tasks x 8 against
E1's students as v1 (pooled). **Success at B:** gain over v1 at least +0.05, family interval above 0, Holm p below 0.05;
exhaustion share within 10 points of base. **At 4B** (60,456 tokens, 36 turns), the same E2 and E1 students x 4 attempts,
in their own slot: non-inferior at 0.05 (lower end of the family interval of E2 minus v1 above -0.05). E2's arm needs
both.

**E3 (breadth dose-response; candidate `distill-transfer.family-breadth`).** Seven folds, each holding out two rungs
({1,8}, {2,9}, {3,10}, {4,11}, {5,12}, {6,13}, {7,14}; all 168 tasks over 14 families). Per fold 4, 8 and 12 training rungs,
nested seeded subsets (seed 20261121) of the 12 others, at an equal 122 passes drawn uniformly (seeded) from those rungs'
passes (built 119-122 records), seeds 1 and 2; examined on the fold's held-out tasks x 4 per seed, the three doses in one
slot. Contrasts r8 minus r4 and r12 minus r4 (Holm over 2). **Success:** monotone point estimates (r4 <= r8 <= r12), with
r12 minus r4 at least +0.05, family interval above 0 and Holm p below 0.05.

Selections and dataset ids: `lane-exp/data/{selections,design}.json` and `datasets.jsonl`; script `lane-exp/scripts/select.py`.
