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
