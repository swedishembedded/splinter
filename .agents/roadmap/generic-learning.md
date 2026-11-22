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
| GL-P8 | Human-guided trials (bench's `task play`) as a privileged source for GL-P5: fork the student at a verified guided path's states WITHOUT the person's messages and keep the verified unhinted suffixes. Never supervise a token sampled with a person's message in context | partial: ingestion keeps `extra.sven.origin: human`; `dataset build` withholds guided experiences (`human_context`); `compare-arms` refuses guided trials; the resampling is the `path_guided` intervention of `experiences repair` (`PrivilegedPaths`, `unguided`, `fork_points`; spec `tests/it/path_guided.rs`: no person's text reaches the student's requests, the kept repair has no human turn). No sven change was needed: the SDK fork takes the cut, unguided transcript. Open: (1) a compiled dataset's run (export plus a report naming the dataset) is repaired through the examiner's compiled-task session under the image's Sven configuration and its MCP tools (spec `tests/it/compiled_repair.rs`); on vim, a recorded call whose tool reply names a random scratch path cannot be replayed (bench knowledge 076, about half of `train-k1`'s trials) until the dataset's tool is fixed; (2) done: a teacher's passes are privileged paths when the run names their set (`--paths`); the forks are the path's open states (its verifier does not pass there yet), the earliest first, samples beyond them try them again, and a continuation without a tool call is refused (knowledge 211); (3) a labelled diagnostic view keeping the person's messages, refused by default | a guided trial's verified states yield unhinted records whose requests hold no person's text (met on the scripted spec); E4's yield on never-solved tasks | bench GL-B3 (compiled datasets), bench GL-B4 |
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
| E4 | **Path-guided unhinted resampling** on tasks the champion never solves: add these suffix records to the champion's data, trained from base. Control: equal-count extra teacher records | yield at least 30% of never-solved tasks get a verified unhinted suffix; from-scratch pass@4 on them +0.10, held-out families +0.05 over control, retention non-inferior at 0.10 | 4-5 | `4b-data.path-guided-resampling`; **gate failed on vim** (7 of 73, knowledge 211): move to a multi-step domain |
| E5 | **Branch-value step credit** on 0 < p < 1 tasks: shared-prefix pairs at the decisive step against RFT at equal rollout tokens | held-out +0.05 over the RFT control with interval above 0, retention non-inferior. Review's prior about 0.3 | 8-10 | `trajectory-credit.branch-value-pairs` |

E0 to E3 need no new method and fit in one afternoon on four cards.

**Decision rule: E4 and E5 do not start before E1 shows the instrument can detect +0.05.** If E1 is falsified, breadth (E3, bench's environment factory) is funded before any algorithm.

## External research report (2026-11-21): mapping and additions

A literature report pasted by the user. Its citations dated 2026 are not verified here, so its effect sizes are not used. Where it matches the plan above:

- **Generalization ladder and family-level splits** are rule 3 and E1 (held-out family, +0.118 [+0.021, +0.220]).
- **Verified branch repair and prefix branching** are E4 (GL-P5, `path_guided`) and E5 (GL-P3). A branch starts from a restored state (bench GL-B3), never a replayed history it cannot reproduce.
- **Matched-state action preferences** are E5's shared-prefix pairs.
- **Do not train on answer-conditioned rationales** is rule 1. Its "E4 rationalization filter" does not apply: the prose filters are retired (SC-9), and our E4 is resampling, not a filter.
- **Curriculum at 0 < p < 1** is the loop's routing by measured pass rate.

New, entered as gap items. Each one tests one variable and is gated like E4/E5:

| Id | Capability | Status | Acceptance | Depends on |
|---|---|---|---|---|
| GL-P9 | **Transition records from every trial, failed ones included.** An experience view whose target is the environment's next observation given the state and action. The tool's own output is the target, never a model's explanation of it. It passes rule 1, because the observation was produced by the environment, not sampled under a hint. Trained as an auxiliary loss beside the policy records, with observation tokens excluded from any policy-gradient term | gap | E6 | brain GL-R (per-record loss weights) |
| GL-P10 | **Group-signal diagnostics for every PG/RL round:** share of all-fail and all-pass groups, advantage variance, KL, entropy, update norm, and train versus held-out pass. These are reported, never gated, so that the next "no transfer" (R19) can be attributed | gap | printed per round by `splinter` from the trainer's records | brain per-sample metrics |

| # | Experiment (one variable) | Pre-registered success | Gate |
|---|---|---|---|
| E6 | **Early-experience transitions.** v1's recipe plus GL-P9 records from the same trials (failed trials included), against v1 plus an equal token count of extra policy records | held-out families +0.05 over control, interval above 0, retention non-inferior at 0.10 | E1 met; runs after E4 so that only one new data class enters at a time |
| E7 | **Knowledge necessity (bench GL-B6).** Effect of access to the knowledge base: success with the base minus success without it, on tasks whose rule exists only in the base, under correct, absent, distractor, contradicting and updated evidence | the effect of access is above 0 with its interval, and the distractor and updated-evidence arms are not worse than the absent arm. Until this holds, the A2 knowledge-base result (barely consulted) is uninterpretable | bench GL-B6. **Done 2026-11-22 (lane KBN):** pre-registered in bench aa2a3073 (generic-learning.md "GL-B6 run"; after the 2026-11-21 advisory: pages preloaded, harm read against the correct arm, a 27B validity pass); results bench ee8f6ea0, knowledge 080. Base 4B, 24 tasks x 4, company-clustered 95%: correct - absent **+0.115 [+0.042, +0.198]**, updated - absent +0.052 [+0.010, +0.104], distractor - correct **-0.115 [-0.198, -0.042]**, contradicting - correct -0.073 [-0.177, +0.031]; the 27B 64/64 with the page, 0/32 without. Access helps causally; a second plausible page erases it; the 4B mostly fails to apply the rule, not to find it. The registered criterion as written (not worse than absent) holds but is vacuous at the floor |

## Stop doing (splinter)

- Any filter that reads the prose of a trace (word lists, answer-presence, command-explained, rewrite passes): retired by `engineering-cleanup.md` SC-9; items 24c and 24d of `experience-to-competence.md` are superseded.
- Training on any token sampled with a hint or knowledge page in context.
- Turning thinking off to control length: knowledge 203 is superseded pending E2.
- Further rounds of teacher-free GRPO or DPO on a few hundred tasks with trajectory-level credit.
- Reading results from the 14-task vim exam or from wall-clock-limited runs; claiming transfer from in-family held-out instances.

Kept as generic: `--distinct-calls` (structural equality of calls) and `--only-actions` (a tool allow-list).

## E0-E3 runs (lane EXP; pre-registered 2026-11-21 17:17 (commit c26c073bb), before any training (17:18) or trial (17:24))

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

**Deviations (dated 2026-11-21 18:16, before any E1 or E2 trial and before the E3 rerun's analysis).**
- **E3's first exam slot was invalid and is discarded unread.** Folds p1-p3 (18 adapters) ran on one exam server with
  brain's `--adapter-slots 8`: brain served the 8 resident adapters and starved the other 10 arms (0 trials graded in 8
  minutes, 6 requests each waiting) without an error. It was stopped at 17:59 and archived
  (`lane-exp/archive/e3a-invalid-adapter-starvation.tar.zst`). Eight E3 trainings were cancelled at 17:50 for another
  session's window and retrained from the same datasets and seeds.
- **Serving topology for every E1, E2 and E3 exam:** one fold per exam server, each server with `--adapter-slots 24`
  (every arm's adapter resident), all arms of the fold on that server at equal concurrency (E3 6, E1/E2 8 trials in
  flight per arm). No router.
- **Rounds:** each fold's exam runs as consecutive slots of one attempt per task (E3 4 rounds, E1/E2 8 rounds, the 4B
  check 4 rounds), every arm launched together in each round, each round packed before the next (inode quota). The
  rounds of one fold are pooled; nothing else changes.
- **Tooling:** the vim train side is a compiled dataset, so each arm is a plain `harbor run` with sven-fe9cda90 (which
  enforces the profile's `[budget]`) and `--agent-timeout-multiplier 4`, adopted with `bench adopt`, which now records the
  budget (bench 0e574fdf), so `bench trials` classifies exhaustion. Gateway usage is not attributed per trial in this
  path, so bench's own token check reads 0 tokens and exhaustion rests on Sven's enforced stop and the turn count.

**E0 result (read 2026-11-21 18:52; `compare-arms`, splinter-c2a7d877e, interval seed 20261121; 256 graded trials per
arm and load, no infrastructure errors).** The instrument holds: **supported**.

| arm | pass@1 low (4/arm) | pass@1 high (32/arm) | high minus low | task 95% | family 95% (8 rungs) | p two-sided | Holm | exhaustion low / high |
|---|---:|---:|---:|---|---|---:|---:|---|
| base | 0.180 | 0.203 | +0.023 | [-0.023, +0.070] | [-0.016, +0.072] | 0.754 | 1.0 | 8.2% / 9.4% |
| v1s1 | 0.434 | 0.445 | +0.012 | [-0.035, +0.055] | [-0.029, +0.056] | 0.180 | 0.539 | 51.2% / 52.0% |
| v1s2 | 0.441 | 0.438 | -0.004 | [-0.086, +0.086] | [-0.091, +0.069] | 1.0 | 1.0 | 52.0% / 49.2% |

Every interval of high minus low contains 0 and no Holm p is below 0.05, so a budgeted pass rate does not move with
server load (E1's 71% wall-clock timeouts are gone). Secondary, v1 (pooled seeds) minus base on these 16 in-family
validation tasks: +0.258 [+0.111, +0.410] at low load and +0.238 [+0.068, +0.402] at high load (sign p 0.057 and
0.013). Half of v1's trials run out of B: v1 learned the teacher's length, which the budget now counts as a fail. A few
exhausted trials record 0 turns (2-7 per v1 arm): Sven stopped a looping reasoning block and charged it the whole
budget. Reports: `lane-exp/logs/e0-*.compare.json`. Verdict entered as `exp-e0` in the canonical problem state.

**Invalid, discarded unread (infrastructure, 2026-11-21 18:05-18:47).** E3 folds p1-p2 (4 rounds each) and E1/E2
folds a-b (rounds 1-3) ran on two n445 servers started with `--adapter-slots 24`. With that bank brain sized Qwen3-4B at
97 GB, so it wrote its ready file and then answered every request with `500 failed_to_load` (brain knowledge 546).
bench scored each of the 2,358 trials as a `graded` fail with 0 turns (bench knowledge 077). They are archived
(`lane-exp/archive/n445-invalid-unservable-model.tar.zst`) and not analysed. The E3 trainings of fold p7 hung client
side and were stopped. **Rerun protocol (before any rerun trial):** exam servers use `--adapter-slots 8` with at most 6
adapters per server (one fold per server). Before any trial, each slot sends one real request per arm, and any
failure stops it. After each round, any arm with a 0-turn trial carrying an upstream error stops the fold. Everything
else is as registered.

**Rerun placement (dated 2026-11-21 19:05, before any rerun trial).** E0 showed that budgeted pass rates do not depend
on load, so the reruns use more concurrency per arm: E1/E2 16 trials in flight per arm, E3 12, the 4B check 16. Each
E1/E2 fold runs on two identical exam servers, rounds 1-4 on one and rounds 5-8 on the other, every arm on both. Each
E3 fold runs on one server. E3 fold p7's six trainings are rerun one at a time from the same datasets and seeds.

**E1 result (read 2026-11-21 19:58; `compare-arms`, interval seed 20261121; 168 held-out tasks over 14 rungs x 8 per
arm, students' seeds pooled; no infrastructure errors, no timeouts).** Supported: **vim distillation crosses families.**

| arm | trials | pass@1 | minus base | task 95% | family 95% (14 rungs) | better / worse / ties | p two-sided (Holm) | exhaustion |
|---|---:|---:|---:|---|---|---|---:|---:|
| base | 1,344 | 0.235 | | | | | | 10.1% |
| v1 recipe, other rungs (pooled) | 2,688 | 0.353 | **+0.118** | [+0.062, +0.173] | **[+0.021, +0.220]** | 91 / 48 / 29 | 0.0003 | 55.9% / 59.2% |
| seed 1 | 1,344 | 0.366 | +0.131 | [+0.076, +0.186] | [+0.037, +0.230] | 83 / 42 / 43 | 0.0006 | 55.9% |
| seed 2 | 1,344 | 0.340 | +0.105 | [+0.046, +0.163] | [+0.002, +0.215] | 79 / 48 / 41 | 0.0075 | 59.2% |

The pooled gain on rungs never trained on is +0.118, its family interval is above 0, and both seeds agree. The base
sits at 0.235, inside the 0.15-0.85 band. **Decision rule:** E1 shows that this instrument detects a +0.05 effect, so
E4 and E5 may start. The students exhaust B in 56-59% of trials against the base's 10%. Item 25's exam-family null
(knowledge 202) was a wall-clock artefact plus a 14-task exam, not an absence of transfer. Report: `lane-exp/logs/e1.compare.json`;
verdict `exp-e1` entered in the canonical problem state (`distill-transfer.leave-families-out` supported, belief 0.40
to 0.55).

**E2 at B (same slot as E1).** Min-cost selection against v1's curation, pooled: +0.037, task interval
[+0.010, +0.064], family interval **[-0.021, +0.105]**, 75 / 58 / 35, p 0.165. Seed 1 +0.041 [-0.008, +0.105], seed 2
+0.032 [-0.039, +0.108]. The exhaustion share is 50.5% against the base's 10.1%, 40 points apart, so that criterion
fails. The gain is not supported, and +0.05 is not ruled out. The 4B check is running; the verdict is entered after it.

**E2 result (read 2026-11-21 21:05).** **Not supported**; the problem-state rule gives *inconclusive* on the gain.
- **At B** (E1's slot): +0.037, family interval [-0.021, +0.105], p 0.165 (above). Exhaustion 50.5% against the base's
  10.1%: the "within 10 points of base" criterion fails. Min-cost curation cut v1's exhaustion only from 57.6% to 50.5%,
  because a student trained on the cheapest passes still reasons at the teacher's length.
- **At 4B** (60,456 tokens, 36 turns; 168 tasks x 4 per seed, own slot): e2 0.514 against v1 0.486, +0.029, task
  interval [-0.010, +0.070], family interval [-0.025, +0.088], 67 / 54 / 47, p 0.275. Exhaustion 11.5% against 16.1%.
  Nominally non-inferior at 0.05 (lower end -0.025), but **22-27% of trials hit the wall-clock backstop**
  (`infra_timeout`, 600 s x 4: e2 301, v1 361 of 1,344 each). The backstop was not scaled with the 4x token budget, so
  the check excludes long trials unevenly and is weak evidence. A rerun should set `wall_clock_factor` to at least 16.
- Verdict `exp-e2` entered from `logs/e2.compare.json` (belief unchanged at 0.45). Reports:
  `lane-exp/logs/e2*.compare.json`, `e2-4b.compare.json`.

**E3 result (read 2026-11-21 21:05; 7 folds of 2 held-out rungs, all 168 tasks over 14 rungs, 4 attempts per seed;
1,344 graded trials per dose, no infrastructure errors).** **Falsified: breadth at an equal 122 records does not help.**

| dose | pass@1 | minus r4 | task 95% | family 95% | better / worse / ties | p two-sided | Holm | exhaustion |
|---|---:|---:|---|---|---|---:|---:|---:|
| r4 | 0.287 | | | | | | | 48.3% |
| r8 | 0.276 | -0.010 | [-0.039, +0.019] | [-0.049, +0.026] | 49 / 57 / 62 | 0.497 | 0.959 | 49.9% |
| r12 | 0.281 | -0.005 | [-0.033, +0.022] | [-0.041, +0.028] | 45 / 53 / 70 | 0.480 | 0.959 | 52.8% |

Not monotone, and r12 minus r4 has an upper end of +0.028, below the +0.05 target, so a breadth effect of that size is
ruled out at this record count. Descriptive only (the base ran in E1's slot, not this one): every dose sits about
+0.04 to +0.05 above E1's base (0.235), against +0.118 for E1's 343-403 records. In this range the amount of verified
teacher data moved held-out families more than the number of training rungs did. Verdict `exp-e3` entered (belief
0.40 to 0.20). Report: `lane-exp/logs/e3.compare.json`.

**What E0-E3 decide.** The budgeted instrument works (E0), and vim distillation transfers to held-out families
(E1, +0.118), so E4 and E5 may start. Neither min-cost selection (E2) nor breadth at a fixed small record count (E3) is
the lever measured here. Every distilled student exhausts the budget in about half of its trials, so length remains the
students' main failure mode. **GPU-hours:** about 25 across n467 and n445: 1.6 for E0 at low load and 1.4 at high, 4.6
for training, and about 14 for the E1/E2/E3 exams and the 4B check. About 3.5 of the 25 went to the two invalid exam
slots (adapter starvation; an unservable model).

**E2 4B rerun (pre-registered 2026-11-21 21:10, before any of its trials; at the coordinator's request).** This
replaces the first 4B check, which lost 22-27% of its trials to the wall-clock backstop.
- Arms, tasks and attempts are the first check's: E1's and E2's students (seeds pooled) on each fold's held-out tasks
  x 4 attempts per seed. Budget 4B = 60,456 completion tokens and 36 turns; per-turn thinking budget 2,048.
- The backstop scales with the budget: `wall_clock_factor = 16` and Harbor's agent limit x 16 (600 s -> 9,600 s).
- The build is brain-bcd67e5ad, as at B, so the serving build matches every other E2 arm.
- On n424 (local), one fold per exam server (gpu1 a, gpu2 b, gpu3 c, `--adapter-slots 8`), 16 trials in flight per arm,
  4 rounds of one attempt each, with per-arm preflight and the round check.
- Analysis as registered: e2 minus v1, `compare-arms`, family interval; non-inferior at 0.05 when its lower end is
  above -0.05. The `infra_timeout` count is reported per arm.

**E2 4B rerun stopped (2026-11-21 21:55, after rounds 1-2 of folds a and c and round 1 of b).** Scaling the backstop
did not help: Sven's own run limit comes from the task's 600 s (480 s after its margin), not from the run's wall-clock
factor (bench knowledge 078, GL-B8), so 26% of trials still ended as `infra_timeout` (e2 142, v1 138 of 542 each). The
completed rounds, for the record only: e2 0.506 against v1 0.492, +0.021, family interval [-0.029, +0.086]. No revised
verdict: `exp-e2` stays *inconclusive* (at B: +0.037 [-0.021, +0.105], exhaustion 50% against 10%). The 4B
non-inferiority check needs GL-B8 fixed first (fixed in bench 0d8f13de). Interrupted rounds are archived
(`lane-exp/archive/e2-4b-rerun-stopped-rounds.tar.zst`). Ran on n424 gpu1-3 (about 2.2 GPU-h); n445 was not used for it.

**E2 4B rerun 2 (pre-registered 2026-11-21 22:20, before any of its trials).** Identical to the stopped rerun except
for bench 0d8f13de: Sven's run limit is now 600 s x 16 less its margin, 9,480 s. Same arms, tasks, budget (60,456
tokens, 36 turns, factor 16), brain-bcd67e5ad, n424 gpu1-3 (one fold per server, `--adapter-slots 8`), 16 trials in
flight per arm, 4 rounds of one attempt, preflight and round check. The analysis is the registered one: e2 minus v1,
non-inferior at 0.05 when the lower end of the family interval is above -0.05. The verdict on
`distill-transfer.min-cost-selection` is revised from it together with the B result: supported only if B's
gain criterion also held (it did not), so a non-inferior 4B result leaves `exp-e2` inconclusive and an inferior one
falsifies the candidate.

## E8 (lane EXP): teacher capture under the student's budget (pre-registered 2026-11-21 23:50, before any capture, training or trial)

**Question.** Does teacher data captured under the student's own budget lower the students' exhaustion and raise their
held-out pass rate? E1's students fail mostly by running out of B (54-55% of trials, almost all on tokens), not by
wrong answers within B (7%). Their teacher had no per-turn cap and no budget. The single variable is the data source.
- **Teacher capture.** Qwen3.8-27B-FP8 on brain-bcd67e5ad in the students' mode (thinking on, per-turn
  `thinking_budget = 2048`, T 0.6 / top-p 0.95 / top-k 20, run seed 20261121), with budget B (15,114 completion
  tokens, 9 turns, factor 4), sven-fe9cda90 and bench 0d8f13de.
  - All 168 vim train tasks x 4 attempts, split into three task thirds, one per card on n445 gpu1-3 (one replica per
    card, `--max-batch 128`).
  - Four rounds of one attempt each, packed as they finish.
  - Every counted pass is within B by construction (an exhausted trial is a fail). All passes are kept, curated as v1
    (`sft-step --pack --only-actions vim-vim --max-block 16384`).
- **Training (fold a first; held-out rungs 1, 4, 7, 10, 13).** v1's recipe, seeds 1 and 2, steps = round(0.4 x records).
  - Arm **ub**: the under-budget passes of fold a's training rungs (N records).
  - If N is below E1's 402, arm **v1m** trains on N records drawn uniformly (seed 20261121) from E1's fold-a
    selection, so the record count is matched.
- **Exam.** E1's protocol: fold a's 58 held-out tasks x 8 at B, 16 trials in flight per arm, one slot. Rounds 1-4 run
  on one exam server and rounds 5-8 on another, every arm on both, with per-arm preflight and the round check.
  - Arms: ub s1/s2, E1's fold-a students e1 s1/s2 (rerun in this slot), and v1m s1/s2 if trained. Seeds are pooled.
- **Contrasts and success.**
  - ub minus e1 (primary) and ub minus v1m: `compare-arms`, two-sided, Holm over the two. Task intervals over 58 tasks
    are primary, because one fold has only 5 families; family intervals are reported beside them. The exhaustion share
    per arm is also reported.
  - **Supported:** ub's exhaustion share is at least 10 points below e1's, and ub minus e1 is at least +0.05 with the
    task interval above 0 and Holm p below 0.05.
  - **Falsified:** the upper end of ub minus e1 is below +0.05.
  - Only if supported do folds b and c follow.

## E4 on vim (lane PLAY, 2026-11-21): gate failed

Phase 1 only; no training and no exam ran. The 4B's failed attempts of the 73 never-solved train tasks (`train-k1`,
`train-k4`, 145 failures) were forked at the 27B teacher's verified passes (`experiences repair --intervention
path_guided --paths <teacher set>`, brain-bcd67e5ad on n424 gpu0, E1's sampling).
- **Routing:** 29 of 73 tasks have a replayable teacher path. 27 tasks: every pass uses `shell`, which the task's Sven
  configuration does not offer; 11: shell or the vim tool's random scratch path (bench knowledge 076); 2: scratch path
  only; 4: no teacher pass.
- **Yield:** 7 of 29 routed tasks (24%; 7 of 73, 9.6%) got a verified unhinted suffix with a tool call of the 4B's own
  (8 samples per failure). Gate: 30% of never-solved tasks. **Failed.** An earlier pass that kept any passing
  continuation "repaired" 24 tasks, but 101 of its 106 records trained only a final reply after the teacher's work.
- **Why:** a teacher pass has a median of 1 state-changing action (214 passes; 123 have 1, 65 have 2). There is no
  intermediate state on vim, so E4 moves to a multi-step domain. Verdict `play-e4-gate` (`falsified`, belief 0.35 to
  0.15) in the canonical campaign. GPU-hours: about 1.3 (n424 gpu0, three repair passes).

## SD1: one teacher-free self-distillation generation on the champion (pre-registered 2026-11-21, before any trial)

Advisory 2026-11-21, section 4. The distilled students' failures are mostly termination (exhaustion 56-59% at B), and
within B they are right about 83% of the time. One generation of their own budgeted passes may teach finishing.
- **Harvest:** each E1 fold's seed-1 student (`e1-<fold>-s1`, lane EXP) attempts its own training rungs (the tasks of
  the 9-10 rungs its fold trained on) x 8 at B (15,114 completion tokens, 9 turns, per-turn thinking 2,048, sampling
  0.6/0.95/20, seed 20261121, `wall_clock_factor` 16 with bench 0d8f13de). Throughput job: replicas fill the cards.
- **Selection:** verified passes within B, at most 2 per task, lowest total completion tokens first. No teacher
  record anywhere.
- **Training:** from base on those records only, v1's recipe (LoRA r16/a32, lr 2e-4, 4 records per step, steps =
  round(0.4 x records)), seeds 1 and 2, the dataset built as E1's (`sft-step --pack --only-actions vim-vim
  --max-block 16384`).
- **Exam:** SD1's students against E1's students (seeds pooled per arm) on each fold's held-out tasks x 4 per seed,
  in one slot, at E1's rerun load (16 trials in flight per arm), budget B, factor 16. `compare-arms --families`
  (14 rungs), interval seed 20261121; `infra_timeout` counted per arm.
- **Fast read first:** fold a is harvested, trained and examined first; folds b and c follow only if fold a is not
  clearly null (its point estimate above -0.02 or its exhaustion at least 5 points below E1's).
- **Success:** exhaustion at least 10 points below E1's students, and held-out pass@1 above E1's (family interval).
  **MDE:** on 14 families the student-against-student SE is about 0.034, so only about +0.07 or more would be
  supported; a smaller gain reads as not supported, not as ruled out. Wrong-within-B is reported beside the pass rate
  (the risk: earlier submission with more wrong answers).

**SD1 fold a result (read 2026-11-22 01:20): falsified at the fast read; folds b and c not run.**
- Harvest: E1's fold-a seed-1 student on its 110 training tasks x 8 at B (n424 gpu0 and n445 gpu0, 96 in flight
  each): 455 of 880 passed (51.7%), 372 exhausted B. Kept 171 passes on 91 tasks (at most 2, lowest cost); one trial
  the trainer refused (its chat template rendered the final prompt differently) was left out and replaced by the next
  pass of its task; 169 records. Trained from base, seeds 1 and 2, 68 steps.
- Exam (fold a's 58 held-out tasks x 4 per seed, one slot on n424 gpu0, 16 in flight per arm, budget B, factor 16;
  no infrastructure error, no `infra_timeout`, no upstream error in any round):

| arm | trials | pass@1 | exhausted B | wrong within B |
|---|---:|---:|---:|---:|
| E1 students (s1, s2) | 464 | 0.287 | 64.7% | 6.7% |
| SD1 students (s1, s2) | 464 | 0.241 | 68.8% | 7.1% |

  SD1 minus E1: **-0.045**, task interval [-0.093, -0.002], family interval [-0.078, -0.012] (5 rungs), 12 better /
  20 worse / 26 ties, p two-sided 0.215. Exhaustion rose 4 points instead of falling 10. By the fast-read rule
  (estimate above -0.02 or exhaustion 5 points down) fold a is clearly null, so folds b and c stopped (fold b's harvest
  was done, its dataset built: 134 records; fold c's harvest was cancelled). Its own passes within B did not teach the
  student to finish: a student trained on them still reasons to the budget. Verdict `play-sd1-fold-a` (`falsified`,
  0.45 to 0.25) on the new candidate `overthink.self-distill-under-budget`. Report: `lane-play/e4/logs/sd1-a.compare.json`.
- Deviations: training ran with splinter b9ae3a292 and data with d85e06ba4 (both later than the E1 builds; the dataset
  path reproduced E1's fold datasets bit for bit); bench was aa2a3073 to 6172a711, whose `benchlib/` and `agents/` equal
  0d8f13de. GPU-hours: about 2.3 (harvest 0.6, training 0.15, exam 0.75, fold b harvest 0.4, cancelled fold c 0.4,
  plus the E4 repair passes 1.3 counted above).

## Gaps from the 2026-11-21 late review (advisory `~/project/shared/tmp/lane-advice-1121/advisory.md`)

| Gap | Bridge | Lane |
|---|---|---|
| The vim instrument has 14 families; between two students the MDE is about 0.07-0.09, not 0.05 | A calibrated exam of about 40 families: existing bench families plus A2's lineages, with a disjoint calibration pool (bench GL-B2) | KBN, after E7 |
| No extrapolation test: the folds interleave difficulty rungs | An extrapolation fold: train on rungs 1-9, examine rungs 10-14 | EXP |
| Teacher-free has never been measured on the instrument that detected the teacher's transfer (mode, task set and record count differ) | Source x support on E1's folds: S, T-same, T-other at matched count | EXP |
| Students fail by exhausting the budget, not by competence (83% right when they finish) | Autopsy of exhausted trials; a cost curve over per-turn thinking without training; one teacher-free generation from the champion under B | EXP, PLAY |
| vim tasks have one decisive command, so E4 has no intermediate state; the vim tool names a random scratch path and rarely shows the buffer | E4 moves to a multi-step domain; a new vim tool version with a deterministic scratch path and the changed buffer in every observation, published as a new private dataset version for the user to promote | PLAY |
| The 4B consults a mounted knowledge base in about 6% of trials | E7 with preloaded pages, harm measured against correct evidence | KBN |
| On-policy distillation is blocked: teacher scoring of student tokens is unconfirmed | Confirm the 4B and 27B tokenizers are identical and brain's scoring call returns teacher log-probs on student tokens; a reverse-KL term in brain's shared PG path with gradcheck | E6, after E6 |
| No transition source outside tasks | Task-free tool play, declared before use, as an E6 source | E6, after E6 |

**E8 dropped (2026-11-22 00:00, coordinator, after the late review).** It nearly repeats E2: 555 of the 579 existing
teacher passes are already within B, and 489 also within 2,048 per turn. The partial capture (started 23:46, OOM-killed
on two cards at 23:57, stopped at 23:59) is archived unread (`lane-exp/archive/e8-capture-dropped.tar.zst`). Its
pre-registration header should read 23:43 (commit 2c511ac17), not 23:50.

## E9 and E10 (lane EXP; pre-registered 2026-11-22 00:20, before any of their trials or trainings)

**E9: cost curve without training.**
- **Question.** Is the students' exhaustion partly a serving parameter?
- **Arms.** Base and E1's fold-a students (v1 recipe with fold a held out; s1, s2), each at three per-turn thinking
  budgets: 512, 1,024 and 2,048 tokens. All other settings are E1's: budget B, sampling, sven-fe9cda90, bench 0d8f13de,
  brain-bcd67e5ad. That is 9 arm jobs.
- **Exam.** Fold a's 58 held-out tasks x 4 attempts per arm. One round per card: each card runs all 9 arms for one
  attempt, 12 trials in flight per arm. Rounds 1-2 run on n445 gpu1-2 and rounds 3-4 follow on the first free cards.
- **Read.** Per thinking budget, pass@1 and exhaustion share per arm, and student minus base (`compare-arms`, task
  interval over 58 tasks). This is descriptive: a cost curve, with no success rule.

**E10: source by support.** Single variable: the source of about N records per fold (N matched within the fold).

| Arm | Data from the fold's training rungs |
|---|---|
| S | The base's own verified passes from E1's base arm (within B by construction), de-duplicated by identical tool-call sequence, at most 2 per task |
| T-same | Teacher passes (the 579) on the tasks S covers, as many per task as S has, lowest total completion tokens first |
| T-other | Teacher passes on tasks the base never solved in E1 (0 of 8), at most 4 per task, lowest cost first |

- **Folds.** E1's a, b and c, plus an extrapolation fold x that trains on rungs 1-9 and examines rungs 10-14.
- **Matching.** Per fold, N = the smallest of the three arms' counts, and each arm is drawn to N uniformly with seed
  20261121. Datasets are curated as v1. Training: v1's recipe, seeds 1 and 2, steps = round(0.4 x N).
- **Exam.**
  - The fold's held-out tasks: base x 8; S, T-same and T-other x 4 per seed; and, in folds a-c, E1's students as an
    anchor (x 4 per seed).
  - One fold per exam server, at most 8 adapters, 12 trials in flight per arm, and 4 rounds that each give every
    student one attempt and the base two. Per-arm preflight and round check. Folds a-c and x are pooled over all 213
    held-out tasks for the primary contrasts. Fold x is also reported alone as the harder transfer.
- **Contrasts** (`compare-arms`, family intervals, Holm over the three):
  - T-same minus S: content.
  - T-other minus T-same: support.
  - S minus base: the strict teacher-free number.
- **Power, stated in advance.**
  - Student against student at about 100-120 records: E3 measured a family SE of about 0.018, so the minimum detectable
    effect at 80% power is about 0.05. T-same minus S and T-other minus T-same can detect about +0.05.
  - Against the base, the family SE was 0.035-0.052, so the minimum detectable effect is 0.10-0.15. **S minus base can
    neither confirm nor rule out +0.05.** It can only show a gain of about 0.10 or more, or rule out about 0.15.
- **Rule.** Each contrast's verdict follows the problem-state rule: supported, falsified or inconclusive against
  d = 0.05. S minus base is reported with that power caveat.

## E6 (lane E6): early-experience transitions (pre-registered 2026-11-22 00:50, before any E6 training or trial)

**Question.** Is the environment's response to the 4B's own actions, failed attempts included, a learning signal that
transfers to held-out families? Design from the late review (advisory section (d)), replacing E6's registered control
(extra policy records), which has no source: each fold already uses every teacher pass.

**Observation check, before any training (2026-11-22 00:05).**
- Measured on E1's 1,344 base trials (4,114 tool observations): 13.6% show buffer lines (an `output:` section), and
  13.4% after an edit-like call. 58% are Vim errors (exit status, failing script line, E-code), and 28% only file
  messages (`"main.txt" 1L, 7B written`). Length: p50 60 tokens, p99 133, max 409, so the 512-token cut never binds.
- **Vim's observations mostly carry error dynamics, not buffer state.** A null result here is weak evidence against
  E6 in general. A tool version that shows the changed buffer (gap table: lane PLAY) is the better test bed.
- Every error observation names a random scratch path (`/tmp/vim-tool-XXXX`). These tokens are left in as noise, with
  no string rule. Their share of supervised observation tokens is 8.1-8.5% per fold.

**Data (built with splinter-7d5199191: the `transition` view, commit 7d5199191).**
- Source: the base's own exam-mode trials on each fold's training rungs only, failed and passed:
  - E1's base exam jobs `fa/fb/fc-base-*`;
  - lane VIM's `train-k1` and `train-k4` (thinking on, no added instruction).
  - No new rollouts. A held-out-rung trial is refused by the selection, which asserts it.
- 1,430 / 1,339 / 1,599 trials for folds a / b / c.
  - Ingestion refuses 4 / 7 / 11 trials whose score and grade disagree (budget-exhausted verifier passes).
- Records: one per trial, exactly what the student saw (the requests as sent) through its last observation.
  - Every observation of a `vim-vim` call is supervised: its content only, not the template's framing.
  - Every model turn is context, with loss weight 0.3 on the observations (`--loss-weight 0.3`).
  - At most 2 copies of one (task, call, observation digest) are supervised (`--per-transition-cap 2`).
  - Records over 16,384 tokens are left out.
- tr-a/b/c: 1,306 / 1,217 / 1,423 records; 222k / 215k / 254k supervised observation tokens.
- sh-a/b/c: the same build under `--control shuffled` (seed 20261121): every supervised observation gets another's text.
  - The block bound is applied after the shuffle, so the sets differ slightly: 1,307 / 1,216 / 1,425 records and
    3-4% more observation tokens than tr.
- v1-a/b/c: E1's selections rebuilt in this state; the dataset digests are identical to lane EXP's e1-a/b/c.
- **Ratio (dated deviation, coordinator 2026-11-22 00:10).** Observation to policy tokens is 0.33 / 0.37 / 0.32,
  not the advised 1:1. Every de-duplicated transition is used, without upsampling: E1's base trials plus train-k1/k4
  are all there is. At weight 0.3 the observation share of the loss is about 0.1.
- The task-free tool play source (gap table) is not used in this run. It will be declared separately before any use.

**Arms (one fold each; seed 1; Qwen3-4B, v1's recipe: LoRA r16/alpha 32, lr 2e-4, no weight decay, 4 records per
step, eval every 14, patience 4, max block 16,384).**
- Steps = round(0.4 x the arm's records), so every record is seen as often as in v1.

| Arm | Datasets | Purpose |
|---|---|---|
| v1 | v1-f | Anchor (retrained on the E6 brain build) |
| v1+T | v1-f + tr-f | Treatment |
| v1+S | v1-f + sh-f | Same tokens and format, broken dynamics |
| T | tr-f | Strict teacher-free arm (base + transitions only) |

- Training and serving use one frozen brain build: brain main plus the generic per-message `loss_weight` and
  non-assistant-span change (commit recorded at freeze).

**Exam.** E1's protocol on each fold's held-out rungs (58 / 65 / 45 tasks; 168 tasks over 14 families pooled).
- Settings:
  - budget B = 15,114 completion tokens and 9 turns;
  - per-turn thinking budget 2,048;
  - E1's sampling;
  - sven-fe9cda90 and bench 0d8f13de (lane EXP's pinned worktree);
  - wall-clock factor 4.
- Base and the four adapters x 8 attempts, one fold per exam server (5 arms), with a registered per-replica load of
  16 trials in flight per arm.
- 8 rounds of one attempt; per-arm preflight and E1's round check.

**Contrasts** (`compare-arms --families`, interval seed 20261121, Holm over the three):
1. Primary: v1+T minus v1.
2. v1+T minus v1+S: whether the dynamics matter.
3. T minus base: the strict teacher-free number.

**Success.** E6 is supported if contrast 1 is at least +0.05, its family interval is above 0 and its Holm p is below
0.05, and contrast 2's point estimate is above 0. Falsified if contrast 1's family interval lies below +0.05.
Otherwise inconclusive, by the problem-state rule.

**Power, stated in advance.** 168 tasks over 14 families, 8 attempts, one seed per arm.
- Between two students the family SE is about 0.025-0.03 (late review), so the minimum detectable effect at 80% power
  is about +0.07 to +0.09. **This run can confirm only an E6 effect of about +0.07 or more.** A +0.05 effect would
  most likely read as inconclusive.
- T minus base has an MDE of 0.10-0.15 (E10's power note).
- Expansion to more seeds follows only if contrast 1 is not clearly null (point estimate at or above +0.03).
- Exhaustion share and `infra_timeout` are reported per arm.

**Disk-full window (dated 2026-11-22 00:25, before the replacement rounds).** The project disk had 0 free inodes from
about 00:00 to 00:20. Every round with a trial graded in that window is discarded unread and archived
(`lane-exp/archive/disk-full-window-rounds.tar.zst`): the E2 4B rerun 2's round 3 of folds a and b, and E9's first
two rounds. They are replaced with identical settings. Fold a and b each get two new rounds, numbered 5 and 6, so
every fold keeps 4 rounds; fold c's 4 rounds ended before the window. E9 restarts from its first round. From now on,
live job directories are on node-local disk, and only packed jobs are copied to the project disk.

**E6 follow-up: task-free tool play as a transition source (declared 2026-11-22 01:00, before any play trial).**
- **Source.** The base 4B on the vim train side's task images (168 tasks x 1), in E1's mode and budget B. The profile
  appends one instruction (`scripts/mkplay.sh` in lane E6): a free practice session, no goal, many small experiments
  with the tool, print the buffer after edits.
- **Use.** Its trials are a transition source only. The `transition` view supervises the tool's observations and never
  a token the model sampled under that instruction (rule 1). The instruction stays in the record's context.
  - The tool is shared with held-out tasks, so play runs only in the training side's task images.
  - Play trials never enter a fold's data while that fold is examined in this run.
- **Read, descriptive.**
  - The share of observations that show buffer lines, against E1's base trials (13.6%).
  - Observation tokens per trial.
- **Training use.** A training arm on play transitions is a later, separately pre-registered experiment. This run
  only measures whether the source carries more state.
