<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Experience to competence

Scope: the cross-repository programme that turns verified agent experience into measured competence on unseen tasks (bench practice and sealed exams, sven rollouts, brain training, splinter curation, campaigns and the promotion gate): its goal, experiment design, arms, results, open items and standing validation rules.

**Status 2026-11-19.** The loop runs end to end on real models. One signed generalisation result exists (distillation from a
verified teacher, tier c, +0.383). No teacher-free arm has beaten that champion yet. Update this file in the same commit that
changes an item. Sibling ledgers: `campaigns.md`, `failure-repair.md`, `distribution.md`, `generalized-autonomous-learning.md`,
`api-only-brain.md`; bench `measurement-authority.md` + `generation-exams.md`; brain `generic-training-capabilities.md` +
`training-efficiency.md`; sven `training-instrumentation.md`.

## Goal and ownership

Goal (set 2026-11-17): a complete verified loop across the four repositories, shown with real models on unseen tasks, with
ablations, several generations, interruption recovery and an evidence matrix. Ownership: bench practises and examines (sealed
exams, signed verdicts, never a source of training data); sven runs the agent and records ATIF with the policy trace; brain trains
and serves (API only); splinter curates, repairs, trains through brain's jobs, runs campaigns and applies the one promotion gate.

## Evidence

- Private bundle `swedishembedded/experience-to-competence-evidence@splinter-b0f65b12fffa370c40fa` (commit 01dc4011, 12 files,
  150 MB): gen-0 teacher set (550 experiences), SFT dataset, candidate A2s1 with adapter, 5 signed aggregate verdicts.
  `splinter experiences ingest-hf` round-trips it on the real Hub (republish no-op, re-ingest stores nothing).
- Item-level sealed data never leaves bench's private directory; only aggregates are recorded here.
- Evidence matrix (78 rows, audited 2026-11-19 morning): done-verified 32, partial 43, missing 3 (task decomposition then, HF
  ingestion and HF artifacts; the last two have since landed), none untested or blocked. Gaps by impact are the open items below.

## Experiment design (pre-registration summary)

- **Primary H1.** An adapter trained only on train-side trajectories has higher strict pass (reward 1.0) than the base Qwen3-4B on
  a sealed exam, same harness and budget. Unit: per-item pass over attempts, paired by item; cluster bootstrap over lineages,
  stratified by domain; at least 3 training seeds per arm. One pre-registered primary contrast; Holm for further arm contrasts.
  Typed non-model failures are refilled and reported; model-caused timeouts are failures.
- **Arms** (matched on train tasks, LoRA config, optimiser steps/gradient tokens): A0 base + A0' null replicate (noise floor);
  P base + one instruction line; A1 SFT on own verified successes; A2 SFT on verified teacher trajectories (distillation);
  A3 failed prefix + teacher repair, loss on the repaired suffix; A3c cumulative teacher + repairs from base; A4 weights x knowledge
  at inference 2x2; A5 preference family (DPO, IPO, SimPO, ORPO), KTO; A6 GRPO/Dr. GRPO/DAPO/RLOO from the best SFT init versus
  filtered BC at identical rollout count. The teacher is re-run under the student's compact profile and tools. Procedure versus
  content: pass = P(valid answer file) x P(correct | valid); procedure-only demos arm.
- **Tiers.** (a) unseen instance; (b) held-out variants; (c) held-out lineages withheld from all training including teacher demos
  (primary); (d) leave-one-domain-out, base versus best arm, the only basis for cross-domain claims; (e) composition. Retention: a
  suite where base scores 40-70%, non-inferiority one-sided 95%.
- **Controls.** Frozen harness, profile, tools, caps, sampling, server binary, quantisation; caps by steps and tokens; arms
  interleaved on the same servers where possible (the gen-1 exam was not, and says so); base served through the same adapter path.
- **Multi-generation.** Champion k rolls out on train tasks in the frontier band [1/8, 7/8], cumulative data, retrain from base;
  gate = dev gain + retention non-inferiority, never the sealed exam; one fresh sealed slice per generation, burned; three
  generations (gen3 > gen1 with CI, each step non-inferior); mandatory control with the same total rollout budget; the
  self-improvement lineage uses zero teacher calls after gen 0. Interruption: kill at seed-drawn points mid-rollout and
  mid-training, identical dataset manifest hash and outcome within the A/A band.
- **Kill / reframe.** A/A beyond its CI or infra-failure rates differing > 5% -> fix the harness; A2 < 20% strict on support -> the
  student cannot absorb it; P recovering >= 70% of A2's gain -> the headline is a harness fix; seed SD > half the gap -> more
  seeds; < 50 own successes outside accounting -> A1 only from gen 2.
- **Not claimable** without the matching evidence: general capability without leave-one-domain-out + retention; self-improvement
  from arms that used the teacher; arm ordering below about 8 pp.

## Results (aggregates)

Public dev is not sealed generalisation evidence. Intervals are item-paired cluster bootstrap over families.

| Stage | Result |
|---|---|
| Harness defect found 2026-11-17 | the compact prompt told the model not to write the answer file; fixed (sven compact/2): support 3 -> 9 full passes of 96, legal 0 -> 2 of 84. Prompt line P buys nothing (-0.012 [-0.045, +0.021]); null replicate gap -0.027: noise floor ~3 pp |
| Teacher (Qwen3.8-27B FP8) practice | 451 verified demos: support 187, compiled 85, legal 79, accounting 100 |
| Pilot A2 (3 seeds, public dev, 168 items x 2) | base 0.048 -> 0.461-0.485; pooled +0.428 [+0.367, +0.491]; seed SD 0.012; per domain support +0.65, compiled +0.73, legal +0.35, accounting +0.10; retention +0.082 [0.000, +0.170] |
| Pilot A2 sealed exam | tier a +0.492, tier b (held-out variants) +0.410 [+0.327, +0.494]; verdict Invalid (8 leak-flagged passes from git-history probing); slice burned |
| Lane R dev (129 items, one frozen eval build) | base 0.027, A2s1 0.550, gen1 (A2 data + 265 own frontier passes) 0.523/0.500/0.539 (-0.030 [-0.088, +0.025]), A6 GRPO 0.581/0.539 (+0.010 [-0.050, +0.071]), DPO 0.516 (-0.035 [-0.085, +0.019]), A3 repairs only 0.167 (-0.384 vs A2s1, forgetting), A3c cumulative (teacher + 143 repairs, from base, 527 records, one seed) 0.547 (-0.004 [-0.060, +0.053] vs A2s1; +0.380 vs A3; retention not run); KTO dropped (170 s/step) |
| Dev retention | base 0.555, A2s1 0.571, gen1 0.637-0.681 (+0.084 [+0.021, +0.151]), A6 +0.022, DPO +0.049 |
| Gen-1 sealed exam (plan generations-2026-11-r1, 505 tasks, 3 attempts, signed Ed25519) | A2s1 vs base **Promote**: tier c +0.383 [+0.298, +0.469] (MDE 0.117), retention +0.044 non-inferior at 0.10, leaks 0.99%. A6s1 (primary) vs A2s1 Reject +0.004 [-0.023, +0.032]; filtered BC +0.020 [-0.008, +0.049] Reject; DPO -0.029 Reject; A3s1 -0.308, retention failed, Reject |
| Interventions on 1275 trials (543 failures) | 143 repairs accepted (legal 90, accounting 32, compiled 21, support 0), 495 rejected, 260 refused at replay. GPU smoke of 20 failures: decompose 1/16, knowledge 0/15 (plain teacher 0/8), fork 1/9, retry 0/4 |
| Real `splinter campaign` run | two generations over real brain and bench, three SIGKILLs resumed (train attempt reused brain's job by `resume_key`), gate rejected a candidate trained from the base (-0.19 dev, -0.28 retention); see `campaigns.md` |
| Rollouts | 1280 on-policy trials, 6.5 GPU-h, 87% prefix-cache hit; training per run gen1 0.8, A6 2.8-3.0, DPO 1.9 GPU-h |

Reading: distillation from verified teacher trajectories generalises to withheld lineages and keeps retention. Teacher-free
successors trained on A2's own rollouts do not improve on it; repairs alone are harmful. Seed count on the exam is 1 per
candidate; tier b/c share the four training domains; the retention margin is 0.10 after calibration. Every result before
brain 1bdd5400c was served with an injected reasoning-effort instruction (brain knowledge 531): compare within a build only.
Git-history probing persists in trained arms (0.8-1.6% of trials, under the 2% cap).

## Landed (this programme, splinter side)

Bench ingest of `bench.score/1` and the practice executor (`tasks harbor practice`); API-only brain training; the diagnosis
taxonomy and `experiences repair` with four interventions behind one trait; preference/feedback/policy-gradient views with exact
sampled token ids and prompt ids (request-log contexts); one promotion gate with in-process verdict verification and a
six-family minimum; `splinter campaign run|status|rollback`; capability-aware curriculum; packed expdb (`state pack`) and
`experiences import`; `splinter publish` / `ingest-hf`. Training regimes adopted by the 2026-11-18 plan (advantage computed in
brain, named presets, one preference objective, KTO, served distillation, accepted levels L1-L7 per regime) are landed through
L5; the brain-side table is in `generic-training-capabilities.md`. Remaining plan phases: C live rollback and resume (19), D
proposer/solver/verifier roles with learnability reward, SPIN, consolidation arms at equal compute (21-23), E PPO critic only on
evidence, memory/skills, test-time training (24-26); experiments at equal GPU-hours across GRPO, Dr. GRPO, DAPO, RLOO, DPO,
SimPO, ORPO, KTO and a rejection-sampling SFT floor, with fresh prompts per round.

## Standing validation rules

- Never degrade a proven result unknowingly. Any new brain or sven build must pass the **reference-adapter reproduction gate**
  before it becomes the default: the champion adapter served on the new build is non-inferior on a small paired dev, and the
  recipe reproduces within the seed band. Brain holds the tiers (`generic-training-capabilities.md`).
- Each run uses its own frozen binary copy (commit and sha recorded); never switch mid-run.
- Tests and profiling stay fast; GPUs stay busy with short valuable experiments.
- Sealed or item-level files are never written outside bench's private directory (an incident on 2026-11-18 moved them back).
- Few large files: packed experience state, packed bench jobs; project disk has inode quotas.

## Open

| # | Item | Next action | Acceptance |
|---|---|---|---|
| 1 | No teacher-free arm beats A2s1 | own-success-only arm, more steps, harder frontier, multi-pass GRPO with exact behaviour tokens (fresh rollouts on brain >= 1bdd5400c, `behaviour: recorded`) | gen-2 sealed slice: tier c gain with CI above 0 against the champion at min_gain 0.15, retention non-inferior |
| 2 | Git-history probing in every trained arm | drop or negatively reward probing trials in curation and the PG view (drop-step did not help, knowledge 179); retrain A2 | probe share per arm below 0.5% on dev and retention |
| 3 | A3 forgets A2 (A3c done 2026-11-19: cumulative training removes the forgetting; repairs add no measurable dev gain on one seed) | A3c seeds 2-3 and retention, repair-DPO on 103 pairs, sealed-exam A3c only if dev shows a gain | dev non-inferior to A2s1 plus a measurable gain on the repaired classes |
| 4 | Repair yield | support 0 accepted, 260 refused: replay with the student's sven build or normalise nondeterminism; per-task cap; larger real smoke for decompose/knowledge | accepted/graded >= 25% on a 100-failure pass; knowledge effect measured by the 2x2 (A4) |
| 5 | Campaign never run for learning | one real campaign with `[examine.agent]` naming `--plan/--slice`, candidates continuing from the champion (or rehearsing its data), capability curriculum on | two generations, gate decisions signed, interrupted at least once, one command from a clean state |
| 6 | Cross-domain evidence absent | register a leave-one-domain-out exam plan (bench) and run base versus best arm | 5-fold LODO verdict signed |
| 7 | Exact tokens end to end | fresh rollout on brain >= 1bdd5400c; PG view contexts exact as recorded; bench exports from request logs | exact contexts on >= 95% of steps of real rollouts |
| 8 | Release and harness-optimise use the gate only on examined verdicts; `generation run` not retired; retention non-inferiority has no family minimum | give release candidates sealed slices; retire `generation run` per `acquisition.md` | one gate everywhere, no private rules |
| 9 | State sinks | pack bench practice exports (dir per trial), migrate lane states with `state pack` | a campaign's state stays under 10k files |
| 10 | Equal-budget matching and token caps | sven total token cap; arm design records tokens and GPU-h | cost per percentage point reported per arm |
| 11 | Contracts copied by hand | fixture sync check between brain's training-records fixtures and splinter's; ATIF fixture corpus from sven | check fails on drift |
| 12 | Bundles | sign the manifest; carry release lineage; publish the later evidence (gen-1 exam aggregates, lane R datasets) | republish no-op, ingest verifies |
| 13 | Capability labels | three capabilities label nearly every task; add narrow ids or weight by discrimination | curriculum weights differ across capabilities on real data |
| 14 | Failure-class fixtures | add infrastructure, repetition, probing fixtures from eval-set runs (aggregates only) | precision/recall reported per class |
| 15 | Guide | commit one runnable end-to-end example (practice -> train -> exam -> gate) | `docs/getting-started` page that a clean checkout can follow |
| 16 | Real runs for KTO, IPO, SimPO, ORPO | after brain efficiency work (reference pass 55 min, 170 s/step) | one seed each on the frontier data |
| 17 | Two load-sensitive test flakes found by the clean re-verification at dfb62e12f (3619 passed, 2 failed, each passes alone) | `solve_concurrency::a_remote_model_is_asked_up_to_the_configured_width_at_once` (timing) and `distributed_plan::a_retry_writes_its_own_plan_naming_the_same_job` (ETXTBSY, a spawn racing a just-written executable): make both deterministic | 20 consecutive parallel workspace runs without either failing |
| 18 | Campaign demo evidence incomplete at hand-over: generation-2 evaluate and rollback outcomes were not recorded when this ledger was written | record both in `campaigns.md` from the campaign lane's final report | rollback shown moving the champion back on real models |

Verification at the time of writing (clean force-rebuilt gates, no GPU): splinter `make check` exit 0; sven tests 5300 passed, clippy
and doc clean (the wasm target is not installed on the node); brain clippy clean, `ci/static` clean, per-crate suites data 376,
model 1103, qwen3 449, gpu-core 459, rl 73 passed; bench `make check` and `make test` exit 0 (1121 unit, 437 integration).
Worktrees sharing one `CARGO_TARGET_DIR` link each other's workspace crates (knowledge 181): clean the workspace crates first.

## Campaign demonstration (2026-11-19)

The first real `splinter campaign` ran against real brain and bench, with the A2s1 adapter as champion.

- **Generation 1** ran every stage, and the gate rejected it:

  | Comparison | Result |
  |---|---|
  | dev | -0.19 [-0.32, -0.05] |
  | sealed throwaway slice | -0.36 |
  | retention | non-inferiority failed |

  The candidate trains from the base, on little data.
- **Generation 2** finished rollout (only the 4 unmastered tasks), cumulative curate and train. Evaluate was stopped at the
  lease deadline.
- **Interruption:** four SIGKILLs (rollout twice, train, evaluate) each resumed at the first incomplete stage. The train
  stage reused brain's job by resume_key and trained once. A replay without interruption gave an identical dataset id,
  candidate and adapter digest.
- **Rollback** was correctly refused, because nothing had been promoted.
- **Fixes:** 67b1f60d9 (an adapter champion), d1cac5987 (`--brain-adapter-dir`).
- **Open items, in campaigns.md:**
  - orphaned sandboxes after SIGKILL;
  - candidates always train from the base;
  - evaluate runs serially on one card;
  - status is blocked while a run holds the lock.

## Why teacher-free learning showed no gain (diagnosis 2026-11-19, expert review against the literature)

**Status: open; the next experiment below decides it.** The verdict: the approach is NOT refuted, because the
teacher-free arms never truly tested it.

**Evidence from our own runs**

- **GRPO never moved the policy.** From `train-a6-s1.err`: ratio about 1.002, KL about 0.002 per step, clip fraction
  about 1%. That is 67 steps at lr 2e-5, roughly 270 informative trajectories seen once each, and half the records come
  from zero-variance tasks.
- **A2 had unharvested headroom**, from its own 1,280 rollouts on train tasks:

  | Domain | pass@1 | pass@8 |
  |---|---|---|
  | all | 0.572 | 0.781 |
  | legal | 0.40 | 0.70 |
  | accounting | 0.29 | 0.63 |

  So about 0.21 of in-support headroom was left. 35 of 160 tasks were never solved (out of support).

**Literature**

- RLVR mostly sharpens within the base's support: Yue et al. 2025 (arXiv 2504.13837); "The Invisible Leash" 2025. A
  counter-claim exists (arXiv 2506.14245), on math only.
- For small models, distillation beats RL (DeepSeek-R1 2025).
- Teacher-free agentic gains exist only at larger scale, with thousands of tasks and many iterations: Golubev et al.
  2025 (arXiv 2508.03501, 72B, RFT then DAPO, 20% to 39% on SWE-bench Verified); DeepSWE; SkyRL-Agent.
- RLVR needs hundreds to thousands of updates (one-example RLVR, arXiv 2504.20571).
- Learnability filtering: variance of pass rates (arXiv 2504.03380).

**Ranked causes**

1. Under-optimised GRPO (signal/compute).
2. Too few tasks for transfer, and an eval MDE above the plausible 3-5 point effect.
3. No information on out-of-support tasks (approach limit: teacher, repair or hint).
4. gen1 diluted (about 1.2 epochs vs 2).
5. Outcome-only credit over about 11 turns.
6. Off-context behaviour logprobs and the train/serve prompt mismatch (fixed in brain 1bdd5400c and splinter 5a0c7cd43/6e0ea27a1).
7. DPO pairs without a shared prefix.
8. LoRA capacity: unlikely.

**Next experiment (about 18 GPU-h, after the brain efficiency fixes, on brain >= 1bdd5400c with exact tokens)**

1. **Ceiling.** A2 pass@1..16 on 48 fresh held-out practice tasks (12 per domain). Stop if H = pass@16 - pass@1 < 0.08.
2. **Two on-policy GRPO rounds.** 80 tasks × G=8, sampled by p(1-p), zero-variance groups dropped; round 2 re-rolls 40 of
   round 1's frontier tasks.
   - Recorded behaviour logprobs, token-sum loss (Dr. GRPO), lr 1e-4, 3-4 epochs per batch.
   - Target KL 0.02-0.05 and a clip fraction of 10-20%.
   - Probing is rewarded negatively.
3. **Control.** Rejection fine-tuning on the same rollouts, continued from A2 with teacher rehearsal.
4. **Eval.** 48 held-out tasks × 8, paired by task.

**Reading the outcome**

| Outcome | Meaning |
|---|---|
| KL near zero | defect |
| KL moves, train tasks flat | credit assignment (try turn-level rewards) |
| Train tasks gain >= 0.10, held-out flat | memorisation (more tasks needed) |
| Held-out gain >= 0.05 with CI above 0 | first teacher-free gain |
| Never-solved tasks still at 0 at k=16-32 | teacher or repair lane |

## Learning signal per GPU-hour (expert review 2026-11-19; bench side in bench `practice-signal`)

**Finding:** bench verifiers already compute weighted partial credit (`functional_correctness`), but the default
`RewardShape::unshaped()` gives every failure 0. About half of the rollouts sit in all-pass or all-fail groups, which
carry no advantage.

**Ranked splinter items** (signal gains are estimates)

1. **Shaped reward (first package).**
   - Reward: `r = 1` on a pass, else `c * max(0, (f - f_naive) / (1 - f_naive))` with `c` about 0.5, where `f_naive` is
     bench's naive floor. Until bench exports it, use raw `functional_correctness` (the existing `--reward-metric`,
     `--fail-ceiling`).
   - The gate stays strict pass. Track over-listing (F1 sets) and abstention drift.
   - Optional A/B: per-check z-scored advantages, once bench exports `checks`.
   - Zero-GPU validation: on the existing 1,280 rollouts, count groups with a nonzero `functional_correctness` spread,
     against the 82 with a pass spread.
   - Measured 2026-11-19 on lane R's practice exports (1,306 trials; 160 groups of 6-8 attempts, 31 single trials
     skipped): 81/160 groups have a strict-pass spread, 110/160 a `functional_correctness` spread (every pass-spread
     group is among them). Shaped reward would make 29 more groups informative (51% to 69%).
2. **Variance-per-cost selection (first package).**
   - Metric: `LS_i = Var_j(r_ij) / mean cost_ij`, with cost = completion plus uncached prompt tokens (or GPU-seconds).
     Shrink the variance by a Beta posterior.
   - Per family: count lineages, not seeds.
   - Sample in proportion to `sqrt(Var)/cost`; drop zero-variance groups. This replaces the band factor in
     `curriculum/capability.rs`.
   - Validation: rank-correlate `LS_i` with the pass change after one round.
3. **Start-from-state forks at the decisive step.**
   - Replay the prefix through `bench task session-exec`, then roll G suffixes with sven fork.
   - Group by (task, prefix digest); train on the suffix only; check the state digest before forking.
   - Mix in 30-50% full episodes, and move the fork point earlier over generations.
   - Expected 3-5x signal per completion token on forked tasks.
4. **Hint-ladder fading schedule.** Pick the lowest hint level with pass rate at or above 0.3; promote only on hint-0;
   cap hinted tasks at 30% of a batch. The existing `Distill` view on all-fail groups is the no-bench-change alternative.
5. **Potential-based shaping from bench milestone progress**, `F = gamma*Phi(s') - Phi(s)`, if bench exports progress.
6. **Cost.** Tighten the practice step cap to about the p90 of passing trajectories; stop early on repetition or no answer
   file; keep DAPO's over-length penalty.

**First package (3-4 days):** items 1 and 2 here, plus bench items 1 and 2 (`naive_floor` and three dials).
Measure on the same rollouts plus one fresh round:
- informative-group share (82/160 today);
- sum of Var(r) per million completion tokens;
- KL actually moving (0.02-0.05, against 0.002);
- strict pass on 48 paired held-out tasks: the ground truth.
