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
| Brain 1bdd5400c release check (A2s1, 2026-11-19) | tier 3: only the removed 37-token instruction differs (old build without it is bit-identical; first-token agreement 98.8%); tier 4 paired dev, 48 tasks x 4: old 0.510, new 0.557, +0.047 [+0.000, +0.095], non-inferior at 0.05: **pass**, 1bdd5400c is the build for new A2 work (brain `generic-training-capabilities.md`) |
| A2 ceiling (step 1, brain 1bdd5400c) | 48 fresh practice tasks x G=16 (768 graded trials): pass@1/2/4/8/16 0.600/0.699/0.767/0.827/0.875, never solved 12.5%, H = 0.275 (rule: stop below 0.08): **continue** to step 2; per domain in the section below |
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
| 19 | First-package arm run 2026-11-20 (results below, "Item 19: first-package RL round"): no teacher-free gain; the unguarded round damaged A2 through ratio spikes on start-of-run proxy tokens, root-caused and fixed by `dataset build --sampled-only` (b879d37d2, knowledge 188); brain bounds a negative advantage's term by dual-clip PPO since bcd17ba8b (default c 3, `clip_fraction_dual` per step, splinter `train --dual-clip`, db33cd729) | a guarded round on `--sampled-only` data with `behaviour: recorded` (verified: 24 steps, ratio max 12.3, KL 0.018); make exact contexts cover `scrubbed` and `context_reconstructed` actions (48% of tokens are left out today); 2+ seeds, turn-level credit | held-out gain >= 0.05 with CI above 0 against A2s1 on paired fresh tasks, retention non-inferior at 0.10, KL 0.02-0.05 with no guard skips |
| 20 | Bench draws at a rung or of families (bench 15a9220d) were refused when practised: splinter re-staged a draw from count and seed only | done 2026-11-20: the draw identity carries `rung` and `families` into the re-stage (`--rung`, `--family`), `[practice]` takes `rung` and `families` for transfer draws; older practice.json unchanged | `bench_practice` tests: re-stage passes both, old and null draws pass neither |
| 21 | Knowledge-assisted self-learning: no arm has yet learned from written principles instead of trajectories (plan below, "Item 21"; S0 leakage gate PASSED 2026-11-20: `swedishembedded/knowledge@v0.2`, mounted-copy digest `sha256:262ff959...`, bench knowledge audit signed pass on 8 sealed exams with 0 shared 8-grams; S1 2026-11-20: STOP by its rule, A2 collection + instruction - none -0.002 [-0.038, +0.035], but A2 opened the collection in 6% of trials and no domain page (knowledge 191); S1b: page preload +0.029 [-0.050, +0.105], inconclusive, helps support/compiled, hurts accounting VAT/accrual; index preload +0.010; user wording +0.044, no go, fewer look-ups; next: accounting pages back to the author, 96-task confirmation of preload on legal/support/compiled) | bench: `--knowledge <dir>` mount with digest, contamination audit of the collection, open-book exam plan; then S0 and the S1 zero-training check on the base and A2 | S1: collection + instruction against none >= +0.06 pass@1 (or pass@8 +0.05), probe share under 2%; programme: closed-book tier c gain >= 0.05 with CI above 0 against the matched no-knowledge RFT control, retention non-inferior at 0.10 |

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
   - **Done 2026-11-19: H = 0.275, continue.** A2s1 on brain 1bdd5400c (frozen, sha256 `ae65e6f1...`), splinter-bc8fd17de
     `tasks harbor practice`, sven-a0235c71 compact/2, sampling 1.0/0.95/20 with per-trial seeds, policy traces recorded; 48
     fresh train-side tasks (`bench task practice --count 12`, seeds 2026111901-04, bench 80efacd6, no overlap with earlier
     draws), 768 graded trials, none ungraded. Unbiased pass@k:

     | Domain | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | never solved | H |
     |---|---|---|---|---|---|---|---|
     | support | 0.875 | 0.958 | 0.985 | 0.999 | 1.000 | 0/12 | 0.125 |
     | legal | 0.385 | 0.492 | 0.608 | 0.736 | 0.833 | 2/12 | 0.448 |
     | accounting | 0.365 | 0.513 | 0.623 | 0.697 | 0.750 | 3/12 | 0.385 |
     | compiled | 0.776 | 0.833 | 0.854 | 0.875 | 0.917 | 1/12 | 0.141 |
     | all | 0.600 | 0.699 | 0.767 | 0.827 | 0.875 | 6/48 | 0.275 |

     The headroom is in legal and accounting (H about 0.4); support is near saturated. The 6 never-solved tasks are the teacher
     or repair lane's. Job dirs packed into `archive/lane-l4-ceiling-practice-2026-11-19.tar.zst`, the expdb packed.
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
   - **Landed 2026-11-19 (6ac02f92d):** `dataset build --reward shaped-partial [--fail-ceiling C] --practice DIR`, a named arm
     (never the default). Floors come from `practice.json` `tasks[].naive_floor`, matched by the task's content digest; a
     task without one is measured from 0. The manifest's `partial_credit` records the credit, the floors, their
     `practice.json` sha256s and `groups_without_floor`. Campaigns: `[curate] reward = "shaped-partial"` (floors from
     `[practice]`). Bench does not export `naive_floor` yet (lane L2), so every number below is at `f_naive = 0`.
2. **Variance-per-cost selection (first package).**
   - Metric: `LS_i = Var_j(r_ij) / mean cost_ij`, with cost = completion plus uncached prompt tokens (or GPU-seconds).
     Shrink the variance by a Beta posterior.
   - Per family: count lineages, not seeds.
   - Sample in proportion to `sqrt(Var)/cost`; drop zero-variance groups. This replaces the band factor in
     `curriculum/capability.rs`.
   - Validation: rank-correlate `LS_i` with the pass change after one round.
   - **Landed 2026-11-19 (6ac02f92d):** the metric in `splinter-eval` (`campaign::learning_signal`): `V = (n s2 + 2 v0) / (n + 2)`,
     `v0` the Beta(1, 1) posterior's expected `p(1-p)` with rewards as fractional successes; cost from the score's usage
     (`TokenUsage::cost`). Reported per task and per family (lineage from `task.toml` `bench_defect_lineage`) in the
     dataset build report (`signal`) and in `experiences capabilities` (strict and shaped-partial side by side).
     Selection: `[rollout] curriculum = "signal"` beside `capability` (the capability curriculum is unchanged); it
     drops measured uninformative groups, mastered tasks and infrastructure-only tasks, and records every weight.
   - **Zero-GPU check (2026-11-19)** on lane R's practice exports (9 runs, 1,312 trials, one policy A2s1, 160 task
     groups of 2+ graded attempts, 47 lineages; `splinter experiences capabilities <runs>`; the grouping is by staged
     task across runs and counts graded trials only, so the strict share is 83/160, not lane L1's 81/160 per run):

     | Reward | Draw | Informative groups | Informative lineages | sum Var(r) per M completion tokens |
     |---|---|---|---|---|
     | strict pass | uniform | 83/160 (0.519) | 37/47 | 1.155 |
     | shaped-partial (`f_naive` = 0, `c` = 0.5) | uniform | 110/160 (0.688) | 42/47 | 0.903 |
     | strict pass | variance per cost | 1.000 | 37/47 | 3.333 |
     | shaped-partial | variance per cost | 1.000 | 42/47 | 2.290 |

     Partial credit adds groups but compresses the pass/fail gap, so the absolute variance per token falls 22%
     (knowledge 186); the selection is an in-sample estimate (the next round's variance regresses towards the mean).
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

**RL recipe (landed 2026-11-19, 6ac02f92d):** `splinter train --recipe on-policy-rl` (campaign `[train] recipe`), splinter-side
because brain's preset table (`--algorithm`) shapes only the objective: `behaviour: recorded`, `loss_agg: token-sum`, lr
1e-4, 4 passes over the rollouts (steps sized to them; brain's sampler reshuffles each epoch over the records, so several
passes over one batch need no brain change), named flags win. The candidate records `recipe` with the mean KL against
0.02-0.05 and the clip fraction against 0.10-0.20 (`below`/`within`/`above`). Not yet run on a GPU.

**First package (3-4 days):** items 1 and 2 here, plus bench items 1 and 2 (`naive_floor` and three dials).
Measure on the same rollouts plus one fresh round:
- informative-group share (82/160 today);
- sum of Var(r) per million completion tokens;
- KL actually moving (0.02-0.05, against 0.002);
- strict pass on 48 paired held-out tasks: the ground truth.

## Item 19: first-package RL round (2026-11-20, lane L6, n123)

**Setup.** Policy A2s1 on frozen brain 1bdd5400c (exact tokens, no injected prompt), practice through splinter's executor
(splinter a02473bde, sven a0235c71 compact, T 1.0, top-p 0.95, top-k 20, logprobs, policy traces), bench at 89e3189a with
scratch per-rung generator configs. Pool: 80 fresh train-side tasks (legal deadlines rung 0 x 20, accounting rung -1 x 20,
support refund rung 2 x 20, legal and accounting default rung x 10 each); held-out: 48 fresh tasks (12 per domain:
support, legal, accounting, compiled; drawn first, never practised). Datasets: policy-gradient view, `--reward
shaped-partial` (C 0.5, floors from `naive_floor`), zero-variance groups dropped by the view. Training: `--recipe
on-policy-rl` (token-sum, lr 1e-4) with `--reference start` and `--behaviour start`, 8 records per step.

**Cuts (recorded decisions).**
- `behaviour: recorded` was refused: 934,787 of 1,962,062 supervised tokens had no recorded log-probability (actions
  counted `scrubbed` 652 and `context_reconstructed` 475 of 2,668). The runs used `start`, which keeps the sampler's
  log-probabilities where recorded and the start adapter's elsewhere.
- The recipe's 4 passes over 2,913 per-action records were 1,314 steps at 184 s/step (66 h). Every round was capped at 72
  steps (3.7 GPU-h, 576 of 2,628 trainable records, 0.2 of a pass).
- The RFT control (sft-step view of the round-1 verified passes plus the gen-0 teacher set, 585 records, from A2s1, lr
  2e-4) got 144 steps x 8 records, matching both RL rounds' steps.

**Training signal.**

| Round | Informative groups | sum Var(r) per M completion tokens | Median KL | Median clip | Guard skips |
|---|---|---|---|---|---|
| zero-GPU uniform rows (strict / shaped, `f_naive` 0) | 83/160 / 110/160 | 1.155 / 0.903 | | | |
| round 1 (A2s1 rollouts, 640 trials) | 70/80 (0.875) | 1.49 | rl1 0.018, rl1g 0.016 | 0.059 / 0.056 | rl1 none, rl1g 13 of 72 |
| round 2 (rl1 rollouts, 640 trials) | 76/80 (0.95) | 2.05 | rl2 0.016 | 0.056 | 8 of 72 |

The unguarded round (rl1) had 9 spike steps: mean ratio 4.8-484, ratio max up to 1.6e6, KL 31-6,414, one gradient norm
1.4e5. The guarded rerun (rl1g: same data, same seed, `--max-step-kl 0.1 --max-step-grad-norm 5`) spiked on the same
steps; at step 6 the policy had moved by KL < 0.008. rl2 continued rl1 on round 2's rollouts (40 top-LS round-1 tasks
re-rolled plus 40 new), with the same guards.

**Held-out (48 fresh tasks x 8, paired by task against A2s1; task-cluster bootstrap 95%).**

| Arm | pass@1 | pass@8 | delta pass@1 | support | legal | accounting | compiled | timeouts | probe share | NLL/token |
|---|---|---|---|---|---|---|---|---|---|---|
| A2s1 | 0.607 | 0.854 | | 0.885 | 0.562 | 0.312 | 0.667 | 18 | 0.013 | 0.394 |
| RFT control | 0.646 | 0.833 | +0.039 [-0.013, +0.094] | 0.927 | 0.583 | 0.365 | 0.708 | 15 | 0.005 | 0.312 |
| rl1 (unguarded) | 0.378 | 0.729 | -0.229 [-0.294, -0.159] | 0.531 | 0.344 | 0.156 | 0.479 | 23 | 0.016 | 0.384 |
| rl1g (guarded, 1 seed) | 0.576 | 0.792 | -0.031 [-0.081, +0.018] | 0.875 | 0.490 | 0.302 | 0.635 | 28 | 0.052 | 0.428 |
| rl2 (from rl1, guarded) | 0.510 | 0.792 | -0.096 [-0.151, -0.039] | 0.729 | 0.458 | 0.250 | 0.604 | 18 | 0.010 | 0.311 |

**Training tasks** (round 1's 40 top-LS tasks x 8, seeds shared by every arm): A2s1 0.575; RFT +0.094 [+0.019, +0.172];
rl1 -0.191 [-0.263, -0.116]; rl1g -0.056 [-0.131, +0.016]; rl2 -0.094 [-0.159, -0.028]. Spearman of round 1's `LS_i`
with the pass change: rl1 -0.33 [-0.61, -0.02], rl1g -0.13 [-0.47, +0.21], RFT -0.28 [-0.55, +0.04] (restricted range:
the 40 highest LS).

**Retention** (orchestration 32 + workspace-tools-reference 59 tasks, 2 attempts, lane R's dev profile; paired by task;
non-inferior at 0.10): A2s1 0.599; RFT +0.005 [-0.077, +0.088] non-inferior; rl1 -0.165 [-0.258, -0.071] fails; rl1g
-0.038 [-0.121, +0.038] not shown non-inferior; rl2 -0.104 [-0.181, -0.027] fails.

**Compute (approximate GPU-h).** Rollouts 5.5 (plus 1.0 lost when a harness limit killed round 1, whose rerun restarts
every trial, knowledge 190); training rl1 3.6 (plus 1.9 lost the same way), rl1g 3.6, rl2 3.6, RFT 0.8; evaluation about
20.

**Root cause and fix (2026-11-20).**
- **Which tokens.** Every spike was on a start-of-run proxy token. Recorded behaviour log-probabilities are all at or above
  -5.18 (1,027,275 tokens of the round-1 dataset). With the trainer's log-probability at or below 0, a recorded token's
  ratio is at most e^5.18 = 177, and every spike (518 to 1.6e6) is above that bound.
- **What a proxy token is.** The proxy tokens are the actions counted `scrubbed` (652) and `context_reconstructed` (475).
  Their text is re-rendered (not the ids the policy drew) and weighed against the start adapter. Under the unclipped
  negative-advantage side of the surrogate, one such token's ratio enters the gradient (norm 1.4e5).
- **Fix.** `dataset build --sampled-only` leaves every record with a proxy action out as `proxy_tokens` and keeps its trial
  in the baseline (round 1: 1,902 records, 1,127 left out). Training then runs `behaviour: recorded`.
- **Verified with a 24-step diagnostic, not a round.** Same recipe, seed and reference; ratio max per step 2.1-12.3
  against rl1's 1.9e4, 1.3e5, 1.0e4, 1.6e6 and 3.1e4 on steps 6, 7, 15, 19 and 23. KL 0.003-0.029 (mean 0.018), clip
  0.005-0.133 (mean 0.071), gradient norm at most 7.6, no guard skips.

- **Dual clip (brain bcd17ba8b, splinter db33cd729).** brain's shared policy-gradient term now takes `max(min(r*A,
  clip(r)*A), c*A)` for a negative advantage (dual-clip PPO, default `c` 3; `dual_clip` job param, `brain finetune
  --dual-clip`, `splinter train --dual-clip`). A term past `c` gives no gradient, and each step reports
  `clip_fraction_dual`. Specs: loss value, zero gradient at `r` 10 and at log-ratio 14.3 with `A` < 0, unchanged terms for
  `A` > 0, and a finite-difference gradcheck on a tiny hybrid decoder with dual-clipped tokens (CPU). Not measured on a
  GPU training run yet.

**Reading (the ledger's rule).**
- KL is not near zero: the optimiser moves (median 0.016-0.018, just under the 0.02-0.05 band; clip 0.056, under 0.10).
- Unguarded, the round damaged the policy on every measure. The step guard alone removes most of it (single seed):
  held-out -0.229 to -0.031, training tasks -0.191 to -0.056, retention -0.165 to -0.038.
- Guarded, KL moves and the training tasks are flat: the credit-assignment row. With 0.2 of a pass, the round is also
  under-optimised, and half the trained tokens are off-policy proxies. Turn-level credit and exact contexts come next.
- No teacher-free gain: no RL arm reaches held-out +0.05 with CI above 0.
- The RFT control is not teacher-free (gen-0 rehearsal). It gains on its training tasks (+0.094) and keeps retention,
  but its held-out +0.039 has CI including 0.
- None of this is a sealed claim.

### Item 19 round R19: guarded on-policy RL on sampled tokens (pre-registered 2026-11-20 22:00, lane R19, n110, before any job ran)

**Builds (one for every arm, A2s1's baseline re-measured on it).** brain 5e25d5b82 (tool-call parser fix and dual clip;
frozen `containers/bin/brain-5e25d5b82`, brain sha256 a8ead99e..., worker 509d51d6...), splinter 7220b1d5d (holds
`--sampled-only` b879d37d2 and `train --dual-clip` db33cd729), sven a0235c71 compact, bench 62da1dbb (worktree
`r19-bench`). A2s1 = lane L6's `a2-s1.safetensors` (sha256 386cc64d...). Sampling T 1.0, top-p 0.95, top-k 20, logprobs.

**Tasks (drawn 2026-11-20 21:40, held-out first).** Held-out: 48 fresh train-side tasks, 12 per domain (legal 19181,
support 19182, accounting 19183, compiled 19184), never practised. Pool: 80 fresh train-side tasks by round 1's
learning-signal pool rule: legal deadlines rung 0 x 20 (19101), accounting rung -1 x 20 (19102), support refund rung 2 x 20
(19103), legal x 10 (19104) and accounting x 10 (19105) at the default rung. No name is shared between the two sets.

**Practice.** A2s1, G = 8 (640 trials), practice seeds 19201-19205, `want_logprobs` and policy traces.

**Dataset.** Policy-gradient view, `--reward shaped-partial` (C 0.5, floors from `naive_floor`), `--sampled-only`, max
block 32768. Reported: records kept and left out (`proxy_tokens`), share of supervised tokens kept.

**Train (2 seeds, separate cards).** `--recipe on-policy-rl --behaviour recorded --reference start --dual-clip 3
--max-step-kl 0.1 --max-step-grad-norm 5 --records-per-step 8 --eval-every 0 --patience 0`, recipe lr 1e-4, seeds 1 and 2.
Turn-level credit: splinter 7220b1d5d has none (neither `dataset build` nor `train` takes one; every action of a trial
carries its trial's advantage), so the round runs without it. Steps N, fixed before launch from the kept records K:
N = min(2K/8, 164) (164 x 175 s = 8 GPU-h), lowered if needed so that training ends by 09:00 on 11-21 (the lease ends
about 14:15 and the evaluation needs about 4 h on 4 cards). N is recorded with the launch. Targets: median KL 0.02-0.05,
median clip 0.10-0.20, no guard skips; `clip_fraction_dual` reported. No setting is tuned on a result of this round.

**Control.** RFT: the sft-step view of this round's verified passes (strict pass only, no teacher data), from A2s1, lr
2e-4, max block 16384, 8 records per step, the same N steps, patience 0, seed 1.

**Evaluation (arms A2s1, rl-s1, rl-s2, rft; one build).**
- Held-out: the 48 tasks x 8, sampling seeds per domain 19281-19284 shared by every arm; paired by task against A2s1,
  task-cluster bootstrap 95% (10,000 resamples).
- Training tasks: the 40 tasks of the pool with the highest learning signal `LS_i` x 8, seeds 19291-19295 shared.
- Retention: orchestration 32 + workspace-tools-reference 59, 2 attempts, lane R's dev profile (sven-5889a139),
  non-inferior when the paired lower bound is above -0.10.
- Per domain and family, timeouts, probe share and sampled-token NLL for every arm.

**Decision rule (item 19's acceptance).** The RL arm is the two seeds' per-task mean pass rate. Accept when all hold:
held-out gain against A2s1 >= 0.05 with the 95% lower bound above 0; retention non-inferior at 0.10; each seed's median KL
in 0.02-0.05 with no guard skips. Each seed is also reported alone, and RL against RFT as a secondary contrast. Anything
else is "no teacher-free gain" for this round. Not a sealed claim either way.

## Item 21: knowledge-assisted self-learning (plan 2026-11-20)

Distillation through a written knowledge base, no trajectory teacher. Design from the 2026-11-20 ML review, with the
user's changes of the same day (they override the review where the two differ). Nothing here has been run.

**Goal (the user's words).** An agentic system that has the resources it needs to derive correct answers on its own, which
is then taught the skill itself. Success means it derives answers without the knowledge base on held-out tasks: that proves
the system learned on its own. Not claimed: "no larger model" (a larger model wrote the collection).

**The knowledge input is interchangeable.** Bench and sven need only a directory, optionally with `index.md` files. Bench
mounts the directory given by an explicit CLI parameter (for example `--knowledge <dir>`, never an environment variable)
read-only at `/knowledge`, outside the workspace and without a `.git`, and records its digest in every launch record.
Nothing may depend on a collection's folder names, front-matter or tiers; an experiment that wants a subset builds a
filtered copy. One possible input: the private HF dataset `swedishembedded/knowledge` at revision `v0.1` (commit c392495:
20 pages of 600-1000 tokens, a flat index, rule / procedure / worked example / common mistakes per page, sections marked
tier 1-3, a reviewer index; lint and the author's leakage self-audit clean; no human expert review yet).

**Grounding instruction** (system prompt, held in the harness or profile config, in no task; one line in the read tool's
description names the mount): "A knowledge collection is mounted read-only at `<path>`. Start from `<path>/index.md`, use
grep and the folder index.md files to find what applies, and read the relevant files. Before each decision that depends on
a rule or procedure, state that rule in one or two sentences in your own words, then act. If the collection does not cover
something, say so and use your judgement. Do not read more than four pages." It names no domain and no file. It is a
harness change, hence its own arm.

**Examiner and library.** Information flows one way. Bench may read the whole collection freely (contamination audit,
coverage of held-out failures, judging whether a task is derivable from it). Bench's oracles, seeds and item-level results
never reach the collection or the learner. Gate before any learner sees a collection: bench's contamination audit against
sealed exams passes (bench owns and runs it; the author's self-audit is not that gate). An open-book exam also needs the
exam plan to pre-register it as open-book. Tier c then means "held out from trajectories", not "held out from the
collection".

**Lineages.** Primary: from the base Qwen3-4B, because A2 holds 27B demonstrations and "learned on its own" needs a start
without them; its rehearsal is its own earlier verified solutions, and its matched control also starts from the base.
Secondary: from A2 (rehearsal: A2's teacher set). No 27B arm or contrast in this item.

**Context arms** (same weights, same step and token caps; the step cap is raised by the allowed reads in every arm): none;
collection mounted; collection + instruction; placebo (same instruction, an irrelevant library of the same size); oracle
page (the page for the task's family pre-loaded: separates "cannot find" from "cannot use"); and the tier copies T1, T1+T2,
T1-T3 against the whole collection and against none.

**Training views** of each verified open-book trajectory: open-book (prompt and reads as served); closed-book (no-knowledge
prompt, read calls and results removed, the model's own rule restatements kept); reads stripped with nothing in their place
(ablation only). Train a mixture of open and closed, starting near 50/50. Primary contrast: against a matched control that
does RFT on its own closed-book passes with the same rollout count, steps and rehearsal, since plain RFT alone showed
+0.039.

**Continuous-improvement loop (the multi-generation design).**

1. Generation k practises open-book on the train pool.
2. It trains on its own verified solutions (mixture view).
3. It is evaluated closed-book on held-out tasks: the primary measure. Open-book on the same tasks is reported beside it.
4. For each train-side task it still fails: does the collection already hold the guidance needed to derive the answer? If
   yes, the task stays in the pool for generation k+1. If no, record a gap; gaps are filled only from train-side evidence,
   in a new collection version, and a fresh sealed slice is burned afterwards. Coverage of held-out failures is checked by
   bench alone and reported as aggregates per lineage; the learner never sees held-out items.
5. Generation k+1 starts from generation k's weights, gets the collection again and must solve more of the pool.

Success: the number of train-pool tasks solved open-book rises generation over generation and the closed-book held-out
score rises with it; the slope of both is pre-registered. Starting from generation k's weights departs from the review
("retrain from A2 with accumulated data"), so collapse and forgetting are the risks to monitor every generation: pass@8,
distinct-solution count, retention (non-inferior at 0.10, closed and open). Fallback when retention fails: retrain from the
base model on the accumulated data.

**Reported separately:** retrieval gain (collection against none, same weights); instruction gain (collection + instruction
against collection); non-specific gain (placebo against none); internalised gain (generation k closed-book against the
matched control closed-book); open-book gain (generation k open against generation 0 open); residual reliance (generation k
open against closed). Weights are compared only within one context, contexts only within one set of weights. If the
zero-training context effect explains at least 70% of a gain, the headline is a harness result.

**Stages** (GPU-hours are the review's estimates less the removed 27B cells; about 150-180 in all):

| Stage | Work | Go / stop |
|---|---|---|
| S0, no GPU | freeze the collection digest; bench's audit; token-budget lint; the three views and a lint that fails the build on a knowledge path or read call in a closed-book record; rendered-prefix byte equality between a training record and a live request per arm; a few smoke trials | all green |
| S1 zero-training check, about 15-20 | 48 fresh train tasks x 8 (legal 16, accounting 16, support 8, compiled 8); base and A2 under all context arms | go: collection + instruction against none at least +0.06 pass@1, or pass@8 up at least 0.05 with fewer never-solved tasks, probe share under 2%. Only the oracle page helps: fix the index, rerun once. Nothing helps, oracle page included: the content is the problem, back to the author |
| S2 generation 1, about 45-55 | 160 train tasks x 8 open-book; matched closed-book control rollouts; train mixture, stripped-only, open-only and control; 96 x 8 dev closed and open plus retention | go: mixture closed-book minus control at least +0.04 with CI above 0, retention non-inferior, closed-book knowledge-path attempts under 1%. Stop: contrast at most 0 and open-book no better than generation 0 open-book |
| S3 generations 2-3, about 35 each | the loop above, best view only | go: dev non-inferior to the previous generation and open-book train-pool pass@8 rising. Stop: two flat generations (or the fallback) |
| S4 confirmation, about 20-30 | three seeds of the final generation and of the control; sealed exam closed-book (primary) and, if registered, open-book | the primary contrast below |
| S5 | RL on knowledge trajectories | only after item 19's acceptance is met |

**Statistics.** Primary, one contrast: final generation against the matched control, closed-book, tier c, sealed; paired by
item, lineage-cluster bootstrap, one-sided alpha 0.05, minimum gain 0.05; an exact sign or permutation test over lineages
beside it. Secondary, Holm-corrected: the generation trend (slope of generation k minus the generation-0 anchor re-examined
on every fresh slice, k = 1..3 or 4, permutation over lineages; conjunction G3 - G1 above 0 with each step non-inferior at
0.03), the open-book gain, tier subsets, per-domain splits. Gates use 96 tasks x 8 (MDE about 0.05). One seed per
generation while developing.

**Checklist (the review's, verbatim except where the user's changes above replace a line; replaced text in brackets).**

Do this:

1. Call the result "distillation through a written knowledge base, no trajectory teacher"; record the KB digest in every
   launch.
2. Run S0, then the S1 zero-training check, on [the base and A2], with placebo and oracle-page cells, before any training.
3. Put the grounding instruction in the system prompt; make it ask for the rule in the model's own words and cap reads at
   four.
4. Keep the index under about 1.2k tokens and pages under about 1k; use file navigation with existing tools.
5. Train a mixture of open-book and closed-book views (reads removed, the model's own rule restatement kept); keep
   [rehearsal; accumulate data; continue from the previous generation's weights, with retraining from the base as the
   fallback].
6. Lint closed-book records for KB paths and read calls; fail the dataset build on any hit.
7. Assert rendered-prefix byte equality between training records and live requests, per arm.
8. Compare weights only within one context condition; make the primary contrast final generation against a matched no-KB
   RFT control, closed-book, tier c, sealed.
9. Use the same step and token caps in all arms; report tokens per arm; count timeouts as failures.
10. Run the bench-side overlap scan of the KB against exam material, aggregates only; burn a fresh slice after any KB
    revision.
11. Track per generation: probe share, closed-book KB-path attempts, pass@8, per-family coverage, pages read by passing and
    failing trials, retention.
12. Pre-register the single-step contrast as primary and the generation trend as secondary, with an anchor on every slice
    and a permutation test over lineages.

Never do this:

1. Never compare a KB-trained model against [its starting weights] alone without the matched RFT control.
2. Never train on trajectories with the reads stripped and nothing in their place as the main target.
3. Never let a larger model rewrite the 4B's reasoning for the closed-book view.
4. Never edit the KB after seeing dev or exam results from held-out tasks.
5. Never run RL on KB trajectories before the ratio-spike cause is found and `behaviour: recorded` covers all supervised
   tokens.
6. Never mention the KB, its paths or `index.md` in the closed-book prompt or closed-book training records.
7. Never mount the KB inside the task workspace or with a git directory.
8. Never give the KB arms a larger step, token or time budget than the others without a placebo arm.
9. Never read a flat exam result before reading the optimiser's KL and clip, the read rate and the page-use logs.

**Later, not in this item:** a 27B-with-knowledge run, a separate experiment about the 27B itself.

**Prerequisites elsewhere:** bench: `--knowledge <dir>` with the digest in the launch record, the contamination audit and
aggregate coverage report, open-book exam plans. sven: the instruction as profile configuration, the read cap, restatements
as visible assistant content if the template drops earlier reasoning. splinter: the three views and their lint, the matched
control, the gap ledger.

### Item 21 S1: zero-training check (pre-registered 2026-11-20, lane S1, n110, before any job ran)

**Material.** 48 fresh train-side tasks drawn with bench 3fa3820c (`bench task practice --count N --seed S`): legal 16
(seed 9811, default rung), accounting 16 (9812, default rung), compiled 8 (9814, default rung), support 8 (9813, rung 2,
family `refund-eligibility`: A2 is 0.885 on support at the default rung, 0.71 at refund rung 2 per the dial check). 8
attempts per task; per-trial seeds from one run seed per domain (legal 7211, accounting 7212, support 7213, compiled 7214),
identical in every cell.

**Serving and harness, identical in every cell.** brain-5e25d5b82 (frozen, `--max-batch 64`), one server per GH200, base
Qwen3-4B with A2s1 registered on the same server; sven-a0235c71 `compact`, sampling T 1.0, top-p 0.95, top-k 20,
logprobs; splinter practice executor (this commit's `tasks harbor practice --knowledge`). Caps: sven's default 200 tool
rounds (median use about 11, so the four allowed reads need no raise), the task's own agent time limit, the same context
window. A model-caused timeout counts as a failure.

**Cells.** A2s1: (a) none; (b) collection mounted, no instruction (profile without `knowledge_prompt`); (c) collection +
instruction (`profiles/brain-local-knowledge.toml`'s `knowledge_prompt`, verbatim); (d) placebo: the same instruction with a
same-shape public-domain collection (Franklin's Autobiography, Project Gutenberg #148: 6 folders with an index each, 20
passages, 27 files, 97 kB against the collection's 95 kB; no task-domain rule and no workflow advice; digest
`sha256:25323eca987a26b5598c842bde22eb9ee09ac1784e209ac9c849003fc803c126`); (e) oracle page: the instruction with a mount
that holds only the task's domain folder, `calculation/` and `agent-workflow/`, root index reduced to those entries
(digests: legal `sha256:8c7cd1c1...`, accounting `sha256:a9fb111a...`, support `sha256:760df516...`, compiled
`sha256:c6b2b6fb...`). Base Qwen3-4B: (a) none; (c) collection + instruction. Collection: `swedishembedded/knowledge@v0.2`,
`kb.py export --max-tier 3`, digest `sha256:262ff959b54abf6d36c400a6e4f32fe12a155899ac4c601040ad6f2bb01bade8`.
Interleaving: all 28 (cell, domain) runs at once, cell k's domain d on card (d + k) mod 4, so every cell spans all four
cards in the same time slot.

**Measures.** pass@1 = mean over tasks of the per-task pass rate over graded trials (an infrastructure failure is reported
and left out; a timeout is a failure); pass@8 = share of tasks solved at least once; never-solved = tasks with 0 of 8.
Paired differences by task with a 95% task bootstrap (10,000 resamples) and a sign-flip permutation test over task
families (the generator family, company name removed). Per domain. Mean completion tokens and wall time per trial.
Timeouts; probe share (trials with a git-history probe); `tool_call_rescued_from_think` share (trials with a response that
has calls, reasoning and no text). Reads: files under `/knowledge` opened per trial, share of trials that read `index.md`,
pages per trial, pass rate split by whether a page of the task's domain was read.

**Decision rule (fixed now).**
- Go: A2(c) - A2(a) >= +0.06 pass@1, or pass@8 up >= 0.05 with fewer never-solved tasks; and the probe share of A2(c)
  under 2%.
- Navigation repair: only (e) beats (a) (the go margin on (e) - (a), with (c) failing it): fix the index and rerun once.
- Stop: neither (c) nor (e) beats (a): the content is the problem, back to the author.
- Reported beside it: availability gain (b) - (a), instruction gain (c) - (b), non-specific gain (d) - (a); base (c) - (a).
  If (c) - (a) clears the go margin but (c) - (d) does not, the gain is not specific to the collection and the result is
  recorded as a harness effect, not a go.

**S1 results (2026-11-20, runs 14:37-18:10, about 14 GPU-h on 4 GH200s; splinter 7220b1d5d, evidence
`~/project/shared/archive/lane-s1-2026-11-20.tar.zst`).** 2,688 trials, 17 infrastructure failures (left out), all 28 runs
exit 0, every open-book run's report records the digest it mounted.

| Cell | pass@1 | pass@8 | never | legal | acct | support | compiled | timeouts | probe | rescued | compl. tok | wall s | touched /knowledge | read index.md | pages/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 (a) none | 0.529 | 0.812 | 9 | 0.484 | 0.312 | 0.719 | 0.859 | 19 | 0.013 | 0.146 | 13,608 | 299 | 0 | 0 | 0 |
| A2 (b) mounted | 0.522 | 0.771 | 11 | 0.445 | 0.317 | 0.719 | 0.891 | 12 | 0.034 | 0.154 | 14,790 | 301 | 0.00 | 0.00 | 0.00 |
| A2 (c) mounted + instruction | 0.527 | 0.812 | 9 | 0.432 | 0.359 | 0.703 | 0.875 | 13 | 0.036 | 0.133 | 13,238 | 271 | 0.06 | 0.03 | 0.01 |
| A2 (d) placebo + instruction | 0.514 | 0.792 | 10 | 0.424 | 0.352 | 0.672 | 0.859 | 9 | 0.013 | 0.094 | 14,149 | 283 | 0.08 | 0.05 | 0.10 |
| A2 (e) oracle mount + instruction | 0.517 | 0.771 | 11 | 0.445 | 0.372 | 0.656 | 0.812 | 15 | 0.026 | 0.125 | 13,036 | 296 | 0.08 | 0.04 | 0.03 |
| base (a) none | 0.044 | 0.188 | 39 | 0.008 | 0.086 | 0.078 | 0.000 | 0 | 0.000 | 0.018 | 5,076 | 122 | 0 | 0 | 0 |
| base (c) mounted + instruction | 0.036 | 0.125 | 42 | 0.023 | 0.078 | 0.016 | 0.000 | 1 | 0.000 | 0.057 | 4,906 | 121 | 0.13 | 0.13 | 0.20 |

Paired by task (95% task bootstrap; sign-flip over task families, two-sided):

| Contrast | delta pass@1 | delta pass@8 | family p |
|---|---|---|---|
| A2 (c) - (a), primary | -0.002 [-0.038, +0.035] | 0.000 [-0.083, +0.083] | 0.96 |
| A2 (b) - (a), availability | -0.006 [-0.063, +0.052] | -0.042 [-0.146, +0.062] | 0.97 |
| A2 (c) - (b), instruction | +0.004 [-0.045, +0.053] | +0.042 [-0.042, +0.125] | 0.92 |
| A2 (d) - (a), non-specific | -0.015 [-0.062, +0.033] | -0.021 [-0.104, +0.062] | 0.80 |
| A2 (c) - (d), specific | +0.013 [-0.036, +0.061] | +0.021 [-0.042, +0.083] | 0.73 |
| A2 (e) - (a), oracle mount | -0.011 [-0.065, +0.043] | -0.042 [-0.146, +0.062] | 0.74 |
| base (c) - (a) | -0.008 [-0.031, +0.013] | -0.062 [-0.146, +0.021] | 1.00 |

Reads (knowledge 191): A2 (c) opened `index.md` in 11 of 382 trials and no domain page; several calls named paths that do
not exist (`/knowledge/reference/rule_table.md`, the task's own file names under the mount). A2 (e) opened a domain page in
2 trials (both failed). Base (c) opened `index.md` in 50 trials and a domain page in 37 (`compiled/apply-rules.md` 29),
and passed none of those 37. The pass-rate split by "read the domain's page" is therefore empty for A2 and 0.000 against
0.040 for base.

**Decision (pre-registered rule): STOP.** Neither (c) nor (e) beats (a); the probe share of A2 (c) is 0.036, also over the
2% cap (A2 (b), without the instruction, 0.034; A2 (a) 0.013). The rule says "back to the author". The reads show that
this null does not test the content: the policies hardly opened it. A2 (trained on 27B demonstrations that never read a
collection) ignores the system-prompt line, and base Qwen3-4B reads but cannot do the tasks (pass@1 0.04). Not supported
by S1: any claim about the collection's quality. Next cheap checks, before any training: (1) the review's oracle-page
*injection* (the domain page pre-loaded into the first user turn, which this run's narrowed mount did not do) to separate
"does not look" from "cannot use"; (2) the read tool's description naming the mount (a sven prerequisite not yet built);
(3) the advisory's reframe, a habit SFT on the few open-book trials that read and passed. No 27B cell was run (user's
exclusion), so "only the 27B gains" was not tested.

**Correction to S1 (2026-11-20).** The pre-registration said the per-trial seeds were identical in every cell. They were
not: bench derives a trial's seed from the run seed and the trial's name (`sha256(f'{run_seed}:{trial}')`), and Harbor's
trial names carry a random suffix. Each cell shared its run seed per domain, but its trial seeds differed. The paired
analysis pairs by task, not by trial, so the contrasts stand; the sampling noise is that of independent attempts.

**S1 probe check.** Every probe counted in S1 was a read of the task workspace's own git history (`git show HEAD`, `git log`,
`git cat-file -p`), in every cell. No call read `/knowledge` and was counted, and bench's leak audit does not match
`/knowledge` (it flags the snapshot mount only), so no classifier needs a fix. Trials with a probe: (a) 5, (b) 13, (c) 14,
(d) 5, (e) 10 of about 382. Cell (b) never touched the mount, so the rise is not caused by reading the collection.

### Item 21 S1b: does not look, or cannot use (pre-registered 2026-11-20, before any job)

**Mechanism.** bench 62da1dbb adds `[sven].knowledge_preload = [PATH...]`: files of the mounted copy appended to the system
prompt after the knowledge prompt, recorded as `knowledge.preload` (path, sha256, bytes) in the run and every trial record.

**Cells, A2s1 only, the same 48 tasks x 8 as S1** (same draws, brain-5e25d5b82, sven-a0235c71 compact, splinter
7220b1d5d, same sampling, run seeds and caps; bench 62da1dbb):
- (a) none, rerun interleaved (S1's (a) ran under bench 3fa3820c in another time slot).
- (f) index preload: the full v0.2 collection mounted, the instruction, `index.md` preloaded.
- (g) page preload: the full collection mounted, the instruction, and the task domain's folder (index and pages) plus the
  `calculation/` pages preloaded. Legal, support and compiled: about 6.0-6.1k tokens (words x 1.3), 26-28 kB.
  Accounting is over the cap (about 7.9k tokens, 35 kB): it gets `accounting/index.md` plus the two pages that fit the task's
  family: accrual `period-end` + `money`, bank-rec `bank-reconciliation` + `money`, consolidation `consolidation` + `money`,
  missing-info-vat-category `vat` + `agent-workflow/missing-information`, trial-balance `journal-errors` + `money`, vat
  `vat` + `money` (one run per family group).
- Interleaving as in S1: cell k's domain d on card (d + k) mod 4, all runs at once.

**Decision rule (fixed now).**
- (g) - (a) >= +0.06 pass@1: the content helps when it is read. The next work is the habit or navigation (preload by default,
  or a habit SFT).
- (g) <= (a): the content does not help A2 even when read. Return to the content, with a per-domain analysis of the
  failures.
- Between the two: inconclusive at this sample size; report the interval.
- (f) - (a) and (g) - (f) say whether the index alone is enough.
Reported as in S1, plus the probe share per cell.

**S1b cell (h), added 2026-11-20 at the user's request, pre-registered before any (h) job.** Same 48 tasks x 8, draws,
builds, caps and run seeds as S1b; the full v0.2 collection mounted, no preload, and in place of the grounding text the
user's wording, verbatim: "A knowledge base is available at /knowledge (start from /knowledge/index.md). Whenever you can
not directly solve the problem, look in the knowledge base for the correct approach. Find it and use it." Placement: bench
can only append to Sven's system prompt (`[sven].knowledge_prompt` becomes `--append-system-prompt`; putting text first
would mean replacing Sven's whole prompt with `--system-prompt-file`), so the text is appended, as (c)'s was. (h) runs
after (a), (f) and (g) started, in a later time slot, so it is not interleaved with its baseline. Decision rule, the same go
rule as S1: (h) - (a) >= +0.06 pass@1, or pass@8 up >= 0.05 with fewer never-solved tasks, with the probe share under 2%.
Also reported: (h) - (c) against S1's (c) (another time slot and bench 3fa3820c, so indicative only), and the read rate
(trials touching `/knowledge`, reading `index.md`, opening a domain page) against (c)'s 6% / 3% / 0%.

**S1b results (2026-11-20, runs 18:46-22:19; bench 62da1dbb, splinter 7220b1d5d; evidence
`~/project/shared/archive/lane-s1b-2026-11-20.tar.zst`).** A2s1, the same 48 tasks x 8.

| Cell | pass@1 | pass@8 | never | legal | acct | support | compiled | timeouts | probe | rescued | compl. tok | wall s | touched /knowledge | index.md | domain page opened |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) none | 0.487 | 0.750 | 12 | 0.432 | 0.350 | 0.547 | 0.812 | 18 | 0.026 | 0.159 | 14,453 | 298 | 0 | 0 | 0 |
| (f) index preload | 0.497 | 0.750 | 12 | 0.438 | 0.273 | 0.719 | 0.844 | 11 | 0.021 | 0.109 | 13,725 | 295 | 0.05 | 0.01 | 0 |
| (g) page preload | 0.517 | 0.833 | 8 | 0.484 | 0.238 | 0.719 | 0.938 | 26 | 0.026 | 0.130 | 14,908 | 308 | 0.04 | 0.00 | 4 trials |
| (h) user wording | 0.531 | 0.854 | 7 | 0.445 | 0.336 | 0.734 | 0.891 | 12 | 0.029 | 0.154 | 13,921 | 238 | 0.02 | 0.02 | 2 trials |

| Contrast | delta pass@1 (95% task bootstrap) | delta pass@8 | family p | legal / acct / support / compiled |
|---|---|---|---|---|
| (g) - (a) | +0.029 [-0.050, +0.105] | +0.083 [0.000, +0.188] | 0.88 | +0.05 / -0.11 / +0.17 / +0.13 |
| (f) - (a) | +0.010 [-0.045, +0.065] | 0.000 | 0.94 | +0.01 / -0.08 / +0.17 / +0.03 |
| (g) - (f) | +0.019 [-0.049, +0.091] | +0.083 [+0.021, +0.167] | 0.81 | |
| (h) - (a) | +0.044 [-0.012, +0.102] | +0.104 [+0.021, +0.188] | 0.10 | +0.01 / -0.02 / +0.19 / +0.08 |
| (h) - S1 (c), other slot | +0.004 [-0.040, +0.044] | | | |
| S1b (a) - S1 (a), same cell, other slot | -0.041 [-0.090, +0.008] | | | |

Trials the leak audit marked `exposed` (a 7, f 2, g 1, h 0) were left out as above. Counted as failures instead, they give
(g) - (a) +0.036 [-0.042, +0.112] and (h) - (a) +0.052 [-0.005, +0.109].

Accounting under (g), per family (pass rate (a) -> (g), 7-16 trials each): vat-single-rate 0.88 -> 0.50, vat-boundary 0.86 ->
0.43, vat-multi-rate 0.57 -> 0.25, accrual-accrued 1.00 -> 0.12, missing-info-vat-category 0.69 -> 0.50, bank-rec-timing 0.12
-> 0.50; the rest of accounting was at or near 0 in both cells (trial-balance, consolidation, bank-rec-errors).

**Decisions (pre-registered rules).**
- (g): +0.029 is between the bars, so the result is INCONCLUSIVE at this sample size. The baseline itself moved by -0.041
  between two time slots, about the size of the effect. By domain, the preloaded pages helped support (+0.17) and compiled
  (+0.13), but HURT accounting (-0.11): the VAT and accrual families fell sharply with `accounting/vat.md`,
  `period-end.md` and `calculation/money.md` in the prompt. Return the accounting pages to the author with this
  per-family list. Legal, support and compiled are worth a confirmation at 96 tasks.
- (f): the index alone does nothing (+0.010; the policy still opened no domain page).
- (h): no go. Pass@8 rose by 0.104 with fewer never-solved tasks (12 -> 7), but pass@1 is +0.044 and the probe share is
  0.029, over the 2% cap (the baseline's is 0.026 in this slot). The conditional wording LOWERED look-ups: 2% of trials
  touched `/knowledge` and 2 opened a domain page, against (c)'s 6%. So whatever (h) gained did not come through the
  collection, and (h) is no different from S1's (c) (+0.004). It ran after (a), in a less loaded slot (wall 238 s against
  298 s), so its gain is not separated from load. The text was appended, not placed first: bench can only append to Sven's
  system prompt without replacing it.
- Probes: all S1b probes are again reads of the workspace's own git history; the cap is breached in the baseline too, so
  it measures the policy, not the mount.

**Run note.** The lane's driver script was edited while three runs were inside it (knowledge 192). Their logs were
truncated and they recorded rc 127, but their reports hold every trial and were harvested. The first (h) accounting run was
stopped at 24 of 128 trials and rerun as three task subsets on three cards; the stopped run is not analysed.

### Item 21 S1c: confirmation with knowledge v0.3 (pre-registered 2026-11-20, before any job)

**Collection.** `swedishembedded/knowledge@v0.3` (94bc175): every page opens with "the task's files win", worked examples
are marked illustration only, `period-end` is split into `accrued-costs`, `prepaid-costs` and `depreciation`, and `money` and
`vat` are rewritten; bench's leakage audit passed on 8 sealed exams. Exports: tier 3
`sha256:625bb3ad5dedfa7aaa84a2bf2b5c4eb270f2da598c3f047838016912d15d206f` (29 files), tier 2 (no worked examples)
`sha256:e170326a2790f7229e88644e7d944cdc163e278fb085cb2061a6a6b960d0973b`.

**Material.** 96 fresh train-side tasks, 24 per domain (bench 62da1dbb; seeds 9831 legal, 9832 accounting, 9833 support at
refund-eligibility rung 2, 9834 compiled; default rungs otherwise), 8 attempts; A2s1 on brain-5e25d5b82, sven-a0235c71,
splinter 7220b1d5d; run seeds 7311-7314 per domain; caps as in S1.

**Cells.** (a) none; (g3) v0.3 tier 3 mounted, the grounding instruction, pages preloaded as in (g): the domain folder plus
the `calculation/` pages for legal, support and compiled (about 6.0k tokens). Accounting is over the 32 KiB cap (about 9.9k
tokens), so it gets `accounting/index.md` plus the two pages for the task's family: accrued `accrued-costs` + `money`,
prepaid `prepaid-costs` + `money`, depreciation `depreciation` + `money`, bank-rec `bank-reconciliation` + `money`,
consolidation `consolidation` + `money`, missing-info-vat-category `vat` + `agent-workflow/missing-information`,
trial-balance `journal-errors` + `money`, vat `vat` + `money`. (g3t2) the same paths from the tier-2 export. (h3) the user's
wording (S1b (h), appended), v0.3 tier 3 mounted, no preload.

**Interleaving.** The 96 tasks form 8 units of 12, half a domain each. A card runs all four cells of one unit at once (10
trials at a time per cell), and takes the next unit (accounting first) when it is done. Every task's trials in every cell
therefore share a card and a time slot. Cards are taken from lane R19 as it frees them.

**Decision rule (fixed now).**
- Primary, (g3) - (a): go if pass@1 >= +0.06, or pass@8 >= +0.05 with fewer never-solved tasks; and probe share (g3) minus
  probe share (a) under +0.02. **Change from S1:** the probe gate is on the difference against (a), not on the absolute
  share, because S1b's baseline itself was at 0.026, over the 2% cap. Every probe so far was a read of the workspace's git
  history by the policy, in every cell, so the gate should catch what the context adds, not what the policy already does.
- Secondary, Holm-corrected over two family sign-flip tests: (g3t2) - (g3), (h3) - (a).
- Reported: per domain, and accounting's VAT and accrual families against S1b's harm; steps per trial (the v0.2 pages
  doubled to quadrupled steps in accounting); the same read, timeout, token and rescued-call metrics as S1.
