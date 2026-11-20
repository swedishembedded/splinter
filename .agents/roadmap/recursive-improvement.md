<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Recursive self-improvement: what failed, why, and the plan that closes the gaps

Scope: proving that a model improves itself across generations with splinter, brain and sven (registered as
RSI1 in `papers/splinter-brain-sven-flywheel/prereg/recursive-improvement.md`), the causes found when it did
not, and the work that removes them.

Status is recorded here and in the registration's dated amendments; the run records and analyses are in the
flywheel evidence dataset (`gen27b/`, `rsi1-lua/`) and the curated 27B dataset (`qwen38-27b-flywheel`).

## What has been measured (2026-11-05 .. 11-08)

| Experiment | Result | Why it does not show recursive improvement |
|---|---|---|
| 27B agent loop, 2 generations (repo, repo-hard, firmware) | Gen-0 promotion (+0.034, p 0.047) did not reproduce; on one stack gen-1 vs base 13/16, gen-2 vs base 17/14 | Exam near ceiling (base ~0.9); generation 0 compared against a base exam run on another brain build whose OOMs ended base trials; tiny updates (KL ~0.007); 114 of 138 training tasks always solved (no group signal) |
| 27B efficiency (exploratory) | Tokens per solved task fell base > gen1 > gen2 (~10-13 %, sign p 0.02 / <0.0001 / 0.02) | Found after the data; registered as co-primary H-EFF, not yet confirmed |
| 8B agent on repo tasks (S0) | 25/530 repo, 0/479 repo-hard | No headroom for a small agent model on these tasks |
| Lua closed-book arm A (4B, 4 replicates) | Gen1 +8.75, gen2 +4.75 (p .001), gen3 -0.5 | Plateau; and see the re-grade below |
| Lua arm B (starvation fix c7d11d932) | Gen1 +11, then ~+1, ~0 | Same plateau, reached faster |
| Neutral re-grade of A/B | Base ~57-61 % (no thinking) / ~71 % (thinking, pass@16 80-84 %) on the same probes vs ~40 % under splinter's probing; champions ~= base | **The A/B gains were adaptation to splinter's probing protocol, not added competence.** The paper's earlier Lua claims (SL20-SL23, 8B 25 -> 49) need the same re-grade |
| Arm H (exam = 889 code-written items the base never answered, neutral grading) | **Criterion met (2026-11-08 17:50):** gen1 38/31/34, gen2 60/63/49, gen3 67/83/59 of 889 (base 0); pooled sign tests gen1 vs base 103/0 p 1e-31, gen2 vs gen1 95/26 p 1e-10, gen3 vs gen2 103/66 p 0.0027; no replicate fell | Shown: three self-taught generations, each significantly better on what the base could not do. Open: forgetting (26 then 66 losses), absolute level still low (7-9 %), thinking off; arms R and P test search, procedures and negative rewards |
| Arm HP (arm H on Python, 4 replicates, 820-item exam, base 2) | Gen1 vs base 49/7 (p 4e-9), gen2 vs gen1 83/8 (p 4e-17); **gen3 vs gen2 59/50 (p 0.22), r0 fell** - criterion not met (2026-11-09) | Two generations of self-improvement replicate in a second language; the third generation forgets as much as it gains, invisible to the loop's own probes. One replicate (r1) never got past generation 1 |
| Arm HR (arm HP + replay + pooled gate) | Registered criterion met (66/4, 16/1, 25/3; no replicate fell); losses per step 4/1/3 vs HP 7/8/50 | Forgetting fixed, but each step's gain is a different replicate (r0, r3, r2) and no lineage improved twice; HR accepted 11 rounds in 12 runs vs HP 30. The criterion must be per lineage: the same replicate better at gen1, gen2 and gen3 |
| Arm H per lineage (exploratory, 2026-11-10; not the registered test) | Within-replicate sign tests: r1 31 -> 63 -> 83 (31/0 p 5e-10, 40/8 p 2e-6, 49/29 p 0.015); r0 38 -> 60 -> 67 (gen3 26/19 p 0.19); r2 34 -> 49 -> 59 (gen3 28/18 p 0.09) | One Lua lineage improved significantly at three consecutive generations, and every lineage rose at every step; post hoc, so it needs a registered per-lineage replication |
| Arm HL (registered per-lineage replication, 2026-11-10; rerun 2026-11-13) | First attempt (2026-11-10): stopped by the project disk's inode quota in every replicate before generation 3; only r1 generation 1 measured (46 of 889, base 0, p 1.4e-14). Rerun (2026-11-13, cleared disk, same scripts and binaries: splinter b39782f43, brain 69b1834fa): **criterion not met, 1 of 4 lineages succeeded (2 required).** r3 16/0, 14/5 (p 0.032), 33/12 (p 0.0012) succeeds; r2 38/0, 62/16 (p 7.6e-8), then 27/36 at gen3; r0 59/0, 35/25 (p 0.12), 37/28 (p 0.16); r1 36/0 then no round accepted in gen2/gen3. Pooled gen1 149/0, gen2 111/46 (p 1.1e-7), gen3 97/76 (p 0.064) | Third-generation steps are fragile: one lineage of four improves three times in a row; gains at gen3 are offset by losses (forgetting, family trades) and some lineages stall (no round accepted). Records: evidence dataset `rsi1-hl2/` (swedishembedded/splinter-flywheel-evidence, commit 0fa5f71e) |
| RSI2 (on-policy PG, served brain, bench-examined, 2026-11-13) | Criterion not met: generation 1 vs base 4/0, 2/0, 3/0, 3/0 (p 0.0625-0.25) on the 889-item exam; generation 2 blocked by a shared adapter card id (fixed 1e5883c98) | On-policy RL sharpened the learner's own dev set (0.49 to 0.56) but barely reached items the base never answers; self-proposed tasks reproduced 1-2 exam programs in three lineages (bench's overlap audit) |
| RSI3 (self-taught SFT on interpreter labels of the policy's own proposals, bench-examined with overlap exclusion, 2026-11-13) | Criterion not met: 2 of 4 lineages improved at all three generations (r2 0->67->86->121, r3 0->91->118->142); r0 flat and r1 fell at gen 3; pooled 288/0, 199/60, 142/92 (p 2e-87, 7e-19, 0.00065); 0 overlap | Every lineage improved twice; generation-3 steps stall as the policy's own proposals stop carrying new information (568 -> 229 mispredicted) and continued training drifts; RSI4 retrains from the base on frontier data |
| RSI4 (cumulative retraining from the base on the frontier of the policy's own proposals, geometric frontier schedule 300/600/1200, bench-examined, 2026-11-15) | **Criterion met**: 4 of 4 lineages improved at all three generations (r0 12->92->138->192, r1 12->131->181->225, r2 12->118->169->241, r3 12->121->187->218); pooled 422/8, 320/107, 354/153 (p 1e-113, 5e-26, 1e-19); signed plan verdict plan_met true | Generation 3 no longer stalls; the dev gate under-predicts exam gains at generation 3 (r1 dev p .23, exam p 2.5e-4); next: X7 gate comparison, other tasks/models, more generations |

## Root causes (ranked by evidence)

1. **Measuring the wrong thing.** Exams at ceiling (27B), and a probing protocol the model can adapt to (Lua A/B).
   Every claim needs a neutral measurement of the same items, the base measured identically, and an exam drawn
   from what the base cannot do.
2. **No frontier.** Most tasks were always solved or never solved; a group-advantage or self-imitation signal
   exists only on tasks solved sometimes.
3. **Narrow sampling.** 2-4 attempts per task; mixed groups were rare.
4. **Search switched off.** All Lua runs trained and measured without thinking; the model could only learn an
   input-to-output mapping. With thinking the same base answers ~71 % of the probes.
5. **Training on answers, not procedures.** Acquisition keeps one verified final answer per task (the shortest)
   as `SftFinal`; reasoning, intermediate states and repairs are discarded.
6. **Tiny or noisy updates.** Lua: lr 1e-6 x 6 steps with 3-4 records per round; round acceptance on 3-4 dev
   probes per skill; retention bound 0.5 (default 0.25).
7. **Starved self-generation** (fixed in c7d11d932): 66 % of proposals refused, mostly repeats; a round stopped
   at the first barren request and later rounds were not told what earlier rounds had admitted.

Not a cause: relabelling from execution already exists (a mispredicted program is admitted with the
interpreter's output, since 81c611b70; b39782f43 records and counts it).

## Plan

### In flight (2026-11-08)

- **Arm HP** (registered 758ea640e, 7835c9bfe; restarted 71493424b): the arm-H recipe on Python (126 skills from
  Markdown sources), 4 replicates, exam 820 items the base never answered (base greedy 2/820). The first runs spent
  their budget writing probes; generations 1-3 now reuse those probes (`--probes-from`) with 3 h each.

- **Arm H** (registered 11d5f77ff, accfaeabc): the arm-B loop measured on the 889-item base-never exam; criterion
  gen1 > base, gen2 > gen1, gen3 > gen2 (pooled sign test p < 0.05, no replicate falling).
- **Arm R** (registered 53135019e): frontier tasks (solved 1..15 of 16), 16 samples per task with thinking, training
  on the model's own verified reasoning traces, adapter retrained from the base on all traces so far (ReST-EM),
  measured on fresh items of trained families, six never-trained families and the arm-H exam.
- **brain: policy-gradient training on qwen3 bases** (rollouts with advantages, negative rewards push down) -
  today refused ("wired for qwen35 bases, not qwen3"). Then a **policy-gradient arm** (GRPO/DAPO: dynamic sampling
  of mixed groups, clip-higher, token-level loss, k = 16, thinking on) on arm R's pools.

### splinter

- [ ] Promote arm R's driver (`rsi1/rl/star.py`, a script today) into `splinter learn` as an automatic strategy:
  frontier selection by measured pass@k, wide sampling, thinking-on traces, ReST-EM retraining, plateau detection.
- [ ] Dataset views beyond `sft-final`: `sft-diagnose`, `sft-execute` (predict intermediate state), `sft-verify`,
  `policy-trajectory`. Done (2026-11-13): `sft-repair` (wrong attempt + the verifier's feedback -> the fix a later
  attempt found) as a record kind of the campaign controller (`campaign::records`), `learn --answers-per-task N`
  keeps the N shortest distinct verified answers of a task instead of only the shortest (default 1, unchanged),
  positive-only against two-sided advantages are an explicit option (`AdvantageMode`, `--advantage positive`), and
  fresh prompts before revisits are the controller's draw order. An experience-corpus `sft-repair` view for
  `dataset build` is still open.
- [ ] Procedure data from the environment: interpreter-derived execution traces (each sub-call's value) as
  reasoning targets; counterexample generation for stuck skills.
- [ ] Measurement: power-sized dev and confirmation suites (24-64 items per skill), pass@1 / pass@16 / pass@64,
  held-out families and unseen compositions, the base re-measured with every generation under a neutral protocol,
  and the protocol a model is used with measured beside it.
- [ ] Separate exploration from the champion: an explorer policy (base, earlier checkpoints, higher temperature)
  feeding a shared verified store; promotion only through an independent gate.
- [x] Forgetting controls (65c60a2a2): cross-generation replay (`--replay-from RUN`; before it every generation started
  with an empty replay list, so generation N never replayed generations 1..N-1's accepted data) and a pooled retention
  gate (`--pooled-retention-drop`, default 0: any net loss of passes over the other skills rejects; the per-skill
  bound let a loss spread over many skills through). Arms H and HP lost 50-66 items at generation 3 while their
  probes rose; their losses are family trades (r0 gained `list * and +=` and lost `format spec`, r2 the reverse).
- [x] Arm HR (ran 2026-11-09/10, outcome above): arm HP's design with `--replay-from <previous generation>` and the pooled
  gate, 4 replicates x 3 generations from the base (about 10-11 h on 4 GH200s; the arm-HP runs predate the dataset
  field in round records and cannot be replayed from). Criterion as arm H; the decisive comparison is generation 3.
- [ ] Learn more per generation without the forgetting: most rejections are a flat target on 3-4 dev probes or
  the per-skill bound, not the pooled gate. Power the dev probes (more per skill) so real gains pass, and register
  the next arm's criterion per lineage (each replicate better at every generation, sign test within the lineage).
- [ ] Ablation: arm H without interpreter relabelling of mispredicted programs, to test whether the corrected answers
  to the model's own mistakes are what drives arms H and HP.
- [ ] Stricter round acceptance (no promotion on one dev probe) and the 0.25 retention default restored.
- [x] A plateau controller that switches strategy (task distribution, external evidence, repair training,
  exploration breadth, objective) when the discovery rate per GPU-hour falls: `eval::campaign::plateau` reads
  discoveries per GPU-hour over a window against the best window, and the controller moves to the next strategy of
  its ladder (2026-11-13). Which ladder is best is not measured.
- [ ] Re-grade the paper's Lua claims (SL20-SL23, the 8B run) under a neutral protocol before keeping them.

### Campaign controller (2026-11-13)

`splinter_pipelines::campaign` (exposed as `splinter_sdk::campaign`) runs learning as one continuous, resumable
campaign instead of rounds glued together by scripts (`scripts/campaigns/flywheel/loop.py` and the `rsi1`
`seq*.py` drivers still work and are kept for their recorded arms; new campaigns use the controller).

- Done: typed state in one append-only pack per campaign (`store::campaign`: state, head and ledger lines under a
  lock, compare-and-swap on the head so a second controller loses instead of corrupting, torn tails cut,
  compaction); a crash resumes at the arm start, or at the gate when training had finished; a changed
  configuration is refused. Per-task pass-rate state with an offline calibration at k = 8-16 and permanent
  retirement at 0.9 (`eval::campaign::pass_rate`). Dynamic sampling: only groups with a reward spread enter a
  batch, over-provisioned and refilled, all-fail groups routed to skip, teacher, cliff-distill or decompose
  (`eval::campaign::sampling`). Per-step signals as gate inputs: informative-group fraction, KL, clip fraction,
  entropy (absent when unmeasured; the in-process path records no sampled tokens, so its entropy is absent and brain
  does not report one per step, gap 24). Bounded replay of earlier arms' verified data, the dev-validation gate with
  rollback (`eval::campaign::gate`), checkpoint collection keeping digests, the experiment ledger (hypothesis,
  configuration digest, GPU-hours, cost, verdict), power-first arm planning that refuses an arm whose exam cannot
  detect its target effect, alpha spent by gate number as the default (`eval::campaign::plan`), the plateau
  ladder, and a disk preflight that counts files as well as bytes (`pipelines::preflight`).
- Rule: a campaign collects training episodes through Splinter's episode executor (`episodes::SvenEpisodes`,
  `splinter tasks harbor episodes`: Harbor-format practice folders run through sven's agent, graded outside its reach,
  each attempt an experience with its reward, typed outcome and policy trace) or in process, never by driving `bench
  run` on bench collections (the sealed measurement authority); bench is asked for verdicts on finished candidates
  only. The bench-mediated flow (`bench export` then `experiences ingest-bench`) is legacy for new work. (Corrected
  2026-11-17: an earlier entry named a `sven rollout` command and `sven_sdk::rollout`; neither exists on sven main, and
  running episodes for learning is Splinter's, not sven's.)
- Environments (2026-11-13): `campaign::factory` takes a campaign's tasks from the environment factory store: only
  `Admitted` environments whose recorded base band is inside the learnable range enter (`select` refuses the rest with
  reasons), instances are staged per attempt and removed, every ledger entry carries each environment's verifier
  class and the label the weakest class forces on the verdict, and each gate feeds every environment's training
  reward and dev rate to `DivergenceAlarm` (Diverging: the arm stops, the candidate is not adopted, the environment is
  quarantined in the store and out of the pool). In process the factory path runs the answer-file environments
  (program output); environments whose solver needs a shell or a database go through the episode executor
  (`episodes::SvenEpisodes`), which the factory path does not call yet.
- Collectors: the in-process path (`campaign::brain`: a loaded model sampled at temperature 1 against a
  `TaskFamily`, brain's policy-gradient trainer, a greedy dev exam) and the episode path (`campaign::endpoint`:
  an `EpisodeExecutor` that runs Harbor-format tasks in sandboxes and returns, per episode, the reward and the
  sampled calls' token ids and log-probabilities; an unprovable chain is dropped, never re-tokenized).
- Real weights (2026-11-13, Qwen3-4B, in process, one GH200, `samples/campaign-lua`, findings 143-145): calibration at k = 8
  of 240 tasks, batches of 12 informative groups, 12 steps per arm, lr 1e-5, dev 400 programs. Dev pass rate 0.3375
  (base), 0.385, 0.4475, 0.475, 0.49 over four accepted arms (27/8, 27/2, 11/0 significant at the level each gate spends,
  15/9 not), 0.65 GPU-hours, one pack file for the campaign. With zero retention losses allowed every arm was rolled
  back (one of 40 retention tasks lost each time). Informative-group fraction 0.39 falling to 0.09 as frontier tasks
  are used. Brain's reported KL is 1e2 to 1e7 and a token ratio near 1e-8 appears from step 2 (finding 144): open in
  brain. Entropy is unmeasured in process. Not run: the lr/rank sweep (gap 19), the equal-compute method matrix (gap 20),
  the distillation strategy on real weights.
- Open: the factory path does not hand its shell-solver environments to the episode executor
  (`episodes::SvenEpisodes`, landed 2026-11-17 and tested against a scripted policy only); the training
  records do not yet carry the sampler's behaviour log-probabilities (`WireMessage` has no field for brain's
  `behavior_logprobs`, so every campaign step trains on the start-of-run proxy, exact for on-policy rollouts only);
  a hot refit of the champion's adapter through a serving endpoint (the in-process collector reloads the model on
  each accepted arm); batched in-process sampling (the in-process engine serves one generation at a time, so
  rollouts are not batched; a served endpoint is the batched path); the campaign is not a `splinter` command yet
  (the sample `samples/campaign-lua` is the driver).
- Distillation (gap 21): `LearningStrategy::Distill` and `--all-fail cliff-distill` train the policy's own tokens
  on brain's per-token reverse-KL signal (`A_t = log q_teacher - log p_behaviour`, clipped, capped) with the frozen
  model shown what the verifier knows. Missing in brain for a separate, larger teacher (a 27B teacher for a 4B or 8B
  student): `PolicyGradientConfig::teacher` names only the base or the start-of-run adapter of the same model, so
  the teacher cannot be another checkpoint, and a record cannot carry per-token teacher log-probabilities scored by a
  served teacher (finding 442's scored trace is the source). Splinter has nothing more to wire until brain accepts
  either; no real-weight run of the distillation strategy has been made.

### RSI2 training lane (2026-11-13)

- Done: the lineage driver `rsi2` (`samples/campaign-lua`): policy-proposed `predict-lua` tasks over the manual (parallel waves
  through the served policy) beside the seeded generator, neutral protocol, one `brain serve` per lineage for proposals,
  rollouts, dev exam and the training job, adapter hot-load, a candidate hand-off per generation; the served collector,
  validator (kept results) and trainer in `campaign::served`; findings 146 and 147; launch record `prereg/rsi2-launch.md`.
- Blocked at launch: the project disk's inode quota was at 99-100% from other work (423 files free), and a lineage needs a few
  thousand; `~/project/shared/rsi2/run-lineage.sh SERIES CARD SEED` starts a lineage (generations under the card's flock) once
  there is room. No generation has run; no candidate has been handed over.
- Open: the factory program-output plugin as source (b) (the generator is used); the serve/train numerics gate on the pilot's
  own rollouts (brain's finding 446 is cited); outcome entry after the examiner's verdicts.

### Acceptance gate (2026-11-15)

The lineage's dev gate is greedy in the sense of TRUST (arXiv 2611.10787v1): it advances on any net dev gain over the same
400 units every generation. Against bench's sealed exam it accepted 12 of 12 RSI3 steps, 2 of them not improvements (r0 g3
29/28, r1 g3 18/28), the two smallest dev margins: FER_0 2/12 (finding 154).

- Done: the evidence gate as an opt-in beside the greedy default (`eval::campaign::evidence`, `dev_pool`,
  `gate::decide_evidence`): a noninferiority e-process per round (Y = (1+D)/2, boundary (1-eps)/2 + beta*zeta, the
  equal-weight mixture over betting fractions 0.10/0.25/0.50/0.75, anytime-valid acceptance at 1/alpha_t), levels from a
  false evolution budget q over the planned rounds scaled by the accepted count, and dev items drawn per round without
  replacement from a pool with the used set tracked; alpha_max (= q) and rho in [0.01, 0.5] are parameters the paper
  leaves unstated. Wired as a typed choice into the lineage (`--gate evidence --gate-eps --gate-q --gate-beta --dev-draw
  [--dev-batch]`, several rounds per generation, the champion kept on rejection, used set and budget rebuilt from
  `rounds.jsonl`), the campaign controller (`CampaignConfig::acceptance`, budget and used set in the state, evidence
  measurements in the ledger; a greedy configuration's digest is unchanged) and `harness optimise --gate evidence` (each
  evaluation a round of one budget). Property tests: null at -eps accepted <= 0.06 at alpha 0.1; best of 8 nulls on a fully
  reused set greedy 0.36, uncorrected 0.19, beta 0.06 0.03; FER over 12 rounds 0.12 (all null) and 0.05 (half null,
  0.66 of true gains accepted) at q 0.3; power 0.87 fresh and 0.76 fully reused for a +0.05 gain at 200 items.
- Retrospective on RSI3's dev counts (single-step counterfactual, finding 154): eps 0.02, beta 0.01, q 0.3 would accept
  7.9 of 12 steps in expectation, 0.13 of them exam non-improvements; the paper's eps 0.10 accepts all 12 (ties are
  noninferiority evidence at that tolerance).
- Open: X7 (scratchpad section 11: greedy vs significance-required vs evidence gate on one lineage recipe and budget, exam
  FER_0 and gain from bench's verdicts); a dev pool larger than one round's draw for the lineage (the current dev set is
  400 programs, so a draw of 150 reuses most items by round 3); the controller grades a whole draw at once (batched
  early stopping is in the lineage only) and its sign-test power plan still sizes arms under the evidence gate; the
  brain held-out loss of a lineage's training job reads the first 8 dev programs (reported, never selected on).

### brain

Rule for everything below: training is architecture-generic. SFT, policy gradient, self-distillation, full
fine-tune vs adapter, and anything beyond one model or one card live in shared paths an architecture gets by
implementing a contract; a missing capability is refused by name, never an architecture.


- [ ] Policy-gradient rollouts on qwen3 (in flight), then on any base the trainer supports.
- [ ] The 8B bf16 fine-tune ran out of device memory during held-out scoring on an otherwise empty 96 GB card
  (S2, 2026-11-07): reproduce and fix the scoring memory sizing.
- [ ] `ChatFineTune::run_with` should return a typed error, not unwind, on device OOM.
- [ ] LFM3-1.7B (`LiquidAI/LFM3-1.7B`, `-Base`): serving and full fine-tuning from the published files, so the
  arm-R and policy-gradient experiments can run on a small non-Qwen model with full training.
- [ ] Replica routing: drop a finished job of a long-lived open group from `running_jobs`.

### sven / harness

- [ ] Record executable failures, corrections and independent retests as structured experience (for the repair
  and diagnose views).
- [ ] Agent trials must not reach the benchmark's snapshot store (an agent unpacked the workspace snapshot bundles from `<benchmark-root>/snapshots`).

### Open questions to answer with numbers

- Does RL (with negative rewards) raise pass@64 on unseen families and compositions where self-imitation does not?
- Does a small model (LFM3-1.7B, full fine-tune) improve reliably across generations and discover answers its base
  never produced at pass@16 with thinking?

## Paper revision after the experience-to-competence work (2026-11-20)

Applies to "The Frontier Must Grow" (`papers/splinter-brain-sven-flywheel`). Findings from the agentic
experience-to-competence campaign (ledger `experience-to-competence.md`) bear on its framing and its limitations. The
headline (interpreter-labelled self-proposed SFT, 8.3% to 16.7% on the fresh exam) is not contradicted.

1. **Framing.** In the agentic work, competence came only from a source of new correct information (verified teacher
   solutions); the policy's own outcomes gave no gain. In this paper that source is the interpreter. The discussion
   should name the oracle as the information source, not self-generated knowledge.
2. **Negative appendix results.** "No gain on repository tasks / 27B agent" may share the agentic GRPO failure, where the
   update never moved the policy (KL about 0.002, ratio about 1.002, clip about 1%; splinter knowledge 184). Check those
   runs' KL and clip logs. Unless the policy moved, report them as under-optimised configurations, not as no gain.
   **Checked 2026-11-20 (lane L11, CPU): the policy did not move in any of them.** The 27B agent's candidates were RL
   (GRPO and GRPO with privileged self-distillation, k3 KL to the base at beta 0.02, clip 0.2, lr 2e-5, 4 records per
   step, each record drawn at most once). Medians over steps from their training logs (private HF dataset
   `qwen38-27b-flywheel`, `decisions/gen-{0,1}/generation-files.tar.zst`, `train-*.log`):
   gen 0 GRPO (60 steps, 240 of 476 records) KL 0.0044, clip 1.9%, |ratio - 1| 0.002, 4 of 60 steps with KL >= 0.01;
   gen 0 GRPO + self-distill (60 steps, 240 of 440) KL 0.0061, clip 2.5%, |ratio - 1| 0.007, 9 of 60;
   gen 1 frontier GRPO (120 steps, 480 of 2379) KL 0.0077, clip 2.6%, |ratio - 1| 0.012, 27 of 120;
   gen 1 frontier self-distill (120 steps, 480 of 2378) KL 0.0064, clip 2.3%, |ratio - 1| 0.009, 24 of 120.
   The steps above 0.01 are isolated spikes (to 0.34 and 31.9 on single steps), not a sustained move. The same holds for
   S2, the paper's Lua policy-gradient series on the 4B (`splinter-flywheel-evidence`, `rsi2/lineages/r*.log`, lr 6e-6,
   2 x 24 steps per lineage): KL median 0.0025-0.0034, clip 0.9%, |ratio - 1| 0.001-0.002. All five are in the regime of
   knowledge 184 (target KL 0.02-0.05, clip 10-20%), so the paper should report them as under-optimised RL
   configurations, not as "no gain" from the method. The repository-task results are SFT, not RL: the registered query's
   decisive-action adapter (seed 13, 120 steps, lr 5e-5) has no kept training log, and the 27B pilots' kept summaries
   (paper `data/measurements.json`) show the supervised update did move the model: CE8-T monitor loss 0.1738 against
   its base's 0.1799 (113 records, 57 steps), CE8-T-S 0.2315 against 0.2635, u1b 0.1676 against 0.1835 (1200 steps),
   while the tasks did not improve. There the question is transfer of a moved model, not a stalled optimiser.
3. **Small-cluster intervals.** The minimum-family rule (knowledge 177: a family-resampled interval has zero width when a
   few families agree) means intervals over 4 lineages are fragile. Re-check the round 2/3 "both intervals above zero"
   claims with an exact sign or permutation bound over lineages.
4. **Signatures.** Re-sign the verdicts with bench's Ed25519 key (bench 80ab3135), which closes the "symmetric signature
   protects against the learner, not the authors" limitation.
5. **Contamination.** The fresh exam shares templates with the old one (Jaccard 0.54). If GPU time allows, add an exam
   over lineages withheld from all proposal and training (bench tier c, dc82515c).
6. **Retention.** Add the retention anchor with a non-inferiority bound (bench retention-v1, the promotion gate).
7. **Served-prompt confound.** Brain builds before 1bdd5400c injected a "Reasoning effort" instruction into Qwen3
   prompts (brain knowledge 531). Within-build comparisons stand, but absolute accuracies may shift. Run a Tier-3 logit
   parity check on the paper's exam prompts before quoting absolute numbers across builds.
8. **Open directions now available.**
   - The learnability metric and difficulty dials (for RL from outcomes; failure selection is sound for oracle-labelled SFT).
   - Matched-step RFT controls.
   - Faster training (attention is 74% of a step; brain training-efficiency R6), which makes more lineages and
     compute-matched baselines affordable.
