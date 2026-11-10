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
| Arm HL (registered per-lineage replication, 2026-11-10) | Stopped by the project disk's inode quota in every replicate before generation 3; only r1 generation 1 measured (46 of 889, base 0, p 1.4e-14) | Criterion not evaluated; replicate state deleted; needs a fresh run with disk headroom |

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
- [ ] Dataset views beyond `sft-final`: `sft-repair` (wrong attempt + feedback -> fix), `sft-diagnose`,
  `sft-execute` (predict intermediate state), `sft-verify`, `policy-trajectory`; keep more than the shortest answer.
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
- [ ] A plateau controller that switches strategy (task distribution, external evidence, repair training,
  exploration breadth, objective) when the discovery rate per GPU-hour falls.
- [ ] Re-grade the paper's Lua claims (SL20-SL23, the 8B run) under a neutral protocol before keeping them.

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
