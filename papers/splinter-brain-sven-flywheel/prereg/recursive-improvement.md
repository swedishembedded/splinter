<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI1: recursive self-improvement, validated link by link

Written on 2026-11-07, before any run of RSI1 exists. Anything changed after the first run of a stage is a
dated amendment below the line "Amendments", and a stage's outcome is reported whatever it is.

## Why a new registration

The 27B generations of 2026-11-05..07 (section 08c) promoted two candidates, yet on one measurement on one
stack neither beats the base (gen-1 champion vs base 13 won / 16 lost; gen-2 vs base 17 / 14). Three causes
fit the records and are tested here instead of assumed:

1. **The policy hardly moved.** Generation 1 trained 120 steps x 4 records, LoRA rank 16, lr 2e-5, ending at
   KL ~0.007 to the reference: possibly too small an update to change agent behaviour.
2. **The exam could not see a gain.** The base solved ~90 % of the held-out tasks; only 20-30 of 206 paired
   tasks are discordant, so a 2-3 point gain is below the exam's power.
3. **Little signal per step.** 114 of 138 generation-0 training tasks were solved by every attempt (no group
   advantage), and the frontier pool got 3-4 attempts per task instead of 8.

The teacher-free 8B Lua loop, which had headroom, did improve (25 -> 49 of 82 sealed probes, p = 0.0065), so
the loop is not shown to be unable to learn; the 27B setup may have been unable to show it.

## Design, informed by current work on verifier-driven self-improvement

- **Expert iteration as the primary update** (STaR, ReST-EM, rejection-sampling fine-tuning): each
  generation fine-tunes on its own verified successes. It moves the policy far more per GPU-hour than a
  near-KL-bounded policy gradient and has the best record of multi-iteration gains with a fixed verifier.
- **Policy gradient done the way recent recipes do it** (DAPO, Dr. GRPO) as the second arm: dynamic
  sampling (only groups with mixed outcomes enter a batch, refilled until full), clip-higher (asymmetric
  clip, e.g. 0.2 / 0.28), token-level loss, no length normalisation, k = 8 per task, and a learning rate
  swept until the trained tasks actually move (stage 3).
- **A curriculum at the frontier the model sets itself** (Absolute Zero, Self-Challenging agents, R-Zero):
  splinter generates new mutation tasks from the repositories, admits only those the current policy
  solves sometimes (pass rate 0.2-0.8 at k = 8) and drills the functions failed tasks touch
  (`splinter tasks drills --for-tasks`). The pool is rebuilt every generation from the new policy.
- **An exam that can see a gain:** held-out tasks whose base pass rate lies in 0.2-0.8 at k = 8, sized by a
  power calculation for a 5-point paired gain (alpha 0.05, power 0.8), every model on one stack, the
  base re-measured with every generation.

## Stages (each must pass before the next is run)

| Stage | Question | Pass criterion | Cost |
|---|---|---|---|
| S1 | Is the trained adapter what is served? | Served log-probabilities of 50 training records equal the trainer's within 1e-2 mean abs; the candidate's trajectories differ from the champion's (tokens/turns/edits, paired test) | minutes, 1 card |
| S2 | Can training move the exam at all? | SFT on the exam tasks' own verified solutions (deliberate leakage, a control never reported as a gain) raises the exam pass rate by >= 20 points | ~1 h, 1 card |
| S3 | Does an update learn its own tasks? | Expert iteration and the DAPO-style arm, each on ~30 tasks with base pass 0.2-0.8, raise pass@1 on those same tasks by >= 10 points (paired, k = 8) | hours, 1-2 cards |
| S4 | Does it transfer? | Same update, held-out mid-difficulty exam: paired gain > 0 with one-sided p < 0.05 | hours, 2-4 cards |
| S5 | Is it recursive? | Three generations, each trained only on its own verified attempts and self-set frontier: every generation's champion beats the base on the same-stack exam (p < 0.05), and generation 3 beats generation 1 | 1-2 days |

Scale: S1-S4 first on Qwen3-4B (or 8B) on tasks whose base pass rate is checked to lie in 0.2-0.8, then the
27B on a mid-difficulty exam. A stage that fails stops the chain and its cause is fixed and recorded first.

## What is not claimed

Nothing here is claimed until a stage passes on its registered criterion. A failed stage is reported with
its numbers.

## Amendments

2026-11-07 18:08 (before any RSI1 run): **H-EFF, registered from an exploratory finding.** On the held-out exams
the base, gen001-grpo-sd and gen002-frontier-grpo sat on one stack, completion tokens per solved trial (paired
by task, tasks solved by both) fell base > gen1 > gen2: gen2 vs base x0.905 on single faults (59 fewer / 38
more, one-sided sign p = 0.021), x0.896 on three faults (61 / 22, p < 0.0001), x0.873 on firmware (12 / 3,
p = 0.018); evidence gen27b/efficiency-three-way.json in the flywheel evidence dataset. This was found after
the data and over several comparisons, so it is exploratory. Registered for every RSI1 generation from S4
on: the primary outcome is pass@1 (as above); the co-primary efficiency outcome is the geometric-mean ratio
of completion tokens per solved trial against the previous champion and against the base, paired by task,
one-sided sign test, Holm-corrected over the workloads. A generation counts as an improvement if it is not
worse on pass@1 (paired mean >= 0) and either outcome passes.

2026-11-07 21:20 (before S1/S2 run; lease n442, one GH200): **S1 and S2 as measured.** S1's log-probability
comparison needs a scorer the serving CLI does not expose; S1 is measured by its second half (the candidate's
trajectories differ from the champion's, paired) and by the served model id naming the trained adapter.
S2 trains on verified solutions of *training* tasks and examines those same training tasks (the exam seal
correctly refuses held-out tasks as training data); it is still a control that the trained update reaches
the served model and changes outcomes, never reported as a gain. Model for S0-S4: Qwen3-8B (brain
69b1834fa, sven 2ac91b02).

2026-11-07 23:30 (before any Lua run of RSI1; lease n442, four GH200): **S0 result and the closed-book arm.**
S0 measured Qwen3-8B (brain 69b1834fa, sven 2ac91b02) on the repository workloads at k = 4: repo-families
25 passed of 530 graded trials, repo-hard 0 of 479. No task lies in the 0.2-0.8 band at usable numbers, so
S3-S5 cannot run there for an 8B; S2 stopped on a device out-of-memory in training (a defect, not an
outcome). S3-S5 therefore run first in the closed-book arm where a small model is known to learn: the
teacher-free Lua 5.1 loop (`splinter learn --self-generate --reference-teacher --kinds predict-lua`,
structural skill graph, full fine-tune lr 1e-6 x 6 steps, three manual chapters 5.4-5.6, splinter b688e3ee7),
Qwen3-4B, four replicates (one per card). **Generations** are explicit: generation 1 is one learn run from the
base; generation k+1 is one learn run whose `--policy-base` is generation k's final checkpoint and whose skill
graph and probes are generation 1's (`--probes-from`). Each run measures the same sealed final probes before
and after. **Primary outcome:** final-probe passes of generation k against generation k-1 and against the
base, per replicate, combined over replicates (paired over skills: one-sided sign test; over replicates:
the mean gain and its one-sided t-test). **RSI is shown** if, over the replicates, generation 2 beats
generation 1 and generation 3 beats generation 2 on the final probes (p < 0.05 each), and the last
generation beats the base. Anything less is reported as measured.

2026-11-08 04:20 **Outcome of the closed-book arm (registered analysis, rsi1/lua/analysis.py).** Four replicates,
Qwen3-4B, final-probe passes summed per replicate. Generation 1 vs base +9, +7, +9, +10 (mean +8.75, one-sided
t p = 0.0004); generation 2 vs generation 1 +6, +5, +4, +4 (mean +4.75, p = 0.001); generation 3 vs generation 2
-2, +3, 0, -3 (mean -0.5, p = 0.63); generation 3 vs base +13, +15, +13, +11 (p = 0.0003). Two consecutive
generations improve significantly, the third does not: **the registered RSI criterion (generation 3 beats
generation 2) is not met.** Diagnosis from the run records (not a test): 1884 of 2862 self-proposed tasks were
refused (1101 repeats or duplicates, 516 failed checks), leaving 3-4 training records per round; skills at
ceiling (math.sqrt 14/14) and skills whose proposals collapse into repeats (math.frexp: 14 of 16 refused in a
round, 2/11 final probes throughout) stop moving. Any change made because of this is a new, separately
registered arm; generations 4-5 of this arm continue and are reported as descriptive only.

2026-11-08 04:45 (before any run of it) **Closed-book arm B: self-generation that does not starve.** Arm A's
generations 4-5 (descriptive) stayed flat (r1 40 -> 40 -> 40 of 57; r3 41 -> 43 of 68); arm A is stopped.
Arm B differs from arm A in the splinter build only: c7d11d932 (a round keeps asking past a request that
admitted nothing, up to 6 requests for 12 admitted tasks, and every request is shown what earlier rounds of the
skill admitted or refused as repeats; the probes are never shown), which also carries the other splinter
changes since b688e3ee7. Everything else is arm A's: Qwen3-4B, the same four replicates' skill graphs and sealed
probes (`--probes-from` each replicate's arm-A generation-1 run, so arm B's generation-1 baseline is the base on
the same probes), the same training recipe and budgets, explicit generations as in arm A. **Criterion (arm B
alone):** over the four replicates, generation 2 beats generation 1 and generation 3 beats generation 2 (mean
gain > 0, one-sided t-test p < 0.05 each), and generation 3 beats the base. Reported as measured either way;
arm A vs arm B per generation is reported descriptively (the builds differ in more than the fix).

2026-11-08 11:00 **Arm B outcome and a validity check of both arms (rsi1/lua/passk).** Arm B: generation 1 vs
base +11, +12, +13, +9; generation 2 vs 1 0, 0, +2, +2; generation 3 vs 2 +2, 0, -1 (r3 pending): two-generation
criterion not met, the plateau reached after one generation. **Validity check, not registered in advance and
reported because it changes how both arms must be read:** the sealed final probes were re-graded outside
splinter's probing protocol (a neutral request "answer with exactly what it prints, in one fenced code block",
the same exact-match grading for every model, brain 69b1834fa). Without thinking, greedy: replicate r0 base
43 of 75, its arm-B champion 45 of 75; r1 base 0.611, champion 0.542. Sampled (T 0.8, 16 per probe), no
thinking: pass@1 base 0.57 / 0.61 / 0.50 (r0/r1/r2), champions 0.60 / 0.53; with thinking: base pass@1 0.71 /
0.73 / 0.71 and pass@16 0.80 / 0.82 / 0.84, champions 0.73 / 0.74 and 0.85 / 0.86. Under splinter's own
protocol the same base scored about 40 % and the champions about 60 %. **So most of the gain both arms measured
is adaptation to the probing protocol, not added Lua competence, and it does not transfer to a neutral
request;** the base with thinking already answers about three probes in four. The plateau is where that
adaptation is complete. Consequences for the next registration: measure every model under the protocol it
is used with and under a neutral one, allow thinking, and draw the exam from what the base cannot do even at
pass@16 with thinking.

2026-11-08 11:10 (before any run of it) **Arm H: the loop measured on what the base cannot do, under a neutral
protocol.** Arm B's r3 generation 3 was stopped (arm B's criterion had already failed). *Exam:* 1484 Lua 5.1
standard-library programs written by code from 25 templates over randomised and edge-case arguments
(rsi1/hard/make_exam.py, seed 20261108); the reference is what the pinned interpreter prints (twice, equal); no
model writes or answers an exam item. Every item is asked of the base (Qwen3-4B, brain 69b1834fa) with the
neutral request "What does the following Lua code print? ... Answer with exactly what it prints, in one fenced
code block", no thinking, 4 samples at temperature 0.8; **the exam is the items the base never answered** (exact
match, whitespace-normalised). The exam is never shown to any run. *Loop:* arm B's recipe and build (splinter
b39782f43, which only adds relabel counting to c7d11d932), three replicates r0-r2 (their own skill graphs and
probes), explicit generations as before, each generation's champion kept until it is measured. *Measurement:*
the base (generation 0, re-measured on the exam to avoid regression to the mean from the selection) and every
generation's champion answer every exam item greedily under the same neutral protocol, no thinking; accuracy
per replicate. *Criterion:* pooled over the replicates' item pairs (an item that turns from wrong to right is a
win, right to wrong a loss), generation 1 beats generation 0, generation 2 beats generation 1 and generation 3
beats generation 2, each by a one-sided sign test p < 0.05, and no replicate's accuracy falls between
consecutive generations. Reported as measured either way.

2026-11-08 11:25 (before any arm-H measurement of a trained model) **Arm H grading.** Lua's `print` separates
values with tabs and a model writes spaces; exact matching per line would make the exam measure tab characters.
Arm H's grading compares answers token by token per line (whitespace runs collapsed, blank lines ignored),
for the selection of the exam and for every measurement. The exam is selected again under this grading.

2026-11-08 15:20 (before any run of it) **Arm R: self-improvement by the model's own verified reasoning on the
frontier.** Closes the gaps arms A, B and H left: tasks at the frontier, wide sampling, thinking on, training on
procedures (the model's reasoning, not only its final output), the interpreter as the only source of reward,
and a measurement that cannot be protocol adaptation. *Pools* (rsi1/rl; code-written by make_exam.py, seeds 31
and 32, never overlapping arm H's exam): train 1613 items of 19 families; test-seen 534 fresh items of the same
families; test-held 211 items of six families never trained on (math.frexp, string.gsub, table.remove, tonumber,
select/unpack, string.format q). *Iteration* (rsi1/rl/star.py): the policy (Qwen3-4B, then Qwen3-4B with the
latest adapter; brain 69b1834fa) answers 400 random training items 16 times with thinking at temperature 1.0;
an item answered right 1..15 times is on the frontier and up to two of its distinct verified traces (the model's
own reasoning and answer, right by the interpreter's output, token-wise per line) are kept; a LoRA adapter
(rank 32) is retrained from the base on every trace kept so far with brain's served trainer. *Measurement*,
the same for the base (iteration 0) and every iteration: test-seen and test-held, neutral request, thinking on,
4 samples at temperature 0.6 (pass@1 and pass@4), and arm H's exam greedily without thinking. *Criterion:* over
iterations 1..3, each iteration's pass@1 on test-held is above the previous one's (paired by item, one-sided
sign test on per-item pass counts, p < 0.05), and iteration 3 is above the base on test-held and on test-seen.
Reported as measured either way. A policy-gradient arm with negative rewards follows when brain trains rollouts
on qwen3 bases.
