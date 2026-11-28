<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Source to transfer: do trajectories on tasks authored from a manual teach skills that transfer

Scope: the controlled experiment, registered 2026-11-26 before any of its results existed, that tests whether tasks
Splinter authors from source knowledge alone, solved through sven, yield trajectories that teach the next model skills
it applies to unseen tasks, and which property of the training material carries the effect.

## Why this experiment, and why SFT first

The factory makes verifiable tasks; it has not shown that it makes transferable learning experiences. Earlier evidence:
the vim manual generation 1 (582 reviewed tasks, about 960 verified solutions) moved the 4B student from 0 to 4 of 100
exams solved within five attempts (p 0.0625, inconclusive); on-policy RL (R19) and rejection-sampling SFT on practice
tasks gave no held-out gain (`experience-to-competence.md`). Without a negative control, a gain could also be mere
exposure to the tool's format; without arms that differ only in the material, nothing says which property helps. The
experiment is SFT only: if correct, curated trajectories do not transfer under SFT, an RL result would not be
interpretable either.

## Continuous self-improvement: the operational definition this line of work is judged by (written 2026-11-27)

The loop improves itself continuously when, run as a recipe with no human-written training tasks, no human labels and
no human intervention between generations, it shows across at least 3 successive generations:

1. **Held-out gain.** Each generation beats its predecessor (generation 1: the base) on sealed exams that no
   generation trained on, under a registered paired test.
2. **No forgetting.** No generation regresses on the skills earlier generations had, measured by a registered
   forgetting gate on tasks of earlier skills.
3. **Robust and content-borne.** The gains hold across seeds and beat the shuffled-request control (same records and
   tool-use format, requests deranged).
4. **Restartable.** The loop restarts end to end from the recipe and the frozen binaries with no manual step.

Each part is a separate measurement; a part not measured is unmet, not assumed. Prior support for part 1 in another
family: RSI4 (2026-11-15, Lua output prediction, 4 of 4 lineages, 3 generations, each step held under the examiner's
sign test; `papers/splinter-brain-sven-flywheel`, S4), with no forgetting gate and no shuffled control.

## Material (one run, shared by every arm)

- Source: the Vim 9.1 user manual (`usr_*.txt` of the runtime the task image carries), chapter and section headings
  added by bench `environments/vim/manual_headings.py`: 222 sections, split to 263 items of at most about 1500 words.
- Tasks: bench `recipes/vim-manual/transfer.yaml` (the range pipeline: three material variants of one to six steps per
  section, Vim runs every reference, refusals repaired once, a reviewer checks each result against its request, fixes
  re-reviewed, the oracle solves every task kept). The author and teacher is Qwen3.8-27B-FP8, the student Qwen3-4B,
  thinking off, sven's narrate-before-each-call prompt, budget 60 turns and 16384 completion tokens.
- Attempts: the student 8 per task (temperature 0.7, top-p 0.8, top-k 20), the teacher 4.
- T1 hold-out: 10% of the authored tasks by task id (`tasks.pin` `holdout`, seed fixed in the arms recipe); no arm
  trains on any trial of them.

## Arms (from the same base Qwen3-4B, LoRA rank 16, alpha 32, lr 2e-4, 800 steps of 4 records, max block 16384,
## checkpoint chosen by validation loss every 40 steps, patience 4; seeds 1, 2, 3)

| Arm | Training records |
|---|---|
| A | none (the base model) |
| B | every verified successful trial of teacher and student, every action supervised (the existing pipeline) |
| C | B with at most one trial per task, the one with the fewest actions (diversity per token) |
| D | B with only productive actions supervised: an action whose tool calls all returned an error stays in context, unsupervised |
| F | negative control: B with each record's task request replaced by another task's (derangement); same tool-use format, no request-to-action mapping |
| G | B's selection rule on tasks authored backward (bench `recipes/vim-manual/backward.yaml`): the author proposes starting files and a desired multi-change result, the teacher proves the result reachable with the vim tool, the request is written last from the before/after pair; same optimizer budget |

## Measurements

- T2/T4 (unseen material, unseen structure, compositions of several ideas): the 100 held-out course exams
  (`tasks/vim` eval side, hand-written, never authored from the manual or read by any stage), 4 attempts per exam per
  candidate, same sampling and budget as practice, concurrency within the servers' slots.
- T1 (unseen instances of the authored distribution): the held-out authored tasks, 4 attempts each.
- Infrastructure failures are never a fail (bench's classification); a run with more than 2% infrastructure failures in
  any arm is repeated after its cause is fixed.

## Decision rule (fixed now)

- Unit: an exam task. For a pair of arms X, Y: d_t = (passes of X over its 3 seeds x 4 attempts) - (passes of Y over
  12 attempts; for A, 12 attempts of the base). One-sided exact sign test over tasks with d_t != 0.
- Primary family (Holm, alpha 0.05): B > A, D > A, C > A, G > A on the exams.
- The goal's claim ("trajectories teach transferable skills") holds when at least one of B, C, D beats A in the
  primary family AND the same arm beats F on the exams (one-sided sign test, alpha 0.05). Beating A alone could be
  format exposure; beating F says the content of the trajectories matters.
- Secondary, reported with task-bootstrap 95% intervals and never as claims when not significant: G vs B, D vs B, C vs B,
  every arm on T1, per-seed rates, failure classes.
- No exam result selects a checkpoint, a recipe, a seed or a data mix; the exam trials are never trained on.

## Deviations

| Date | Change | Why |
|---|---|---|
| 2026-11-26 | arm G (backward-authored material) added, primary family grows to four comparisons | the user proposed outcome-first authoring while the material run was in progress; added before any arm trained |
| 2026-11-26 | the source is Vim 9.1's own `usr_*.txt` (the text of the manual), not an OCR of its PDF | the OCR'd copy was lost with a node; the text form is what the task image's Vim ships |
| 2026-11-26 | the material run was forked once (`material2`) after a node ran out of memory and the serving fleet hung; its committed stages (authoring, review, tasks, oracle) were reused, its student and teacher attempts collected again from the start | the interrupted attempts could not be resumed by the build that fixed the defects; no attempt of the first run is used |
| 2026-11-26 | servers run `--speculative off` | 7 of 16 servers hung on a device kernel under `auto`; speculative decoding leaves the sampled distribution unchanged |
| 2026-11-26 | each arm-seed trains on one card and evaluates through its node's four cards, 64 attempts at once | a router does not route training jobs; 64 is a card's slots |
| 2026-11-27 | the operational definition of continuous self-improvement (above) written after generation 1's exam results existed and before any verdict against it | the user asked for the claim to be scoped by a definition fixed before judging it; it adds no comparison to the registered family |
| 2026-11-27 02:15 | primary estimand from here on: exam passes per generation of an autonomous loop, per model size (Qwen3-4B, Qwen3-8B; Qwen3.8-27B only if its generations can finish before the lease ends). The loop is outcome-first authoring: each generation's tasks are authored by the 27B from the manual under that generation's fixed rule, attempted by the current champion and the teachers (27B; Inkling 2 Mini from generation 2, every record labelled by model), the verified records train the next candidate from the base with earlier generations' records as replay, and the champion is promoted only on held-out authored tasks and the previous generation's held-out tasks (the forgetting gate). Exams never select, stop, filter or author anything. Test: per size, each generation's champion against the base and against the previous generation's champion on the 100 exams, 4 attempts, exact one-sided sign test over exams (Holm over the generations of a size), plus a one-sided trend test of exam pass rate over generations (Cochran-Armitage on attempts); retention: each generation's candidate against the previous champion on the previous generation's held-out tasks and against the base on the course's hand-written practice tasks. Generation 1 of the 4B is the arm-G seed-3 adapter (lowest validation loss of the three seeds). Registered before any generation-2 exam result was produced. | the user set the iterated loop and model size as the goal; the student-only arm had shown that generation 1's exam gain is the teacher's trajectories' (0 of 1200 exam attempts), so a size large enough to learn from its own and the teachers' attempts is a factor to test |
| 2026-11-27 03:00 | Qwen3-8B generation 1: 581 of 3736 student attempt slots ended as model errors (not graded, not counted) when the four servers of the node were stopped with the command that had started them; the attempts were not repeated, so the 8B's own records come from 3155 counted attempts (81 passed) | the servers were restarted at once; repeating the stage would have changed its identity and cost about an hour of the lease; the 8B's own records are a small share of its dataset beside the teacher's |
| 2026-11-27 04:45 | Inkling 2 Mini removed as a second teacher from generations 2 and 3 of the loop; the 27B stays the only teacher there. Generation 2 of the 4B continues as run `loop-g2n`, a fork of `loop-g2` that reuses every stage up to the 27B teacher attempts; only its dataset (champion and 27B records) and later stages differ | at the 48 concurrent sequences its three cards hold, the Inkling stage of generation 2 alone would have taken about three hours (2292 attempts) and pushed the third generation past the lease; Inkling's contribution is reported from the forward teacher round (8 of 80 tasks the 27B never solved) |
| 2026-11-27 04:50 | Qwen3-8B generation 1 (seed 3) exam evaluation: 19 of 400 base-arm slots were never counted (4.75%, above the 2% bound); the stage was not repeated | the lease does not leave time to repeat it before the 8B generation 2; the candidate arm counted all 400; the unpaired slots are reported with the result |
| 2026-11-27 05:40 | the loop's releases (promotion on held-out authored tasks) judge an evaluation with at most 2% uncounted slots on the slots both arms counted (`promotion` `max_unpaired_share: 0.02`, splinter 0a7434053), matching the registered 2% infrastructure bound; the Qwen3-8B generation-1 release is recomputed with it (held-out tasks 50/204 against the base's 1/200, 4 base slots uncounted, 1.96%) | under the sign test's strict rule a single uncounted slot left the release inconclusive and kept the base 8B as champion, so generation 2 would have attempted its tasks with the base model; the exams are not involved |
| 2026-11-27 06:35 | brain's router: a fully idle upstream takes a model once the resident upstreams hold at least max(slots/8, 8) of its requests (brain 2104ee58f), deployed on two nodes at 06:35 and 06:58 | it had kept every request for a LoRA model on the one card where the base was resident, so evaluation and champion-attempt stages ran on one card of up to ten; throughput only, not what is measured |
| 2026-11-27 | the harbor worker left the sandbox of a trial finished just before a kill and the stage publish failed on it; fixed in splinter 2fd05cc42, the one affected run cleaned up by hand | a manual step, recorded against part 4 of the definition |
| 2026-11-27 07:00 | the 4B generation-3 recipe keeps generation 2's stages on their own champion (role champion-g2), so they are reused, not recomputed | generation 2 was not promoted; its tasks and attempts stay valid inputs |
| 2026-11-27 | 27B teacher rounds on the tasks the teacher never solved (generation 2: 79 tasks, generation 3: 78; 8 more attempts each, no Inkling) | for the dataset only; not part of any arm or generation |
| 2026-11-27 09:06 | brain's router balance and spill no longer read the servers' nominal decode slots (3466): a cached prefix or session yields to an idle card at 8 requests or a quarter of the lighter load (brain 685c185d1); spill at an absolute 8 requests (brain f0423caf8), deployed on n457's router at 09:06 (12 cards) and on n466's after the 4B generation-3 release | four cards had carried all load while eight idled; throughput only, nothing measured changes |
| 2026-11-27 | the 4B generation-3 champion-attempt stage restarted with concurrency 1024 instead of 384 (run loop-g3b instead of loop-g3, about 5 minutes of attempts discarded); the 8B generation 3 also runs at 1024 | concurrency is not part of a stage's identity; every other stage was reused |
| 2026-11-27 09:15 | Defect fixed in the training recipe: a fixed cap of 800 LoRA steps while each generation trains a fresh LoRA from the base on a growing dataset (new records plus all earlier generations' as replay), so every later generation saw its data fewer times. Fix, as a recipe rule (generator gen-scaled.py): epochs held constant, steps = ceil(E x N / grad_accum), E = generation 1's effective epochs at its selected step (4B 1.891, 8B 1.92), N = the generation's training records including replay; everything else unchanged (rank 16, alpha 32, lr 2e-4, grad_accum 4, max_block 16384, validation-loss early stopping with patience 4 every 40 steps); exams play no part. With the rule in the recipe the loop runs autonomously from here. 4B generation 3: the fixed-cap candidate (run loop-g3b) had trained and its exam evaluation had been computed but not inspected when the fix was made; the candidate under the rule (run loop-g3s, fork of loop-g3b, 2385 steps) trains on the same dataset; both are reported, promotion by the registered rule (held-out tasks only). 8B generation 3 trains under the rule (loop-g3-8b-scaled) beside the fixed-cap run (loop-g3-8b). 8B generation 2 is left as trained. | Measured epochs at the selected step (one step = 4 records): 4B generation 1 1100 records, step 520, 1.89 epochs (stopped early on a validation plateau); generation 2 3134 records, step 760 of 800, 0.97 (at the cap, still improving); generation 3 at the cap 5045 records, 0.63. 8B generation 1 1082 records, step 520, 1.92; generation 2 3299 records, step 800, 0.97 (knowledge 235). |

## Status

- 2026-11-26 14:50: material run started on 16 GH200 (4 nodes, one brain per card behind one router).
- Arms options (curation, per-task limit, shuffled control, holdout, composition report) being built in splinter; the backward pipeline in bench.
- Calibration (selects nothing): the base 4B passed 0 of 1012 counted exam attempts (12 per exam, partial run); the
  27B teacher solved 94 of 100 exams within 4 attempts, pass@1 0.63.
- Material (`material2`): 1030 forward tasks (66% need one or two commands); student 8204 counted attempts, pass rate
  0.100, 231 tasks solved; teacher 4095, 0.811, 944 solved. Arm B's dataset: 3739 verified trials of 857 tasks (the
  T1 hold-out excluded), 3411 distinct action sequences.
- Backward material (`backward2`): 705 outcome-first proposals, 562 reached by the teacher, 467 final tasks; student
  3688 attempts, pass rate 0.031, 36 solved; teacher 1868, 0.722, 410 solved: harder than the forward tasks and
  nearer the exams (teacher 0.63, student 0.00).
- 2026-11-26 21:00: all 15 arm-seeds (B, C, D, F, G x 3) training, one card each; 2026-11-27 02:30 exam results below.
- Defects found and fixed on the way (each with its own commit and knowledge entry): sandbox images held in tmpfs
  exhausted a node's memory (splinter 851216e17, k225); the worker's reconcile failed resumes (5fb824d6b); on-disk
  sandbox copies took 436 s to start (shared lower tree plus overlay, 9841886df, k227); attempts lost to interruptions
  spent retries (cc5581333); the gateway's connection pool, Harbor's job lock and a per-command lookup took about 50
  of 70 s per trial (9415eb39b, 2b62ff444, de1149b0a, 418046bd7, k228); brain's router stranded requests behind hung
  upstreams (ae72fc212), packed one model on few cards (b726c045b), opened the GPUs (da6ce90cc); Qwen3 plans withdrew
  a model instead of fitting the free memory (f74314255); a CUDA lane can hang with /health reporting ok (brain k602,
  watchdog open).

## Result on the exams (2026-11-27, registered analysis)

All 15 arm-seeds trained (800 steps, held-out validation loss 2.49 to 0.60 for B seed 1) and were measured on the 100
held-out course exams, 4 attempts per exam, each against a base run in the same evaluation stage. Infrastructure:
at most 11 of 800 attempts of an evaluation stage (1.4%), within the 2% bound in every stage.

| Arm | Exam passes (of 1200 attempts, 3 seeds) | Exams solved | Base passes (paired, about 1180 attempts) |
|---|---|---|---|
| B every verified success | 22 (6, 6, 10) | 10 | 0 |
| C one trial per task | 10 (5, 2, 3) | 6 | 0 |
| D productive actions only | 22 (10, 4, 8) | 11 | 0 |
| F shuffled-request control | 0 (0, 0, 0) | 0 | 0 |
| G backward (outcome-first) tasks | 42 (14, 11, 17) | 25 | 0 |

Primary family (each arm against the base, per exam pooled over seeds, one-sided exact sign test, Holm at 0.05):
B 10 wins / 0 losses, p 0.00098; C 6 / 0, p 0.0156; D 11 / 0, p 0.00049; G 24 / 0, p 6e-8: all significant. Each
against the negative control F: B 10 / 0 (p 0.00098), C 6 / 0 (0.0156), D 11 / 0 (0.00049), G 25 / 0 (3e-8): all
significant. **The registered claim holds: trajectories on tasks authored from the manual alone teach the 4B student
skills it applies to exams it never saw, and the gain is the trajectories' content, not exposure to the tool's
format (F stays at the base's 0).**

Secondary (two-sided, intervals by task bootstrap, not claims of the registered family): G against B 21 / 5,
p 0.0025, pass-rate difference +0.017 [+0.003, +0.031], although G trained on a third of B's records (1100 train
records of 348 tasks against 3359 of 857; recomputed from the arm datasets' `composition.json`, correcting the 1,059
and 3,002 first written here). Not isolated: G kept step 520 (about 1.9 passes over its records) where B ran all 800
(about 0.95), and G's tasks are harder (knowledge 230). D against B 7 / 5, p 0.77, no difference measured. C against
B 2 / 8, p 0.11, -0.010 [-0.019, -0.002]: one solution per task taught less than all verified solutions, on fewer
records (771). Absolute level stays low (G 3.5% of attempts, 25 of 100 exams solved at least once; the teacher's pass@1
is 0.63): the exams combine several ideas on unseen material and the base 4B solves none of them.

Teacher and student records (train split, by each record's `model`): B, D, F 2699 teacher / 660 student; C 629 / 142;
G 1014 / 86. The arms do not separate the teacher's contribution; a student-only arm S (B's rule on the 4B's own
verified successes, 3 seeds) was started on 2026-11-27 to do so.

T1 (held-out authored tasks, secondary; `decide.py results eval-unseen`, all arms on three seeds): on the 94 held-out
forward tasks B 705 / 1128 (0.625) against its base's 126 / 1128, 82 / 1, p 8.7e-24, +0.513 [+0.447, +0.582]; C 642 /
1128 against 113 / 1126, 79 / 3, p 1.9e-20, +0.470 [+0.397, +0.542]; D 723 / 1128 against 119 / 1128, 84 / 1, p 2.2e-24,
+0.535 [+0.463, +0.605]; G on its 51 held-out backward tasks 127 / 612 against 22 / 612, 25 / 0, p 3.0e-8, +0.172
[+0.111, +0.237]; all hold under Holm and B, C, D each beat F (86 / 0, 84 / 0, 87 / 0). F 9 / 1128 against its base's
133 / 1128: 2 tasks won, 26 lost (exploratory, two-sided p 3.0e-6, -0.110 [-0.163, -0.064]): training on actions paired
with other tasks' requests made the model worse than the base at following a request. Secondary on T1: D against B 37 /
24, two-sided p 0.12; C against B 24 / 42, p 0.036, -0.056 [-0.098, -0.016].

Recomputed 2026-11-27 from the per-task outcomes with `decide.py results eval-exam` and `secondary.py results
eval-exam`: every number above matches.

## Verdict against the operational definition (2026-11-27, generation 1)

1. Held-out gain: holds for generation 1 against the base; the three-generation requirement is unmet (one generation).
2. No forgetting: unmet, not measured (no forgetting gate was run).
3. Robust and content-borne: holds for generation 1 (every seed of B, C, D, G passes exams; every arm beats F).
4. Restartable: unmet; the material run was forked by hand after a node ran out of memory and the fleet hung, and
   pipeline defects were fixed during the run.

Claim as supported: one step of the loop in a second task family, not continuous self-improvement.

## Evidence and paper

- Paper: `papers/generalized-continuous-learning` ("Continuous Learning from Executable Environments"), where the Vim
  steps (authoring from the manual, oracle admission, outcome-first authoring, trajectories into adapters) follow the
  Lua pool in one method; registrations, arm letters, deviations and the comparisons without a gain are in its appendix.
- Private HF dataset `swedishembedded/source-to-transfer-evidence` (commit ebbe2fcf): the rerun's evaluation records (`loop/rerun-8b.json`, `loop/rerun-4b.json`), figure previews as PDF, PNG and WebP, arm datasets with lineage, arm
  recipes, locks and frozen revisions, training reports, exam aggregates, the decision and secondary outputs, the
  authoring funnel, the material runs' manifests, the analysis code (decide.py, secondary.py, export-results.py,
  analyse.py, aggregates.py, funnel.py, gen-arms.py).
- Private HF model `swedishembedded/vim-manual-transfer-adapters` (commit 0485d8a1): the 15 adapters with their
  candidate and training records.
- Tasks and trajectories: private `swedishembedded/vim-headless-mastery-manual`, `vim-headless-mastery-sft-manual`.
- Per-task exam outcomes (item level) stay in `~/project/shared/bench-private/evidence/source-to-transfer-2026-11-27/`.

Not claimed: anything about other tools or domains (T3), retention on non-Vim tasks (no anchor was run), or stability
under another authoring seed of the material.

Student-only arm S (2026-11-27, decided outside the registered family, `results/student-only/`): B's selection on the
4B's own verified successes (660 train records, 200 tasks), 3 seeds: 0 of 1166 counted exam attempts, paired base 0 of
1184; generation 1's exam gain comes from the teacher's trajectories. Two of its evaluation stages exceeded the 2%
infrastructure bound (21 and 19 of 800) and were not repeated; its held-out evaluation was not run. Loop estimand
(`loop.py`, `results/loop.json`): Qwen3-4B generation 1 = arm G seed 3, 17 / 400 exam attempts against its base's 0 / 393,
12 / 0 exams won/lost, p 0.00024.

Retention (2026-11-27, exploratory; `retention.py`, `results/retention/`, bench bcbc6dd4 recipes): generation-1-era
candidates against the base on the course's 228 hand-written practice tasks (168 exercises, 60 composites, never trained
on), 4 attempts each: G seed 3 (4B generation 1) 378 / 912 against 138 / 912, 120 tasks won / 16 lost, one-sided p 3.5e-21,
+0.263 [+0.213, +0.315]; D seed 1 360 / 912 against 141 / 910, 109 / 13; C seed 1 336 / 912 against 142 / 912, 103 / 17.
No net regression; nearly all the gain is on the exercises (composites stay near zero for all).

Qwen3-8B generation 1 (2026-11-27, `loop.py`): seed 3, 22 / 400 exam attempts against its paired base's 0 / 381, 16
exams won / 0 lost (task level), one-sided p 1.5e-5; base calibrations 4B 0 / 1156, 8B 0 / 1118. `export-results.py`
now skips evaluations a fork inherited (the candidate must be the run's own adapter) and `analyse.product_attempts` takes a
fork's own (newest) product: before, a fork's export could read its parent's evaluation.

Qwen3-8B generation 1 over three training seeds (`loop.py`, replicates `loop-g1-8b-seed{1,2}` converted by
`attempts-to-results.py`): seed 1 28 / 400 against 0 / 379, 20 / 0 exams, p 9.5e-7, +0.071 [+0.042, +0.104]; seed 2 24 / 400
against 0 / 385, 16 / 0, p 1.5e-5; held-out authored tasks 53 / 204 and 54 / 204 against 0 / 203 and 0 / 200. Not promoted
(release inconclusive); base arms 21 and 15 of 400 slots uncounted (server rejections, cause not determinable). The 8B curve
continues on seed 3.

Qwen3-4B generation 2 (2026-11-27, `loop.py`, run `loop-g2n`, candidate against the generation-1 champion in the same
stages, every slot counted): exams 13 / 400 against 11 / 400, 8 / 5 exams won/lost, one-sided p 0.29, +0.005 [-0.018,
+0.028]; its held-out tasks 66 / 264 against 57 / 264, 16 / 11, p 0.22; forgetting gate on generation 1's held-out tasks
49 / 204 against 39 / 204, 13 / 8, p 0.19. Not promoted (inconclusive): generation 2 did not beat generation 1 at 4B. By the
loop's rule (the champion is the latest promoted release) generation 3 started at 07:00 from the generation-1 champion (run
`loop-g3`, 528 new tasks, records of generations 1 to 3, released against generation 1); no human step chose anything.
Qwen3-8B retention, seed 1, on the 228 course practice tasks: 463 / 912 against its base's 139 / 893, 129 / 14 tasks,
p 8.9e-25, +0.361 [+0.305, +0.415]; base 19 slots uncounted (2.1%).

Qwen3-8B generation 2 (run `loop-g2-8b2`, against the 8B generation-1 champion seed 3, every slot counted): exams 30 / 400
against 23 / 400, 12 / 7, one-sided p 0.18, +0.018 [-0.018, +0.052]; its held-out tasks 81 / 264 against 78 / 264, 15 / 10,
p 0.21; forgetting gate 60 / 204 against 46 / 204, 16 / 10, p 0.16. Not promoted. Both sizes: generation 1 a large significant
gain over the base, generation 2 a small non-significant gain over generation 1 with nothing lost; by the registered
criterion (each generation beats its predecessor, Holm over a size's generations) continuity fails at generation 2 at both
sizes, pending generations 3 and 4. 8B generation 3 started at 09:08 from the generation-1 champion (`loop-g3-8b`, sharing
the 528 generation-3 tasks and teacher attempts with the 4B); 4B generation 4 authoring (propose-outcome-g4.md) runs.
Qwen3-8B retention, seeds 2 and 3: 461 / 912 against 138 / 903, 138 / 10, p 3.1e-30, +0.358 [+0.305, +0.410]; 408 / 912
against 134 / 897, 127 / 14, p 2.9e-24, +0.303 [+0.254, +0.354]. All three 8B seeds retain.

Generation 2, analysis of what limited it (2026-11-27, both sizes): training budget, the 800-step cap gave generation 2 0.97
epochs where generation 1 had 1.89 (4B) and 1.92 (8B), fixed from generation 3 (knowledge 235); training material, 40-45% of
generation 2's new records are the champion's own (6-8% at generation 1) and 64% of those are on tasks it already passes at
least 5 of 8 times, replay is a plain union and the teacher's gradient share rose only 11-13%; tasks, generation 2's are as
large and as hard as generation 1's (workspace 11.9 against 13.0 lines, teacher pass rate 0.72 against 0.68; 15.7% meet the
prompt's 15-line specification); exam resolution, 13-19 discordant exams at 4 attempts, power 0.12-0.44 for a 1.25-1.5x rate
improvement (0.40-0.94 with 40 attempts); pooled over sizes exams 20 / 12 (p 0.11), held-out 31 / 21
(p 0.11), forgetting gate 29 / 18 (p 0.07). Exam failure modes: done
declared with wrong content (65-78% of failures), identical-call loops (62-66% of failing trials), Ex errors (mostly E486),
single-request stalls at 480 s (5-8%). Excluded: prompt or mode mismatch, truncation, duplicates, context exhaustion, budget
differences, forgetting, leakage, uncounted slots.

Second analysis of generation 2 (2026-11-27; scripts in the n457 run archive of 2026-11-27, critic scripts and tmp/rca).
Corrections: the learning rate follows a cosine to a tenth of its peak after a warmup of steps / 20 (brain
crates/sdk/src/finetune.rs), not to zero. Pooled over both measurements of the 4B generation-1 champion, 28 / 800 (3.5%)
against generation 2's 13 / 400 (3.25%): no exam gain at 4B. 8B generation 1 over three seeds and four measurements 97 /
1600 (6.1%) against generation 2's 30 / 400 (7.5%). On generation 1's held-out tasks generation 2 gains about 1.18x at both
sizes (4B 0.240 against 0.203 pooled, 8B 0.294 against 0.249). Training-seed variance is within binomial noise (8B exams 22,
24, 28). The champion's share of the 8B generation-2 new records is 45% by record, 42% by supervised characters; teacher
exposure by characters rose about 12%. Findings: (1) generation 2 made the 8B commit sooner, not edit better: passes / trials
by tool calls per trial, pooled over exam and both held-out sets, champion against generation 2: at most 2 calls 47 / 116
against 75 / 189, 3-6 64 / 284 against 79 / 350, 7-15 31 / 280 against 17 / 170, over 15 5 / 188 against 0 / 159; precision
in the shortest bucket unchanged (0.41 against 0.40); median completion tokens per exam trial about 1530 to about 1000; 34% of
generation-2 champion records are exactly two calls and 48% contain no error (teacher 8% and 13%). (2) Repeated identical
calls are 42% of the champion's exam calls and 62% of generation 2's; 5.3% of supervised champion calls carry sven's repeat
note (teacher 0.6%). (3) The release gate as run: eval-forget had no purpose and defaulted to target, so promotion needed two
superiority tests; the probability of promoting a true 1.25x step is about 0.19 at 4 attempts per task, about 0.85 at 16; the
8B generation 2 was held back with its forgetting evaluation at +0.069 and its own held-out test inconclusive (knowledge
236). (4) The 15-to-80-line specification was missed because the prompt's worked example has files of 12 and 5 lines; output
was not length-limited (median 1.7k of 16k tokens) (knowledge 238). (5) Validation uses only the new generation's records;
generation-2 validation is 41% champion characters (generation 1: 3%), so selection rewards imitating the champion
(knowledge 237). (6) Scaled steps restore generation 1's exposure in records, but generation 1 stopped at step 520 of an
800-step cosine (rate about 35% of peak), so a full cosine over 1.89 epochs carries about 26% less cumulative learning rate
per record; patience stays 160 steps, 0.58 epoch at generation 1 and 0.13 at generation 3. (7) Trials ended by wall clock or
budget are 6% of trials, 27-31% of trial time, and pass none; sven's max_tool_rounds defaults to 200, beyond the registered
60 turns. (8) The course practice tier (228 tasks, base rate 0.47) has power 0.98 for a 1.15x effect at 4 attempts, the most
sensitive measurement available.


Open: generations 3 and 4 (and the trend test once a size has two promoted generations); an unattended restart from the recipe.

## Plan set by the user (2026-11-27 10:00)

1. Generation 3 at both sizes under the epoch rule (4B loop-g3s5: 2385 steps on 5045 records; 8B loop-g3-8b-s: steps from its
   dataset at E 1.92), exam, held-out and forgetting evaluations, compared with generation 1 on the exams.
2. If generation 3 succeeds: generation 2 retrained on its own records under the same rule (4B loop-g2-s: 1482 steps on 3134
   records; 8B loop-g2-8b-s: 1586 steps on 3299 records; started at 10:21 on spare cards, used only if step 1 succeeds).
3. Then generation 4. If generation 2 could not be retrained on its own data, generation 4 follows generation 3 directly.

Restart record: the lease restarted at 09:35 (job 5831740, n541/n545/n547/n550, ends 2026-11-28 09:34). Every node's scratch
was archived first (~/project/shared/runs-archive/2026-11-27/, one zstd tar per node, listings verified) and restored. Every
committed stage is reused once the task repository's operator code matches the old runs byte for byte (bench 83e1b4f8:
operators pin benchlib, environments, sandbox and pyproject.toml; HEAD's environments changed every operator digest).

## Loop v2 for generations 2 and 3 (2026-11-27 14:20, recorded before any v2 result)

The user's goal changed at 14:10: no generation 4; show that generations 1, 2 and 3 each improve, every exam at the
registered 4 attempts per task (the 16-attempt re-measurements were stopped unfinished and are not reported). Findings that
determined the change: generation 3 under the epoch rule did not gain on the exam at 4B (13 / 381 and, fixed cap, 16 / 400
against the generation-1 champion's 11 / 400), and generation 2 retrained at 1.89 epochs did not either (6 / 399, not
promoted), so the training budget alone is not what stalls the loop. Loop v2 changes what the two analyses measured as the
causes, generator `gen-v2.py`:

1. The new generation's training dataset is the teacher's verified records only, so validation is teacher-only (knowledge
   237); the champion's records enter as replay at one per task, drawn at random (45% of generation 2's new records were the
   champion's own, mostly on tasks it already mastered).
2. The LoRA continues the champion's adapter (`start: champion`) instead of training from the base, at half the learning rate
   (1e-4), with replay at half the draws (`replay_share: 0.5`); steps = ceil(E * N_teacher / (grad_accum * 0.5)), E the
   size's generation-1 epochs (4B 1.891: 1226 records, 1160 steps; 8B 1.92: 1226 records, 1179 steps).
3. The forgetting evaluation is a retention test and the release requires target and retention, with the registered 2% bound
   on uncounted slots (knowledge 236: two superiority tests promoted a real 1.25-fold step about one time in five).

Exams, held-out tasks and the champion are as registered; promotion is decided on held-out tasks only; exams are measured.
Runs: 4B loop-g2-v2 (n547), 8B loop-g2-8b-v2 (n541), started 14:15-14:20; generation 3 v2 follows from generation 2 v2's
release under the same rules.

## Loop v2, generation 3: failed actions unsupervised (2026-11-27 14:45, recorded before any generation-3 v2 result)

Generation 3 v2's two new datasets (teacher-only, and the champion's one per task) project with `supervise: productive`: an
action whose tool calls all failed stays in the record as context, unsupervised; the action that recovers and the final answer
stay supervised. Generation 2 v2 was already training with every action supervised and stays as recorded. Why, measured on the
8B datasets and exam trajectories (`building-blocks.py`): 70-80% of training records contain a tool error and recover from it,
so recovery is in the data, but with every action supervised the failed attempts are trained too (19-30 records teach the
broken `:put text` form before its fix); after a first error the student re-reads the file in 9-10% of exam trajectories
against the teacher's 27%. The tool keeping a failed command's edits is Vim's own behaviour, which the agent must learn to
recover from: the training target is the recovery, not the failure.

## Rerun of generations 1-3 on the fixed loop: registration (2026-11-27 15:10, before any result of the rerun)

All earlier runs are superseded (old Vim tool, old recipes) and not comparable with these. Fixed in place before the rerun:
bench b147c21b (vim tool `diff` and `reset`: the agent's version control), 26b12413 and d971e5df (loop recipes: teacher-only
new data, the champion's records one per task in replay, `supervise: productive`, continuing the champion, `params.epochs:
1.9`, a retention gate with the 2% uncounted bound on every release, per-generation champion roles, generation 3 also checks
generation 1's held-out tasks, sven capped at the registered 60 tool rounds); splinter b0d85c4d6 (concurrent stages),
88f8333f0 (`params.epochs`), and fixes in flight for SFT continuing the champion, a champion resolving a run's newest
promoted release, a retention gate that is inconclusive without pairs, and item retries; brain eb2164937 on every card
(KV-pool sizing and back-off recovery, adaptive residency). One authoring run serves both sizes (Qwen3-4B, Qwen3-8B).

Measurement, every exam at the registered 4 attempts per task (no other exam sensor gates or claims):
1. Primary, per generation step and size: the paired task-level sign test of the two arms of the same `eval-exam-gK` stage
   (candidate against its champion, measured together).
2. Secondary, named now: the step test pooled over sizes (exams won and lost summed); generation 3 against generation 1;
   a trend over generations at exam level (a permutation over exams); generation 1 against the base.
3. Promotion is decided on held-out authored tasks only (target) and retention (previous generations' held-out tasks); the
   exam never selects. A generation not promoted is reported as measured; the next continues the latest promoted champion,
   which is then its predecessor.
4. Evaluations with more than 2% uncounted slots are reported and excluded from claims.
5. Known limits, stated now: one training seed per generation; no shuffled-request control under the new tool; a baseline
   arm reused by identity when the champion is unchanged (not re-measured under the same load).

## Rerun on the fixed loop: 8B results (2026-11-27 17:16-23:58)

Runs s8b-g1..g3 on n547 `/scratch/local/work/ws/runs` (splinter 578cfec9b, bench d971e5df), all three promoted, unattended.
Paired, 4 attempts per task, one-sided sign test:

| step | own held-out (target) | earlier held-out (retention) | exam (primary) |
|---|---|---|---|
| g1 vs base | 85/224 vs 9 (79/3, p 1.9e-20) | - | 25/400 vs 0/391 (25/0, p 3e-8), 9 base slots uncounted (2.25%): excluded by rule 4 |
| g2 vs g1 | 55/188 vs 40 (29/14, p 0.016) | g1's: 92/224 vs 86 (29/23) | 30/400 vs 30 (17/17, p 0.57) |
| g3 vs g2 | 59/196 vs 45 (25/11, p 0.014) | g2's: 49/188 vs 54 (14/19); g1's: 109/224 vs 105 (18/14) | 32/400 vs 32 (18/18, p 0.57) |

Verdict: each generation improves on its own held-out authored tasks without forgetting; no claimable exam step (primary
tests null; g1-vs-base excluded). The 9 uncounted slots were base tool-call loops misclassified as endpoint refusals
(HTTP 499), fixed in 27372f64b after the runs. 4B resumed at 23:59 after the 8B finished, per the user. Two seed replicates
of 8B g1 were stopped unused (one named g1's own seed 3 by mistake, so its adapter was rightly reused by identity; no defect).

## Rerun on the fixed loop: 4B results (2026-11-27 23:59 - 2026-11-28 04:57)

| step | own held-out (target) | earlier held-out (retention) | exam (primary) | release |
|---|---|---|---|---|
| g1 vs base | 64/224 vs 6/222 (62/4, p 1e-14) | - | 11/400 vs 0/397 (11/0, p 0.00049; 3 uncounted, within 2%) | promoted |
| g2 vs g1 | 35/188 vs 31 (17/13, p 0.29) | g1's: 76/224 vs 68 (27/19, p 0.15) | 11/400 vs 12 (10/11) | not promoted |
| g3 vs g1 | 41/196 vs 27 (24/10, p 0.012) | g2's: 30/188 vs 31 (11/12); g1's: 83/224 vs 68 (28/13, p 0.014) | 13/400 vs 12 (11/10) | promoted |

Verdict: one claimable exam step (4B g1 over the base); on authored tasks two of three 4B steps improve. Data:
`papers/generalized-continuous-learning/data/vim-transfer/loop/rerun-4b.json`, figure `fig:rerun` (bottom row).

## Seed replicates of the generation steps: registration (2026-11-28 05:20, before any replicate result)

Known limit 5 above (one training seed per generation). Each completed step is repeated with only its newest LoRA stage's
training seed changed (1 to 2): s8b-g2-r2 and s4b-g2-r2 (forks of s8b-g2/s4b-g2, `lora-g2` seed 2), s8b-g3-r2 and
s4b-g3-r2 (forks of s8b-g3/s4b-g3, `lora-g3` seed 2). Same tasks, data, champions and stages; every evaluation re-measures
the new candidate against the same champion at 4 attempts per task. Secondary only: a replicate selects nothing and does
not replace a chain's champion. Reported: per step, both seeds' paired tests on target, retention and exam, and the step
pooled over seeds (tasks won and lost summed).
Added 2026-11-28 06:30, before its result, while a card was free: s4b-g2-r3 (`lora-g2` seed 3), and s4b-g3-r3 (`lora-g3`
seed 3) when a second card frees; reported the same way.
Added 2026-11-28 06:50, before its result: s8b-g2-r3 (`lora-g2` seed 3) on the card s8b-g2-r2 freed; reported the same way.
Added 2026-11-28 07:10, before any of their results, on cards otherwise idle: 4B seeds 4 and 5 of both steps (s4b-g2-r4,
s4b-g3-r4, s4b-g2-r5, s4b-g3-r5) and 8B s8b-g3-r3 (seed 3), s8b-g2-r4 and s8b-g3-r4 (seed 4); a replicate whose evaluations
do not finish before the lease ends (09:34) is reported as unfinished, never as a result.

## Further mitigation (2026-11-27; planned, not run)

In order of effect per hour:

1. Power the gate: 16 attempts on the held-out and forgetting evaluations; eval-forget-gN gets purpose retention; release
   requires [target, retention]. Re-judge generation 2 with the existing adapters first, with a deviation row before any
   generation-3 gate result is read.
2. Cap sven at the registered turn budget: agent.max_tool_rounds 60 in presets/sven-no-thinking.yaml.
3. Project teacher and champion separately: the teacher-only dataset is the training `dataset` (validation teacher-only);
   champion records go into replay with per_task 1 and prefer random; the champion share falls from 45% to about 18%.
4. A warm-start candidate (start: champion, teacher-only new data, replay_share 0.5, lr 1e-4, about 575 steps; about 70
   minutes for the 8B), an additional candidate under the same gate.
5. A three-arm mix test on the existing generation-1 and generation-2 data: teacher only; teacher plus one champion record per
   task; supervise productive; measured on the practice tier and with the calls-per-trial table.
6. Scaled-steps runs with patience in epochs: patience 12 if a run stops early below 80% of its steps.
7. Exams last, at 16 attempts per exam for the promoted candidate and its champion; report the sign test and a per-exam
   permutation test.

From generation 4: an author prompt whose worked example meets the size specification (20-40 lines, two files) and a
reproduce-side rejection of undersized jobs; per-source per_task, exclusion of repeat-note records, and champion records
only from tasks it passed at most 2 of 8 (about 30 lines in crates/factory/src/run/dataset/selection.rs); supervision that
masks repeated identical calls; patience and eval_every scaled with steps; a thinking-mode teacher-only side arm.


## Transfer experiment support (2026-11-26)

Landed in splinter (branch `transfer-arms`), every option absent by default so existing recipes keep their stage
identities and records:

- `dataset.project` `supervise: productive` (arm D): an action whose tool calls were all answered with an error is
  context (`train: false`); the error status is read from the request the model was sent against the tool's raw
  output in the trajectory (knowledge 226). Needs `projection: native-sft`.
- `dataset.project` `per_task: N`, `prefer: fewest-actions | random` (arm C); the rest listed as `per_task_limit`.
- `dataset.project` `control: shuffled-requests` (arm F): a seeded derangement of the tasks' first user messages;
  the manifest records `control`.
- `composition.json` for every SFT dataset (records, trials, tasks, distinct action sequences, supervised and masked
  messages, supervised characters), named by the manifest and shown in `report.md`.
- `tasks.pin` `holdout: { percent, seed, side: kept | held }` (T1): each task's side is a hash of the seed and its id.
- `dataset.project` optional input `tasks`: only trials of those pinned tasks are admitted (`not_in_tasks`).
