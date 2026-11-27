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
- Private HF dataset `swedishembedded/source-to-transfer-evidence` (commit c3eb1ecd): arm datasets with lineage, arm
  recipes, locks and frozen revisions, training reports, exam aggregates, the decision and secondary outputs, the
  authoring funnel, the material runs' manifests, the analysis code (decide.py, secondary.py, export-results.py,
  analyse.py, aggregates.py, funnel.py, gen-arms.py).
- Private HF model `swedishembedded/vim-manual-transfer-adapters` (commit ffc4aea4): the 15 adapters with their
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

Open: generation 2; a forgetting gate; an unattended restart from the recipe.

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
