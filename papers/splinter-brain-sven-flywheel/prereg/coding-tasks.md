<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of the coding-task experiments CE1, CE4, CE6 and CE7

Written and committed on 2026-10-16, 18:04 CEST, before any run of these experiments
exists. Experiment names are prefixed `CE`; the master plan's names are in
parentheses (CE1 = plan E1, CE4 = plan E4, CE6 = plan E6). Anything in this file changed
after the first run of the experiment concerned is a dated amendment below the line
"Amendments", and a result computed under an amended rule is labelled post hoc. Thresholds
and the alpha level are never changed after a result is seen; a sample too small to decide
is labelled a pilot, not rescued by a looser rule.

## Hypothesis (headline)

Verified engineering-agent experience can produce a sustained improvement in completion of
previously unseen repository tasks, under a controlled compute budget, without unacceptable
loss of earlier capability. "Previously unseen" means the qualification partition below.
Stated against: no meaningful gain, a gain explained by a control, or a retention loss, are
each a result and are reported as such.

## Common definitions

- **Task**: a repository snapshot at a pinned revision, an instruction stating a symptom,
  hidden checks (fail-to-pass and pass-to-pass) the solver never receives, and the paths
  that carry them. A task is admitted only if the snapshot's selected tests pass twice, a
  change makes at least one previously passing test fail reproducibly, and the reverse
  change restores green.
- **Family**: the unit of analysis and of splitting. For the Python and Lua repository
  families it is the file a fault was seeded in; for real repositories it is the subsystem
  (crate or top-level module). Tasks within a family are never treated as independent.
- **Partitions**: development (all design choices and pilots), qualification (one
  confirmatory query per registered comparison, never used for selection), and, for CE7,
  an independent repository. Families are assigned to partitions by a stable hash before
  any rollout. A family in the qualification partition is never solved for data purposes.
- **Arms**: the base checkpoint and the same checkpoint with an adapter, in one pinned
  harness profile (agent revision, system prompt, tool schemas, budgets, sampling), the
  same seed per (task, attempt), the same environment image. Isolation is `none` on the
  node; it is recorded in every run, and a task whose soundness needs isolation is not
  admitted.
- **Family outcome**: the fraction of a family's tasks resolved by an arm. The adapter
  wins a family when its fraction is higher and loses when lower; ties are dropped.
- **Primary test (G9)**: one-sided exact sign test over the non-tied qualification
  families, alpha 0.05, comparing the adapter served through `brain serve` and driven by
  Sven against the base in the same harness. Not significant is reported as "not shown",
  never as equivalence.
- **Size rule**: the exact binomial one-sided critical values at alpha 0.05 are 12 wins at
  15 non-tied families, 15 at 20, 18 at 25 and 20 at 30; power at a true win share of 0.80
  among non-tied families is 0.65, 0.80, 0.89 and 0.97 respectively (computed with
  `scripts/build_data.py`, design table of the paper). CE1 measures the discordance;
  qualification partition size is fixed from it before any training. With fewer than 25
  non-tied families the comparison is labelled PILOT and the decision rule is unchanged.
- **Retention (G10)**: the exam families of every earlier training round, solved by the new
  adapter and by its predecessor. Retention holds only if the one-sided 95% lower bound
  of the change in resolve rate (family-cluster bootstrap) exceeds -0.05 absolute. A
  failed retention check refuses the candidate.
- **Exclusions**: harness failures (environment faults) are excluded from denominators and
  always counted; attempts whose trajectory exposes hidden material are excluded and
  counted; nothing else is excluded. Each run record carries the three repository commits,
  a dirty flag, the binary digest and the isolation level.
- **Stop rules** (from the plan, restated so none is added after a result): a task kind
  whose verifier false-pass rate exceeds 5% after one repair is dropped; a source
  admitting fewer than 2 tasks per CPU-hour or with more than 10% witness failures is
  dropped; if the adapter of CE4 does not win on the qualification partition in two
  registered confirmatory queries (CE4 and the best dataset-selection arm), the result is
  "no held-out improvement" and no later phase is run on its strength; a development gain
  with no qualification gain (sign test p above 0.2) is reported as memorisation.

## CE1 (plan E1): what the untrained agent can do

- Question: the baseline resolve rate, failure profile and cost per resolved task of the
  untrained agent.
- Arms: Qwen3-8B and the 27B model served as a Q8_0 GGUF by `brain serve`; the base only.
- Data: development partition and, once, the qualification partition. Attempts per task
  `k` from the throughput probe, fixed before the run.
- Reported, no threshold: resolve rate with a family-cluster bootstrap interval; failures
  by class; tool calls and tokens per resolved task; wall time per task; harness failures;
  discordance between two seeds of the same arm per family (this fixes the qualification
  size above).
- Admission funnels per task source (mutation, history, authored) with witness and
  negative pass rates are reported beside it. Nothing is trained before CE1 is reported.

## CE4 (plan E4): does SFT on verified trajectories help

- Policy: Qwen3-8B. Adapter: LoRA, trained only on trajectories whose patch passes the
  hidden checks and whose exposure audit is clean, on development-partition families
  only. Assistant decisions are supervised; user, tool and system turns are context.
- Development pilot first (selection of steps and data share allowed there), then one
  confirmatory query on the qualification partition with the frozen adapter.
- Primary: the G9 test above, adapter against base, served through `brain serve`, driven
  by Sven. Secondary: G8, agreement of the served and in-process verdicts within the serve
  check's tolerance; G10 retention where an earlier round exists; cost per verified
  outcome (G11), reported with an interval and compared with CE1.
- Controls, each a separate training run on the same budget, reported as development
  comparisons: (a) trajectories regardless of verdict (does verification matter);
  (b) an adapter trained on a different repository's families (did it learn the tool
  protocol rather than the repository); (c) no replay of earlier rounds when a later round
  exists. A gain that control (a) or (b) matches on the development partition is reported
  as not attributable to verified, repository-specific data.
- Success for the headline: a qualification sign test with p below 0.05, retention not
  refused, and G8 agreement within tolerance. Anything else is reported as the stated
  outcome, not as success.

## CE6 (plan E6): does the 27B model train and deploy

Ten stages; each is pass or fail and a failure stops the later ones. (1) the exact model
and tokenizer load with a recorded tokenizer digest; (2) tool-calling inference matches the
production path (GGUF serve, FP8 streamed serve); (3) a small LoRA trains on a controlled
coding dataset of CE1 trajectories; (4) gradients are nonzero and finite and the weights
change by a nonzero amount, asserted in the run report; (5) the adapter saves and reloads
against the adapter contract; (6) base and adapted generation on fixed prompts differ;
(7) held-out task evaluation in process; (8) the adapter is served through the production
API; (9) Sven completes full tasks on the served adapter; (10) correctness, tool-call
validity and resources are compared against the base. Pass criteria: G7 (training
correctness, stages 3-6) and G8 (stage 9 against stage 7 within the serve check's
tolerance). The 27B becomes the policy of a confirmatory query only after G7 and G8 pass.
Measured throughput is reported as measured; no target is registered here.

## CE7: replication on a second, independent repository

- Purpose: to show the CE4 procedure is not specific to the repositories it was developed
  on, by running the unchanged procedure on a repository no part of the pipeline was
  designed or tuned on.
- Repository selection rule, fixed now and applied before any CE4 qualification query: a
  permissively licensed repository (licence obligations read, not assumed) that is not
  one of Splinter, Sven, Brain or the authored families, builds and tests offline, whose
  selected suite runs in under 60 s, and that admits enough families for at least 25
  non-tied qualification families under the discordance CE1 measured on it. It is named in
  an amendment dated before its first rollout.
- Procedure: the same task construction, partitioning, harness profile, arms, training
  recipe, primary test, retention rule, exclusions and stop rules as CE1 and CE4, with no
  parameter changed after CE4 and no selection on the new repository's qualification
  partition. CE1 is run on it first.
- Reported: the primary test result on the new repository's qualification partition
  whatever it is; the cross cell (the CE4 adapter on the new repository, and the
  new-repository adapter on the first repository's families), which separates learning a
  repository from learning the protocol.
- The headline is reproduced only if both repositories pass the primary test
  independently; if one passes and the other does not, the paper reports that and does not
  claim replication.

## Amendments

### Amendment 1, 2026-10-16, 22:00 CEST: teacher-distilled CE4, CE8, harness profile

Written and committed before any run of CE1, CE4, CE6, CE7 or CE8 exists. It changes the
design below where it says so and nowhere else; the text above is left as registered. Where
the amendment and the text above disagree, the amendment governs.

**Reason.** Two pilot observations, both outside this registration: they were made on one
repository family (10 development tasks, one attempt each), with a harness profile that this
amendment replaces, and were not frozen in advance. They are cited as the reason for the
change, not as results of CE1 or CE4. They are recorded in the lane notes
`runs/l1-measure/results.md` and `runs/l1-measure/rerun/results.md` on the project drive. (i)
The untrained Qwen3-8B almost never edited a file (one `edit_file` call in 67 requests of
the re-run) and none of its attempts passed the hidden checks; the diagnosis found that
in non-thinking mode it ends turns with prose after announcing an action. Its own verified
trajectories are therefore too few to train on, and the CE4 text above (trajectories of the
policy being trained) cannot produce a training set. (ii) The served 27B passed some of the
same tasks (2 to 3 of 10 over runs; the difference between runs is within noise at that size).
(iii) `splinter solve` ran with no follow-up, no tool-call cap and no repetition stop, which
the lane diagnosis found to be a cause of both observations.

**A1.1 CE4 changes: where the training trajectories come from.** The student is still
Qwen3-8B with a LoRA adapter, and the training set is still only trajectories whose patch
passes the hidden checks and whose exposure audit is clean, from development-partition
families only, with assistant decisions supervised and other turns as context. What changes:
the trajectories are produced by the teacher, Qwen3.8-27B served as the base (the CE1 serving
path, no adapter), solving development-partition tasks in the harness profile of A1.3. CE4
therefore tests distillation of verified agent experience from a stronger model into a
smaller one. The student's own trajectories are not used for training.
Arms of the primary test: the untrained Qwen3-8B (base) against Qwen3-8B with the adapter,
both served through `brain serve` and driven by Sven in the same pinned profile. The teacher
is not an arm of the test. Its CE1 resolve rate on the qualification partition is reported as
context, with no threshold and no test. The three controls are restated for this source:
(a) teacher trajectories regardless of verdict; (b) verified teacher trajectories from a
different repository's families; (c) no replay of earlier rounds when a later round exists.
A gain that (a) or (b) matches is reported as not attributable to verified,
repository-specific data. The teacher's qualification-partition families are never solved
for data purposes (the existing partition rule applies to the teacher as to any solver).

**A1.2 What does not change in CE4.** The partitions and their stable-hash assignment, the
exposure audit, the primary test (one-sided exact sign test over non-tied qualification
families, alpha 0.05, adapter against the untrained 8B), the size rule and the pilot label,
G8 and G10 and the retention rule, cost per verified outcome (G11), the exclusions, the stop
rules (including the two registered confirmatory queries and the memorisation rule), and the
success condition for the headline. The headline is read as before, with the stated source of
the training data.

**A1.3 Harness profile (all experiments).** Every solve of CE1, CE4, CE6, CE7 and CE8, for data
collection and for evaluation, runs in one pinned profile with these episode bounds: at most
2 follow-up turns, at most 60 tool calls, and a repetition stop at 3. A follow-up is also given when the worker ends a turn with the tree unchanged from the
snapshot, whether or not a visible check is red, and it counts against the 2 follow-ups. The
follow-up text is, from the implementing change (splinter commit 2019ea4): "You have changed
no file yet. Make the change now by calling the edit tools; do not only describe it." The
bounds are implemented by splinter commit afcef42, which makes `splinter solve` use the same
episode bounds as the other verbs; the pinned splinter revision of every run must contain
both commits. All arms use the same profile. Results measured before this profile (the pilot
observations above and any other `splinter solve` output made without the bounds) are not
comparable and are excluded from every CE analysis.

**A1.4 CE8: self-improvement of the 27B model.** Question: does the 27B model improve on
unseen repository tasks when fine-tuned on its own verified trajectories. Policy and teacher
are the same model: Qwen3.8-27B. Training data: trajectories of the served 27B base on
development-partition families in the profile of A1.3, only those whose patch passes the
hidden checks with a clean exposure audit. Everything else follows CE4: the LoRA recipe,
the partitions, the one qualification query, the sign test at alpha 0.05 against the 27B
base, retention, G8, G11, the three controls, exclusions and stop rules. Arms are the 27B base
and the 27B with the adapter, both served through the path that serves the adapter (CE6
stage 8), so base and adapter share weights and precision; the CE1 numbers for a different
serving format are not that arm. Precondition: CE6 passes G7 and G8, so that the adapter
serving path exists; if it does not, CE8 is not run and the paper says so. CE8 is a separate
registered comparison with its own qualification query and its own test; it does not enter
the headline, whose success condition is CE4's (and CE7's for replication), and its result
is reported whatever it is. It is not run on the strength of CE4: CE4 not winning does not
cancel it, and CE4 winning does not substitute for it.

**A1.5 Experiment list.** The registered experiments are CE1, CE4 (amended), CE6, CE7 and CE8.
CE7 repeats CE4 as amended on the independent repository; its selection rule is unchanged.

### Amendment 2, 2026-10-16, 23:58 CEST: run parameters, sizing, evaluation decoding

Written and committed before any run of CE1, CE4, CE6, CE7 or CE8 exists, and before any
admission, reserve or rollout of the code that applies it. It registers values the text above
left open and corrects three statements that the code cannot implement as written. It changes
nothing else, and where it and the text above or Amendment 1 disagree, this amendment
governs. Its sources are the runbook `runs/plan-2026-10-16/ce-runbook.md` on the project
drive (sections 0 to 3) and splinter main d9a122d.

**A2.1 Commit equivalence (corrects A1.3).** A1.3 requires that the pinned splinter revision
contain commits 2019ea4 and afcef42. Neither is an ancestor of main; main carries rebased
copies. `git patch-id --stable` gives the same patch id for each pair: 1fcb155a04bd for
2019ea4 and e4ae3d6, and 698924fba3dd for afcef42 and add9481 (full ids
1fcb155a04bdf9a2b24d0b8c40377e1c45674046 and 698924fba3ddaa75e85b18c885be38ab0ebab85b). The
requirement of A1.3 is met by a revision that contains each of 2019ea4 or e4ae3d6, and
afcef42 or add9481. The text of the follow-up and the episode bounds are unchanged.

**A2.2 Evaluation decoding (corrects "the same seed per (task, attempt)").** Models served
through `brain serve` cannot be seeded or resampled by the harness, so a shared seed per
(task, attempt) cannot be implemented. Both evaluation arms of every comparison (the
untrained model and the model with the adapter) are run with greedy decoding (temperature
0) and one attempt per task (k = 1), in the A1.3 profile, against one server instance and
one build, one arm at a time. Thinking mode is off for every model. The teacher is run twice
for training data only: pass 1 at greedy, pass 2 at temperature 0.7, unseeded and recorded
as such; the two passes are not arms of any test. The "attempts per task k from the
throughput probe" of CE1 is therefore fixed at 1 for the evaluation arms.

**A2.3 Sizing input (replaces the CE1 discordance of two seeds).** Under greedy decoding the
discordance between two seeds of one arm is zero by construction and carries no information.
The sizing input is instead, per development family, the base 8B and the 27B teacher
outcomes in pass 1: the number of families on which their family outcomes differ, as an
upper bound on the share of non-tied families for an adapter that reaches the teacher. It is
reported with CE1. The qualification partition size is registered below and is not changed
by it. The comparison is labelled PILOT when the qualification query yields fewer than 25
non-tied families, by the rule of the text above, whatever that count is.

**A2.4 Partition values (task generation, Q_f, SEED_EXAM).** Tasks are generated with
`--mutants-per-file 40` per file under test. The exam seed is registered now: SEED_EXAM =
20261017, used as `--exam-seed 20261017` for every repository, so the first run remembers it
and later runs refuse another. The qualification partition is the set of families `practice`
reserves: families are ranked by a stable hash of (seed, file path) before any rollout and
the first Q_f of each repository, in rank order, are reserved (the rule of
`crates/pipelines/src/practice/exam.rs`); the development partition is the remaining
families. The numbers reserved per repository (`--exam-families`), Q_f, are: iniparse 6 of
11 files, ttlcache 5 of 9, exprcalc 7 of 12, linediff 5 of 9, mdrender 5 of 9 (the Python
families, 50 files), caldate 5 of 8, graphkit 7 of 12, tokenbucket 6 of 10 (the Lua
families, 30 files). That gives 46 qualification families (28 Python, 18 Lua) and 34
development families (22 Python, 12 Lua), about 56% and 44% of 80 files. Python and Lua
families are both included. The reason: the Python families alone have 50 files, so at most
25 qualification families, and the size rule asks for 25 non-tied families, which cannot be
reached when any family is tied; including the Lua families gives 46 clusters, which reaches
25 non-tied families if the adapter resolves at least about 8% of tasks (derived from about
8 tasks per file, a base near 0, and independent families; a design calculation, not a
measurement). The count of admitted tasks per family is a measured outcome and is reported;
nothing is added or removed after the partition is made.

**A2.5 Family outcome.** For a family, the outcome of an arm is its resolved fraction over
the tasks graded in both arms (`paired::by_cluster` of `crates/eval/src/paired.rs`, applied
to the experiences paired by task id). Attempts whose trajectory exposes hidden material and
attempts with a harness failure are excluded from that pairing and counted, per arm and per
reason, as the Exclusions rule already requires; a task excluded in either arm is dropped
for both. The adapter wins a family when its fraction is higher, loses it when lower, and
ties are dropped.

**A2.6 Serving flags (registered values).** All brain servers are one build of brain, and
the splinter binary is one build, for the whole of CE1 and CE4.
- Qwen3-8B (CE1 base arm, both CE4 arms, controls): `--qwen-ctx 32768 --qwen-max-batch 32
  --qwen-weights-fp32 --adapter-targets all --adapter-rank-cap 16`, with `--adapter <the
  frozen adapter>` for the adapter arm on the same server. The 8B is served with fp32
  weights because the adapter is trained on the fp32 base and the default int8 weights would
  put a quantisation under the adapter that it was never trained against. Concurrency 32.
- Qwen3.8-27B teacher (Q8_0 GGUF): `--qwen35-ctx 32768 --qwen35-kv int8 --qwen35-max-batch
  16 --qwen35-prefix-slots 16 --admit-deadline-ms 900000`. Concurrency 16. The context
  length the server advertises must read 32768.
- The smoke of the runbook sets these. It may lower the 27B `--qwen35-max-batch` and
  `--qwen35-prefix-slots` together, only when the server refuses the configuration for
  memory; any such change is recorded in the run record and is the only change allowed. The
  KV precision, context length and fp32 weights are not changed.

  Flag names, after brain made its serving options model-agnostic (a spelling change, not a
  change of any value above): `--qwen-ctx` and `--qwen35-ctx` are `--ctx`, `--qwen-max-batch`
  and `--qwen35-max-batch` are `--max-batch`, `--qwen-weights-fp32` is `--weight-format f32`,
  `--qwen35-kv int8` is `--kv-cache int8`, `--qwen-weights` is `--checkpoint`. The prefix-cache
  size is no longer a flag: brain sets it to the batch (16 slots at `--max-batch 16`, the
  registered value). The older spellings still work with a deprecation warning.

**A2.7 LoRA recipe and step rule.** Rank 16, alpha twice the rank, learning rate 2e-4, two
epochs, 8 records per step, monitoring share 0.2 held out by whole family, `--max-block
16384`, training seed 1, and the best monitored evaluation is kept (`--keep-evaluations`).
Steps = ceil(2 x R / 8), where R is the number of training records after the monitoring
share is removed. All seven linear projections are adapted. Fewer than 40 verified
trajectories after teacher pass 2 means no training is run and the shortfall is reported as
the outcome. The dataset is the verified, exposure-clean, `executable`-strength `sft-step`
view of A1.1 only; trajectories of compacted contexts are dropped by the view and counted.

**A2.8 Packed or unpacked records.** The unpacked `sft-step` view writes one record per
action, each carrying the whole conversation so far, so its size grows quadratically with
trajectory length. If `dataset build --view sft-step --pack` lands before training with its
equivalence test (the supervised spans of the packed record equal the union of the unpacked
records' spans, and a step labelled bad stays unsupervised), the packed view is the registered
training view; otherwise the unpacked view is used with the records capped evenly by
`--limit`, the cap being recorded. The two weight the loss differently: unpacked, every action
is a record and the loss is averaged per record, so a long trajectory counts once per action
with its earlier context repeated; packed, the loss is averaged over supervised tokens of the
record. This difference is stated and not corrected for.

**A2.9 G8 sample.** The G8 serve-verdict agreement is judged on one task per qualification
family (46 tasks), chosen by stable hash of (SEED_EXAM, task), solved in process with the
adapter and compared with the verdicts of the served adapter on the same tasks, at the
threshold of the serve check (at least 75%).

**A2.10 CE1 and CE4 share the 8B base arm.** The CE1 run of the untrained Qwen3-8B on the
qualification partition and the base arm of CE4 are one run, made once, at the confirmatory
query, on the same server instance as the adapter arm and in the same profile (A2.2, A2.6).
No qualification-partition result of the base exists before that query.

**A2.11 CE7 repository.** python-markdown 3.11 does not meet the CE7 size rule: about 30
module files under test would give at most about 15 qualification families, below the 25
non-tied needed, and its wheel carries no tests. The repository is not chosen here. The
candidates are docutils and sqlglot at a pinned tag; the selection rule of CE7 is
unchanged and is applied by a later dated amendment, which names the repository, tag and
licence, and is committed before the first CE4 qualification query.

### Amendment 3, 2026-10-17, 08:15 CEST: workspace confinement, operational bounds, CE7 named

Made before any qualification-partition query and before any training run. No result of
a registered arm exists; the only agentic run so far (the 27B teacher, pass 1, on the
iniparse development set) is withdrawn by A3.1 and trains nothing.

**A3.1 Confinement of the solver's workspace.** The 27B teacher run of 2026-10-17 ran its
shell with no filesystem confinement. In 27 of 61 trajectories the model searched the host
filesystem, read and diffed the original task sources and tests outside its working tree,
read the repository's history and listed other attempts' trees; 7 of its 28 raw passes did
so, and the exposure audit as registered did not flag it. That run is withdrawn. From this
amendment every command a workspace runs (the solver's shell and every test run) is confined
by the kernel's Landlock to its own tree, a private temporary directory and the system
directories; where Landlock is unavailable the run is refused. The exposure audit gains the
rule `outside_workspace`: a tool call that names a path outside the episode's tree (other than
system directories) marks the experience exposed, and it is excluded like any exposed
attempt. Both apply to every arm, the teacher and the controls.

**A3.2 Leakage at dataset build.** A teacher trajectory may touch a qualification family's
file even when confined, since the snapshot holds the whole repository. The training dataset
is built with `dataset build --drop-leaking`: every experience that would cross the exam seal
(an attempt at a reserved family, or a supervised action naming a reserved family's file) is
left out and counted; the seal is then applied to what remains and refuses the build if
anything still leaks. The count and the dropped ids are reported with the dataset.

**A3.3 Operational bounds.** The wall-clock bound of one task's solve is 3600 s
(`--solve-deadline 3600`) instead of 300 s: under one shared served model the 300 s bound
measured the queue, not the solver (every task of the first development set timed out at
concurrency 16). It guards against a stuck episode only; episodes stay bounded by the
registered 60 tool calls, two follow-ups and the repetition stop. The 27B is served with the
A2.6 flags; the concurrency of the 27B arm may be lowered from 16 to 8 when the server
measurably serves fewer requests per minute at 16 (it did: about 4 against 15), the choice
and its measurement being recorded. Greedy outputs do not depend on concurrency.

**A3.4 Binaries.** Every registered run from this amendment on uses one splinter binary and
one brain binary, named in the run record. Admission of the eight repositories was made with
splinter builds that differ from that binary only in dependency pins, flags, the exam seal,
packing and run reports, none of which changes admission; it stands.

**A3.5 Correction to A2.8.** The training step averages its loss over every supervised token
of the records it accumulates, in both the packed and the unpacked view. Packing changes what
one record holds (a whole trajectory instead of one action with its context), so it changes
how many records an epoch draws and how much repeated context is computed, not how a token is
weighted within a step.

**A3.6 CE7 repository: sqlglot.** Applying the CE7 selection rule to docutils, sqlglot and
python-markdown: python-markdown fails the size rule (A2.11); docutils admitted no task in a
three-file funnel; sqlglot v30.23.0 (commit ad8764a0436a7b5c500b783ebe50dbf20c1e477c, MIT
licence, 174 module files, offline build and test, `python3 -m unittest` over 55 test
modules, excluding the two that need numpy and pandas, in about 17 s) admitted 4, 10 and 5
tasks on its three-file funnel at 40 mutants per file. CE7's repository is sqlglot at that
commit, built as a fresh git repository of its tracked files (without the integration-test
submodule). Its admission keeps a head and a tail of an over-long test report and decides the
verdict from the runner's summary (`--test-output-cap`, 1 MiB per stream), and compares
near-duplicate faults by their failing tests only; the eight CE4 repositories keep the
admission they have.

### Amendment 4, 2026-10-18, 23:55 CEST: controls and arms as built, teacher passes, binaries

Made after the containment smoke and the start of teacher pass 1 on the development
partition, and before any dataset is built, any training run and any qualification-partition
query. No result of a registered arm exists. It registers what A1.1 and the roadmap left as
names and changes no threshold.

**A4.1 Teacher pass 1 as run.** Pass 1 (greedy, one attempt, `--solve-deadline 3600`, concurrency
8 per server, A3.3) runs on two 27B servers, one per GH200, each with the A2.6 flags. The
development sets are split by repository: card 0 solves iniparse, exprcalc, linediff and caldate
(194 tasks), card 1 ttlcache, mdrender, graphkit and tokenbucket (178 tasks). Greedy outputs do
not depend on the split. The binaries of the run are splinter b5fb1674 (built against local
brain and sven checkouts, whose commits are in the run record) and brain 5c1aa4863; the run
passes `--test-output-cap 65536` and the default search path, because the tasks were admitted
under those and a solve refuses an environment that differs (A3.4). Containment was checked
before it on 5 development tasks (0 host paths in 5 trajectories, 1 `outside_workspace`
exposure from a `find /` call that returned nothing).

**A4.2 What "cancelled" means.** A solve concludes `cancelled` when the harness itself stops the
run: the cap of 60 tool calls, the repetition stop of A1.3 (the same call returning the same
result three times), or a caller's cancel. The experience records the conclusion but not which
of them it was; none counts as a pass and none carries a final output. The count is reported
with CE1. This is a description of the registered harness, not a change to it.

**A4.3 Controls as built.** All are development-partition comparisons (A1.1), one adapter each on
the A2.7 recipe and the same number of optimiser steps as the primary adapter, evaluated like the
primary adapter on the development partition, greedy, k = 1, same server build, one arm at a
time, compared with `compare`.
- (a) Unverified: `dataset build --view sft-step --pack --drop-leaking --control unverified`,
  the same experiences whatever their verdict, never one an exposure audit failed.
- (b) Other repository: the primary recipe on the verified, exposure-clean teacher trajectories of
  the CE7 repository (sqlglot) only, records capped evenly to the number of the primary dataset.
- (c) Replay: not run; no earlier round exists.
- A5, shuffled: `--control shuffled --control-seed N` on the primary dataset's experiences:
  every trajectory under another task's instruction, by a seeded permutation with no fixed point
  (a negative control for pairing, with format, tools and token count preserved).
- A1, plain patch SFT: `--view sft-final` on the primary dataset's experiences (instruction to
  final patch, no tool turns), as the ordinary-fine-tuning baseline.
A3 (replay) and A4 (preference) of the roadmap are not registered and not run: A3 needs an earlier
round, A4 a pair view over solved and failed attempts of one task, which pass 1 and pass 2 may
not yield. If a control's dataset has fewer than 40 trajectories it is reported as not run for
that reason (A2.7 applies to every arm). A control that matches the primary adapter on the
development partition is read as A1.1 says.

**A4.4 Pass 2.** Pass 2 of the teacher (temperature 0.7, unseeded) runs after pass 1 on the same
servers, development partition only, same bounds. Training data are the verified,
exposure-clean trajectories of both passes (A2.7); a task solved in both passes contributes both
trajectories.

**A4.5 Binary switches.** A binary is named per solve of one development set (A3.4). A faster
brain build may replace the one in use between two sets of pass 1 or pass 2, never inside one;
the run record of each set names its binaries, the number of sets run on each is reported, and
no result compares sets across builds. Teacher trajectories are training data, not an arm of any
test, so a switch between sets does not change a test. The two arms of every comparison (the
untrained 8B and an adapter, and every control against the primary adapter) are always run
against one server instance of one build, one arm at a time (A2.2, A2.10); a switch before
the first arm of a comparison starts is allowed, and between its arms is not. The numerics of
a faster build (precision, kernels) are reported with the evidence that they preserve
accuracy, and the first set run on a new build is repeated on the old one when it can be, to
report the agreement of verdicts.

**A4.6 Infrastructure errors and resumed solves.** The node's memory cap has killed servers
mid-run. An attempt in which the solver could not be reached or failed to run (a dead server, a
closed connection) is an infrastructure error: it is reported with its task, it has no experience
and no conclusion, it counts as neither a pass nor a failure, and the exclusion rule for harness
failures applies to it. The solve keeps every attempt that completed (made durable as it finishes,
with a partial experience set named in its progress lines) and `solve --resume <set>` asks only
the tasks the set does not hold, against a restarted server with the same flags; the experience
of a resumed task is that of its first complete attempt, and greedy outputs do not depend on the
restart. The number of infrastructure errors and resumes of every set is reported with CE1.
Concurrency slots are also no longer held by a finished attempt behind a slower one (the harness
now starts the next task as soon as any attempt ends); this changes wall time, not outputs.

**A4.7 Teacher server flags per set.** A2.6 registers the 27B server's context (32768), KV
precision, batch and prefix slots; the first two sets of pass 1 (iniparse, ttlcache) ran on exactly
those flags. Three findings then changed what serves the remaining sets, none of them a
decoding parameter. (i) `--speculative off`, which brain documents as leaving greedy output
unchanged and which measured 33% faster on a multi-turn replay (runs/b27/status.md), is used from
the third set on. (ii) The batch and prefix slots are lowered together to the 8 concurrent
episodes of A3.3 (A2.6 allows lowering them together). (iii) A served context of 65536 was tried
on two sets (mdrender on card 1, exprcalc on card 0), because 22k-token prompts reached the
agent's compaction threshold in a 32768 context and 32 of the 61 withdrawn-run episodes show
compaction before they were stopped. It measured 68.5 GB resident and served 44 k-token prompts,
but an episode still outgrew it (67140 tokens, mdrender lost, A4.8) and requests of 20 to 60 k
tokens made a set several times slower than at 32768, so the context returns to the registered
32768 for every later set. Every set records the flags and the binaries of its server in
`runs/ce/STATUS.md`, and the number of compacted trajectories per set is reported with CE1. The
student is unaffected: the 8B arms keep the A2.6 flags and no training record exceeds the
16384-token block.

**A4.8 Lost set, new splinter build, stricter containment.** The first solve of the iniparse
development set (card 0, splinter b5fb1674, 61 tasks, 9457 s) ended with an infrastructure error:
the server failed an over-context request ("needs 33805 tokens, engine capacity is 32768") without
answering it, the client gave up after its 600 s first-byte bound, and the old harness discarded
every result of the set, which it held in memory until the end. It produced no experience, no
verdict and no number, and the set is solved again from the start; the loss is recorded here and
in `runs/ce/STATUS.md`, not counted as a solve. A second set (mdrender, 6697 s) was lost the same
way when an episode outgrew even the 65536 context (67140 tokens): enlarging the context only
delays an overflow, so the harness now records an episode whose request the server refuses as
too long as a skipped task (`episode outgrew the served context`), reported with the set, never
retried and never a pass, and keeps every other result. The sets after it run on splinter
19295459 (the commit of this amendment's branch that built the binary), which
contains the A4.6 handling and the containment of splinter 8384f9dc (network denial, a scrubbed
environment, a non-dumpable supervisor, delegated children pinned to the project root, and an
exposure audit that reports an outside path as denied, absent or reached; all three still mark the
experience exposed, so the A3.1 rule is unchanged); it changes no environment digest of an
admitted task. The brain binary of the servers stays 5c1aa4863 (A4.5: a binary is named per set).
The set-by-set record of binaries and server flags is in `runs/ce/STATUS.md` and is reported
with CE1.
### Amendment 5, 2026-10-19, 09:45 CEST: the 8B numeric tier, admission of training evidence, binaries of datasets

Made before any dataset is built, any training run and any qualification-partition query. CE1
pass 1 on the development partition continues under Amendments 3 and 4 and is not changed.

**A5.1 The tier of the 8B arms.** A2.6 serves Qwen3-8B with fp32 weights and the training of A2.7
adapts the fp32 base; both were measured on brain's exact fp32 GEMM path, which runs the 8B at
984 prefill tokens/s and 275 LoRA training tokens/s. Brain now computes on tensor cores by
default (TF32 GEMMs, native attention): on real Qwen3-8B weights, 16 prompts of 256 tokens, int8
KV, against the exact tier the greedy tokens are equal on 144 of 144, the top-1 token on 16 of
16, the KL 1.6e-4 mean and 6.4e-4 at most; the training forward has relative RMS logit
difference 6.8e-4 and top-1 2048 of 2048, the LoRA step is 8.5 times faster and a served
prefill about 6 times faster (runs/b8/results.md, brain 502e12d46). The 8B base arm, every
adapter arm, every control and the training of every adapter therefore run on the default
tensor-core tier, the same for both arms of every comparison, with fp32 weights
(`--qwen-weights-fp32`) and no `--qwen-gemm-fp32`; every command and the run record name the
brain build (502e12d46) and these flags. An 8B server without `--qwen-weights-fp32` serves int8
weights (greedy tokens equal on 49 of 144 against fp32) and is never used for a registered result.
The exact tier stays the reference: the per-task verdict agreement of the untrained 8B on one
development set between the two tiers is reported with CE4, with no threshold. Nothing else of
A2.6 and A2.7 changes.

**A5.2 Admission of training evidence.** Every training dataset of CE4 and CE7 and every control
that trains on verified evidence is built with `--control outcome-verified`: only experiences
decided pass at executable strength by the hidden checks are admitted, and an action is never
admitted on a step label alone from an experience nobody verified or whose verdict failed. The
manifest records it. The unverified control of A4.3 is the one dataset that does not.

**A5.3 Datasets are tied to a binary.** A dataset is built by the splinter binary named in its run
record (64cad0e6 for CE1 and CE4), and a dataset built by another binary is built again before it
is trained on. Training links brain at the commit named in A5.1.
### Amendment 6, 2026-10-19, 09:48 CEST: CE7 admission parameters

Made before the admission of sqlglot and before any rollout on it. It fixes the values A3.6
left open and nothing else.

**A6.1 Files, tests, partition.** The files under test are 60 of the 113 module files of
`sqlglot/` with at least 60 lines whose stem a selected test names (the rule of the selection
funnel), taken at evenly spaced ranks of the sorted list (`runs/ce/ce7-files.txt`); the suite
is `python3 -m unittest` over the 55 test modules of the selection record
(`runs/ce7-select/sqlglot-modules.txt`: all but the executor and optimizer modules, which need
numpy, pandas and duckdb), in the virtual environment of that record placed first on PATH, which
becomes part of the recorded environment of every sqlglot task. Tasks are generated with
`--mutants-per-file 40`, `--test-output-cap 1048576` and `--workspace-max-processes 512`, with
the exam seed 20261017 and 30 of the 60 families reserved for qualification (`--exam-families
30`); the other 30 are the development partition. As for the eight repositories, the count of
admitted tasks is a measured outcome and nothing is added or removed after the partition is made.

**A6.2 What counts for the 25 non-tied families.** The CE7 qualification query is labelled
PILOT when it has fewer than 25 non-tied families, by the existing rule; 30 families are
reserved so that 25 are possible.

**A6.3 Parallel admission, stratified partition.** One call over the 60 files evaluates the mutants
one at a time at about 20 s each (the 55-module suite), which would take about 13 h, and the first
such call, started at 09:46, was stopped at 13:20 before it had stored a task. The admission therefore
runs as ten parallel calls into the one state root: call g takes the files at ranks g, g+10, g+20,
... of the list (six files) and reserves three families (`--exam-families 3`), by the same stable
hash of the exam seed and the file path within its group. The partition is thereby stratified:
30 of the 60 families are reserved, three of every group of six, and the union of the ten exam
sets is the qualification partition and the union of the ten development sets the development
partition. Everything else of A6.1 is unchanged. No task of sqlglot has been solved.
### Amendment 7, 2026-10-19, 14:41 CEST: pass 1 only, the controls that are run, how the steps are fixed

Made before the training dataset is built, before any training run and before any
qualification-partition query. It rests on the sets of pass 1, whose counts are in the paper's CE1
table and are not used to choose anything here.

**A7.1 Teacher pass 2 is not run.** A4.4 registered a second teacher pass (temperature 0.7) as
further training data. The compute lease that this experiment runs in ends before such a pass could
finish and still leave room for the evaluation arms, so the training data are the verified,
exposure-clean, executable-strength trajectories of pass 1 alone (A2.7, A3.2, A5.2). The floor of
40 trajectories of A2.7 applies to them; if fewer remain, no adapter is trained and the shortfall is
the result.

**A7.2 Steps.** The step count of A2.7, ceil(2R/8), is computed from R, the number of training
records after the monitoring share is removed, as the trainer reports it for the dataset; if the
count differs from the one the run was started with, the run is repeated with the reported one.

**A7.3 The controls run, and the one that does not.** On the development partition, with the same
recipe and the same number of steps as the primary adapter: the unverified control and the shuffled
control of A4.3, and plain patch fine-tuning (A1, `--view sft-final`) on the same experiences.
Control (b) of A1.1 (the other repository's trajectories) needs the CE7 teacher pass and is not
run; the paper says so. A3 and A4 are not run (A4.3). The G8 sample of A2.9 is not run in this
lease and is reported as not run.

**A7.4 Yield accounting.** The exposure rule of A3.1 is applied as registered. The paper reports, per
set, the verified attempts by the hidden checks and those that remain after the exposure audit, by
the audit's classes, so that a reader sees how much of the evidence the rule removes.

**A7.5 Where arms run.** The base arm and the adapter arm of one comparison run on one 8B server
instance (A2.10) on one card; different comparisons may run on different cards at the same time.
### Amendment 8, 2026-10-19, 14:58 CEST: parallel arms, qualification in two halves, controls not run

Made before any 8B arm has been run on any partition and before any adapter has been served; the
primary adapter is trained but no outcome of it exists.

**A8.1 Arms in parallel.** A2.2 and A2.10 ask for the base arm and the adapter arm on one server
instance, one after the other. The lease that remains for the confirmatory query is too short for that
for 487 agentic tasks per arm, so the two arms of the primary comparison run at the same time on two
8B servers of the same build, flags and tier (A5.1), one per card, both started with the primary
adapter in their adapter bank, each asked for its own model id. Greedy outputs do not depend on the
instance or on the concurrency (A3.3); the build, the flags, the harness profile and the
concurrency are the same for both arms. The same holds for the arms of the development comparisons.

**A8.2 Two halves.** The qualification partition is solved in two halves, chosen by the stable family
hash of the exam seed before any arm has run: the first half is the 23 families (253 tasks) that
`tasks pick --seed 20261017 --family-share 0.5` returns, the second half the other 23 (234 tasks).
Both arms solve the first half, then the second. If time ends before the second half is complete in
both arms, the comparison is made on the first half alone, over complete families, with the family
count stated, and is labelled PILOT by the rule of A2.3 whatever that count is; a family is never
split between the two halves, and no task is dropped from a half after its outcome is seen.

**A8.3 Controls.** The plain-patch control of A7.3 builds no record for workspace tasks (the
view admits none), so it is not run. A7.3's other controls are run after the primary comparison and
only if time allows; the paper reports which were run.
### Amendment 9, 2026-10-19, 16:05 CEST: the training records, the step count, the qualification subset

Made before any verdict of an 8B arm exists on any partition. The first adapter (trained on 67
packed records, 9 steps) was served and driven for about ten minutes on the first qualification half;
no episode of it finished (every request ran to the 4096-token cap) and no verdict was taken, so that
adapter, its partial arm and the partial base arm that ran beside it are discarded
(`runs/ce/runs-ce1/superseded/`).

**A9.1 Why the first adapter was wrong.** The agent in the pinned sven exports a whole run as one
user step and one agent step holding every tool call and observation (graphkit: 2 steps, one with 61
calls). The packed sft-step record of such a trajectory is the instruction and one assistant turn
that holds every call and none of the results (3 messages for each of the 67 records), which is not
what an agent sees. The projection therefore reads such a trajectory, with `dataset build
--serialise-calls`, as one call per turn, each followed by its result, then the final reply; a trajectory
with several agent steps is projected as before. Each request of the run is recorded on the step with
its tokens: of the 193 experiences that carry them, 118 (61%) have as many requests as calls or one
more, and 2269 calls were made in 2128 requests, so about 94% of the calls were the only call of
their response. The approximation is the recorded behaviour for those and serialises the
remaining parallel batches.

**A9.2 Records and steps.** The training set is the unpacked sft-step view (one record per
supervised action, A2.8's alternative) with `--serialise-calls --control outcome-verified
--drop-leaking --min-strength executable`, over the eight finished pass-1 sets, built by splinter
ec7ab240; the packed records are longer than the block for 70 of 86 trajectories. The block bound
is 8192 tokens and not the 16384 of A2.7: with 16384 a step measured about six minutes, which
fits no schedule (a first run was stopped after five steps, 16384-token records, 398 of them);
with 8192 the dataset holds N = 231 records. The trainer reports R only after it has split off the
monitoring families, so the step count is fixed from N as ceil(2 x 0.8 N / 8) = 47 steps; the
reported R and the step count it implies are stated with the result, and a difference of more
than five steps is stated as a deviation. The floor of 40 is met.

**A9.3 Qualification subset.** The 8B server does not reuse a conversation's prefix between turns,
so every request re-prefills the whole history; two arms measured about 0.7 tasks per minute each
at concurrency 32, and 487 tasks per arm would take about 11 hours. Both arms therefore solve the
subset `S3` of the first half of A8.2: at most three tasks of each of its 23 families, chosen by
`tasks pick --seed 20261017 --per-family 3` (69 tasks) before any verdict exists. The comparison is
a PILOT over 23 families by the rule of A2.3, and the second half is not run in this lease.

**A9.4 Binaries.** Datasets and training: splinter ec7ab240 (sven b87b1640, brain 502e12d46 linked);
the arms: splinter 64cad0e6, brain 502e12d46 servers (A5.1); the base arm's server carries an earlier,
discarded adapter in its bank, which does not act on base requests.

**A9.5 Extension of the subset.** The subset arms ran faster than the first estimate (the first 32 of
69 tasks of the base arm finished in 15 minutes). Both arms therefore also solve the remaining tasks
of the first half of A8.2 (`tasks pick --seed 20261017 --per-family 3` returns the rest as its second
set), base arm and adapter arm each on one server as before. The comparison reported as the primary
pilot is the whole first half (23 families, 253 tasks) if both arms have it complete by 18:45 CEST,
and `S3` (69 tasks) otherwise; the other is reported beside it. This rule is fixed before the
verdict of any arm has been read.

**A9.6 One task per family.** The 8B arms finished far fewer requests than the first estimate: at 32
clients a request waits about two minutes (every turn re-prefills its history, A9.3) and about 19
requests complete per minute, so the 69 tasks of `S3` would need about two hours per arm and the
remaining time does not allow it. The confirmatory pilot is therefore `S1`, one task of each of the 23
families of the first half (`tasks pick --seed 20261017 --per-family 1`, 23 tasks), solved by both
arms with all 23 episodes at once, base arm and adapter arm each on one 8B server. A9.5 does not
apply (neither `S3` nor the whole half is run). An attempt lost to a client time-out is an
infrastructure error and is repeated by a resumed solve (A4.6); it is never a failure. This
choice is made before the verdict of any arm has been read: the first base arm was lost to
time-outs (67 of 69 attempts) and has no verdicts.

### Amendment 10, 2026-10-19, 19:47 CEST: the CE7 teacher pass and dataset

Made before any sqlglot task has been solved and before any verdict on it exists. It fixes
how the CE7 teacher pass and the CE7 training dataset are produced and changes no parameter
of A6.

**A10.1 Admission of group 0.** The first call of group 0 of A6.3 was still running when the previous lease ended and stored no
result; it is run again alone, with the same command and the same binary (splinter 64cad0e6),
over `runs/ce/ce7/group0.txt`. The development partition is the union of the ten development sets
and the qualification partition the union of the ten exam sets, built with `tasks pick --seed
20261017 --family-share 1` over the ten admission runs.

**A10.2 Teacher pass.** One pass over the development partition: the 27B teacher, greedy (temperature
0), one attempt per task, `--solve-deadline 3600`, concurrency 8 per server (A3.3), pass 2 not run (A7.1),
`--test-output-cap 1048576 --workspace-max-processes 512` and the virtual environment of A6.1 first on
PATH (the recorded environment of the tasks). The server flags are those of A2.6 with the values of A4.7
(context 32768, batch 8, prefix slots 8, speculative off) on brain 98106c845. The splinter binary is the
build of origin/main 7bd63719, which links sven 319a2e05 and brain 83e2f44a3 (admission by 64cad0e6
stands, A3.4); sven 319a2e05 exports a run as a whole conversation with its true turns, so the
projection of A9.1 is not needed. Sets that cannot be run in one call are split by family and each
set is named with its binary in the run record (A4.5). A second server on the other card, when
one is available, takes a disjoint half of the development partition; greedy outputs do not depend on
the split.

**A10.3 Dataset.** Verification and the exposure audit are applied as registered (A3.1, A3.2, A7.4). The CE7
training dataset is built by the same binary as the teacher pass (A5.3), unpacked sft-step view, `--control
outcome-verified --drop-leaking --min-strength executable`, with block bound 8192 as in A9.2 and
without `--serialise-calls`. The count of records, the number of turns per record and the token
counts are reported as measured. Nothing is trained on it until the CE4 data audit of the first repository has been
resolved; no training parameter is fixed here.

### Amendment 11, 2026-10-19, 20:03 CEST: the teacher's development pass is regenerated

Made before any regenerated trajectory exists and before any CE4 or CE7 training run.

**A11.1 Training data come from whole-conversation trajectories.** The CE1 pass-1 trajectories of A4.1 (sven
d1fd8bac) export a run as one step and are not used to train any adapter of CE4 or CE7; the dataset of A9.2 is
discarded and the CE4 adapter is trained again on regenerated data. The pass-1 verdicts and the
exposure audit do not depend on the export and stand as CE1 results.

**A11.2 Regeneration.** The teacher solves the CE1 development tasks again, one greedy attempt, `--solve-deadline
3600`, concurrency 8 per 27B server, the flags of A2.6 and A4.7, the test-output cap and PATH of each set's
recorded environment, with the splinter build of origin/main 7bd63719 (sven 319a2e05: whole-conversation
trajectories with true turns), after the CE7 development pass of A10.2. A splinter revision
that validates call/result pairing at dataset build time replaces it for the dataset build when it is
pushed (named in the run record, A5.3). The tasks are solved in whole sets, in the order of the
raw pass fraction of pass 1 (ttlcache 34/46, tokenbucket 22/38, caldate 13/25, graphkit 24/51, linediff 18/50,
iniparse 18/61, mdrender 13/43, exprcalc 10/56), so that the sets most likely to yield verified
trajectories come first; this order is a choice of training data and not of evaluation (no arm is compared on
these tasks, A1.1), and no later set is skipped except for lack of lease time, which is reported.

### Amendment 12, 2026-10-19, 23:08 CEST: the recording build of the regenerated passes

Made before any regenerated trajectory exists. It replaces the revisions named in A10.2 and A11.2 and
changes no other parameter.

**A12.1 Binaries.** The regenerated teacher passes (CE7 development, then the CE1 development sets in the
order of A11.2) are recorded by splinter 7bd63719 linked against sven f0cfb82c (one agent step per model
response, distinct call ids) and brain 83e2f44a3, with 27B servers on brain c2d4c9244 (flash prefill
default; flags of A2.6 and A4.7 unchanged) on both cards. The servers' build is one per set (A4.5) and each set's
revisions are recorded in `runs/ce/STATUS.md`. Before a full set is run, two tasks are recorded and their
trajectories are checked to hold one agent step per model response, each call followed by its own result; a
set is not run if they do not. The CE7 development pass already run on sven 319a2e05 (A10.2) keeps its
verdicts as CE7 teacher results and is not training data. The training dataset is built by the splinter
revision named for it in its run record (A5.3).

### Amendment 13, 2026-10-19, 20:46 CEST: the exposure rule judges each probe by its own result

Made before any dataset is built with the changed rule and before any CE4 or CE7 training run. It changes
how the `outside_workspace` rule of A3.1 classifies a tool call and which classes fail the audit; it changes
no verifier, no task, no environment digest and no parameter of the gates.

**A13.1 Defect.** A verifier audit of the CE1 pass-1 records re-computed every recorded exposure hit. The rule
classed an outside-path call by the whole step's output: a step of several parallel calls pooled all their
results, and the shell tool's own `[exit code N]` trailer counted as a result. A `cd /app` that failed with
"No such file or directory" was therefore "reached" whenever a sibling call in its step returned text, or
whenever the command exited non-zero. A3.1 and the note of A4.6 also made every class (reached, denied, absent) fail
the audit.

**A13.2 The rule from now on.** (a) A call is classed by its own result (the result at its position in the step;
a step whose results do not pair one to one with its calls is read as a whole), and the `[exit code N]` line is not
a result. (b) An outside-path call whose own result is only a refusal, a missing path or nothing is a recorded
probe (`outside_probes` in the audit's evidence, kinds `outside_workspace_denied` and `outside_workspace_absent`);
it does not fail the audit and the attempt is not excluded for it. An outside-path call that returned anything else
(`outside_workspace`) fails the audit and the attempt is excluded from training as before. (c) A call that names a
hidden path and whose own result is only a refusal or "no such file" read nothing: it is a recorded probe
(`hidden_path_absent`) and does not fail the audit. (d) A hidden path named in what a tool returned or in a message shown to the
solver, a `content_shingle` of hidden material, and a forbidden string (the state root) remain hard exclusions.

**A13.3 Scope and reporting.** The change applies to every dataset and every verification run from this amendment on.
The CE1 pass-1 yield numbers already published were computed under the old rule and stand as results of that rule:
152 of 370 attempts pass the hidden checks, 100 remain after the exposure audit. Re-evaluated under A13.2 from the stored
trajectories (an independent re-implementation of the rule, `runs/audit/verifier/q4rules.py`; the verified binary is
not rerun on the stored sets), 129 of the 152 remain; the two numbers are reported side by side wherever the yield is
quoted, per set, with the classes of A7.4. No stored verdict is rewritten (an experience and its annotations are
immutable); a re-verification appends new annotations.

### Amendment 14, 2026-10-20, 01:07 CEST: the dataset build, the step rule and the serving build of the corrected CE4 adapter

Made before any dataset is built from the regenerated trajectories, any training run on them and any 8B verdict
on them. It changes no gate and no parameter of A2.7 other than the one stated.

**A14.1 Dataset.** The training set is built by splinter cf3f02d2 (brain 19f0ba146, sven f0cfb82c; it pairs a
result to its call within the step, refuses ambiguous steps and unconserved projections, and runs brain's
record check before storing) from the regenerated sqlglot development pass of A12.1 (set named in the run record),
unpacked `sft-step` view (one record per answer), without `--serialise-calls`, `--control outcome-verified
--drop-leaking --min-strength executable`, block bound 8192 as in A9.2, under the exposure rule of A13.2. The
count of records, of exclusions by reason and of tokens is reported; three records are rendered through the
trainer's path and checked for one result per call and unique call ids before training.

**A14.2 Training.** Recipe A2.7, with the step count the registered rule ceil(2R/8) from the R the trainer
reports (no extra steps); the loss on the training and the monitoring records is read every few steps and the
best monitored evaluation kept. Training that shows memorisation (training loss below half its initial value
while the monitoring loss rises) is reported as such.

**A14.3 Serving build and a pipeline check.** The 8B arms of a check on the adapter are served by brain c373ae464
(adapter in the bank, one resident, default flags, tensor-core tier of A5.1), the same server for both arms; A4.5
applies. The first such run is on the 23-task subset `S1` of A9.6 and is a pipeline check only: it reports
the repetition-stop counts, the tool-call mix and the verified counts, it is not the confirmatory query of
A2.3, and no result of it enters the paper.

### Amendment 15, 2026-10-20, 02:10 CEST: what a task's request may say and name

Made before any sqlglot task is re-admitted and before any further task is generated. It changes how the request of a
workspace task is composed and which candidate faults are refused; it changes no verifier, no gate and no environment
digest of a task already admitted.

**A15.1 Defects found.** A diagnosis of the CE7 sqlglot teacher pass (0 passed of 112; regenerated 0 of 104, runs/audit/regen-diagnosis.md)
found that every one of its 112 tasks is valid (the untouched tree fails the hidden checks, the reference fix passes them), and
that the requests could not be acted on: a failing test was named by its method name alone, so a request read `databricks`,
`time` or `cast` (2.07 words per label in the set against 5.1 in ttlcache), with the instruction that the tests in the tree must keep
passing and listing the test command of the whole suite, which names the hidden modules (every run printed `ModuleNotFoundError`
for them) and, in 28 tasks, withheld the base module that every in-tree dialect test imports.

**A15.2 The command.** The request names the test command with the test modules taken out of the solver's tree left out of
it (for a `unittest` command, each dotted module argument whose file is hidden); any other command is the project's own and is
named whole.

**A15.3 Modules the tree imports.** A candidate is refused (reason `hidden_imported_by_tree`) when the tests that stay in the solver's
tree do not all run there without the hidden files: a module an in-tree test imports is never withheld, because withholding it breaks
the tree and leaving it would show the tests.

**A15.4 Symptoms a solver can act on.** A failing test whose label (its short docstring, else its name in words) has fewer than 3
words is not listed by that label alone. It is rendered from what its failed check observed, and from nothing else: the exception's own
message when it is an equality (`first != second`: the two values, each on one line and at most 300 characters) or an error the code under
test raised (`Type: message`). Never shown: any traceback frame (file paths and the lines of the test that ran, which are the
hidden test's code), the failure's diff lines, the test's file path and the call of the assertion. A failure that is neither shape
(a bare assertion, a truthiness check, an overlong value) cannot be rendered, and the candidate is refused with the counted reason
`symptom_not_actionable`. A label of 3 or more words is unchanged.

**A15.5 The line between behaviour and code.** The shown values are the one disclosure a task makes of its hidden tests: what the code
returned and what it was required to return. They are the literals of the test and so can repeat a run of its text. The admission check
masks them before it looks for a quotation of the hidden tests (a request still may not quote 8 words of the hidden files beyond the
shown values, may not name a hidden file and may not quote a changed line), and the exposure audit counts the task's own request as
text the solver may see, so that a shown value is not a content shingle while a run of the hidden test's text found anywhere else in
what the solver saw still fails the audit.

**A15.6 Scope.** The amendment applies to tasks admitted from now on. The CE7 sqlglot tasks are re-admitted with this generator in the
next lease; the recorded CE7 pass-1 sets stay as recorded and are reported as 0 of 112 passed, with the cause of A15.1. The CE1
repositories are not affected: the 370 task records of the CE1 pass-1 sets were read offline against the rules: the shortest label is 4
words, every command is `python3 -m unittest` (it names no module), so none of the 370 requests would change or be refused.

### Amendment 16, 2026-10-20, 03:40 CEST: the recording build changes (compaction, per-call bound), servers on brain 3c5bf0b1a

Made before any set is recorded with the new build. It replaces the recording build of A12.1 for every
regenerated set started after the two-task check below and changes no other parameter.

**A16.1 Recording build.** Splinter 34db203f linked against sven ba4f08c7 and brain 87dfd94b0. Against the build
of A12.1 the sven revisions change what a recorded run does in three ways: a compaction of the conversation keeps
the task message, the compaction trigger uses the serving layer's own token count, and a shell call is bounded by
`tools.max_timeout_secs` (300 s by default), so no call runs unbounded. Every regenerated set started after the
check records with this build; a set already running finishes on the build it started with (A4.5) and its
build is named in its run record. The two-task check of A12.1 is repeated on this build before its first set.

**A16.2 Servers.** The 27B teacher servers run brain 3c5bf0b1a (batched decode on tensor cores, flash prefill
default; flags of A2.6 and A4.7 unchanged), switched only between two sets (A4.5); the set boundaries and the
build of each set are recorded in `runs/ce/STATUS.md`.

### Amendment 17, 2026-10-20, 03:46 CEST: a learning-progress curriculum arm of GEN1 (Gate D), an ablation

Made before any generation of any arm below has run. It adds one arm to GEN1 of `acquisition.md` and changes
nothing else: not the E3 plan, the exam, the acceptance criterion of GEN1 (at least three accepted generations,
BWT of the full retention matrix not below -0.05), the limits of a generation, the gate, the verifiers or the
replay arms. It is an ablation of how the solve and teach budget is spent. It is not a headline, and no result
of it is part of the paper's claim that the loop works.

**A17.1 What is added.** `splinter practice` and `splinter generation run` take `--curriculum
frontier|learning-progress|gain-ranked`. The new `schedule` stage runs between `reserve` and `solve` and puts the
tasks left after the exam is reserved in an order; the solve and teach stages go through them in that order and stop
at `--budget`; the pass@k frontier classification that follows is unchanged. A task family is the file a fault was
seeded in. The competence history of a family is its verified pass rate (decided attempts of the policy
closed-book, not the teacher's and not a critic's) under each release the policy has been, oldest first, read from
the experience store, with the device-busy seconds of those attempts from their attempt reports.
(a) Learning progress of a family is the absolute difference between the mean pass rate over the newest W releases
and over the W before them (W = `--lp-window`, fewer when the history is short, half of it, at least one); it is 0
with fewer than two measured releases. (b) The weight of a family is its progress divided by its GPU seconds per
attempt relative to the mean of the families that recorded device time (a family that recorded none is priced at
that mean, never at zero; the ratio is floored at 0.05), plus a novelty bonus N / sqrt(1 + attempts), normalised
over families and mixed with the uniform distribution: weight = (1 - E) * normalised + E / n. (c)
`learning-progress` draws every family once, without replacement and in proportion to its weight, from
`--curriculum-seed`, the order depending on the seed and the family names alone; `gain-ranked` orders families by
weight, heaviest first, ties by name, with no draw. Tasks of one family stay together in their admission order.
(d) The parameters are fixed here and are not tuned on any result: W = 2, N = 0.1, E = 0.1 (the defaults of
`--lp-window`, `--lp-novelty`, `--lp-explore`). Every family keeps at least E / n of the draw, so none starves.
It follows the bandit framing of Self-Evolving Curriculum for LLM Reasoning (2025, arXiv 2505.14970), each task
family an arm of a non-stationary bandit, but its reward is the measured change in verified pass rate per GPU
second, where that paper's reward is the absolute policy-gradient advantage and its curriculum policy is learned;
nothing here is learned, and no result of that paper is claimed. The implementation is the commit named
in `data/commits.json` as `acq-lp`; the arms are run with a splinter binary built from it or a later one, one
binary and one brain binary for all arms.

**A17.2 Arms.** On the E3 plan (`tasks/e3-plan.toml`), three generations, the same `--exam-families`,
`--exam-seed`, training `--seed`, `--k`, teacher, `--replay uniform` and `--budget` in every arm and generation:
the frontier arm (`--curriculum frontier`, the admission order; the existing behaviour), the learning-progress arm
(`--curriculum learning-progress`, one `--curriculum-seed` per replicate) and the gain-ranked arm
(`--curriculum gain-ranked`). The arms start from the same base on fresh state roots. The budget is fixed before the
first generation of any arm as 60 percent of the wall-clock time of the solve and teach stages of an
unbudgeted frontier-arm generation 1, measured once and written to `runs/acq/lp-budget.txt`; a budget that does
not truncate any arm's solve is the case where the order cannot matter, and the report then states it (the
`solve.stopped` field of every generation). Replicates are three draws of the training seed and, for the
learning-progress arm, the curriculum seed, drawn once from `/dev/urandom` and written to `runs/acq/lp-seeds.txt`
before the first generation; a frontier-arm replicate and a learning-progress replicate with the same index share
the training seed. If the lease allows fewer than three replicates, the comparison is a pilot of the replicates
that ran and carries no claim.

**A17.3 Outcomes.** For each arm and replicate: the number of accepted generations and the BWT of
`splinter generation matrix` (the Gate D metrics, unchanged); and the verified held-out gain per GPU hour, the sum
over the three generations of `assessment.new_gain` (the candidate's gain over the champion on that
generation's own exam, accepted or not) divided by the GPU hours the arm held its card for (the wall-clock of its
`card`-reserved commands, summed, which includes solving, teaching, training and every measurement of the panel).
Secondary, each generation: the report's `yield` (tasks kept per GPU hour of the solve and teach attempts, with the
count of attempts that recorded no device time) and the `schedule.informative` count.

**A17.4 What counts.** The learning-progress arm is reported as ahead of the frontier arm only if, in at least two
of three replicates, its verified held-out gain per GPU hour is greater than the paired frontier replicate's, its
accepted generations are not fewer, and its BWT is not below -0.05. Otherwise the result is reported as no
demonstrated advantage, whatever the means. With three replicates no interval is claimed; the three paired
differences are listed. The gain-ranked arm is reported on the same outcomes and is not part of the decision. A
generation that is Stopped or whose exam could not be measured is reported as such and counts as a gain of 0 for
that replicate only in the denominator-bearing sum, with the reason beside it.

**A17.5 A stated limit on what this can show.** A family has history only once a release has attempted it. In the
E3 plan each generation trains on repositories no earlier generation touched, so at generation N every family is
new and `schedule.informative` is 0: the learning-progress order is then the novelty-weighted draw, which is a
seeded shuffle of the families, and `gain-ranked` is the order by novelty, which is the admission order up to
ties. A result in which `schedule.informative` is 0 in every generation of an arm is reported as a comparison of
a seeded family shuffle against the admission order under a truncating budget, and not as evidence about learning
progress. The signal can only act where a family is attempted under two releases, for instance a second
`generation run` over the same repository at a later revision; no such arm is registered here.


### Amendment 18, 2026-10-20, 03:58 CEST: a learning-progress replay arm of GEN1 (Gate D), an ablation

Made before any generation of any arm of Amendments 17 or 18 has run. It adds replay arms to GEN1 and changes nothing
else: not the E3 plan, the exam, the limits of a generation, the gate, the verifiers or the acceptance criterion of
GEN1. Amendment 17 stays as registered, and its task-order arms are expected to be uninformative on E3 for the reason
A17.5 gives (no family has history at the generation that first meets it); the sentence of A17.5 that no arm is
registered where a history exists is superseded by this amendment. This is an ablation, not a headline.

**A18.1 What is added.** `splinter generation run --replay learning-progress` (`ReplayChoice::LearningProgress`,
recorded in the ledger as `replay_policy` of kind `learning_progress` with its weights). Every generation measures its
champion on the whole protected panel, so each earlier generation's exam (a panel column) has a pass rate under every
release that has been the champion. The history of a column is one round per release measured on it, oldest first,
from the ledger's champion measurements and the current generation's own (two measurements of one release are one
round, the newer; a cell is counted when its task is in the current panel and it has a verdict; a round's attempts are
the cells counted and its passes those graded right). The weight of a column is the learning-progress weight of
Amendment 17 (A17.1 a, b, with W = 2, N = 0.1, E = 0.1, the same flags, fixed here and not tuned): the absolute
difference of the mean pass rates of the newest and the preceding window of releases, a novelty bonus that shrinks with
the cells counted, mixed with the uniform distribution so that no generation gets less than E / n of the replay. A
panel measurement records no device time, so the cost divisor is neutral (every column priced at the mean). Each
earlier release's training families take the weight of that release's exam (an exam's families are reserved apart from
the families trained on, so the weight is carried across, as the failure-targeted replay carries the champion's
failure rate), and the replay of earlier records is allocated over them in proportion exactly as for the
failure-targeted replay: the fraction, seed and per-record draw are unchanged. Generation 1 has nothing to replay.
Expected behaviour, stated so it is not read as a finding: the first change in a column's pass rate is the rise from the
generation that trained on it, so a recently learned generation weighs more, and a generation being forgotten weighs
more by the absolute value; a column measured on few tasks moves by noise, and the exam sizes (cells per column) are
reported with every weight.

**A18.2 Arms.** On the E3 plan, three generations, with the task order fixed to the frontier order in every arm
(`--curriculum frontier`) so that the arms differ in replay alone, and the same `--exam-families`, `--exam-seed`,
training `--seed`, `--k`, teacher and `--budget` as Amendment 17 A17.2: `--replay learning-progress`, `--replay uniform`
(the arm that is also Amendment 17's frontier arm, run once and used by both) and `--replay failure-targeted`. Fresh
state roots, the same base, three replicates with the training seeds of A17.2 (the learning-progress replay needs no
seed of its own: its weights are a function of the measurements and the record draw uses the training seed). If the
lease allows fewer than three replicates the comparison is a pilot and carries no claim.

**A18.3 Outcomes.** Per arm and replicate, as A17.3: accepted generations, the BWT of `splinter generation matrix`, and
the verified held-out gain per GPU hour (the sum over the three generations of `assessment.new_gain` over the
card-reserved wall hours of the arm's commands). Reported beside them: the recorded replay weights and the cells per
column of every generation, and the average final performance of the full matrix.

**A18.4 What counts.** The learning-progress replay is reported as ahead of a comparator (`uniform`, and separately
`failure-targeted`; two decisions, no correction for there being two) only if, in at least two of three paired
replicates, its BWT is not lower and its held-out gain per GPU hour is greater than the comparator's, its accepted
generations are not fewer, and its BWT is not below -0.05. Otherwise the result is reported as no demonstrated
advantage. With three replicates no interval is claimed; the paired differences are listed. A generation that is
Stopped or whose exam could not be measured is reported as such, with the reason.

### Amendment 19, 2026-10-20, 07:32 CEST: CE8, the 27B trained on its own verified trajectories

Made before any CE8 training run and before any CE8 qualification query. It fixes what A1.4 leaves open (the base the adapter trains on, the data, the view,
the step and stopping rules, how it is served, the evaluation) and changes nothing else.

**A19.1 Data.** Only trajectories of the 27B teacher (the Q8_0 GGUF server of A2.6, no adapter) on development-partition tasks, recorded by splinter 34db203f
with sven ba4f08c7 (one agent step per model response, A12.1, A16.1) against brain servers that give every tool call its own id. The experience sets are the
regenerated CE1 sets of A11.2, each verified under the exposure rule of A13.2: ttlcache blake3:1f4be47c753a198b747aa1d79a2eb6772890351fed0bccc058b48a5f9b94a517; tokenbucket blake3:6b19c95138ec69ba4ef7a4e08fb1bae9eb52a38d626e04f405249f5e78ec7143; graphkit blake3:46e9cb04b87fa49b8cd8fe70d530bad563a94fcf466485e89bd2975eee8e6c65; linediff blake3:93103c07039095ad827163e931e110277995c926533b222589c32280899742c7; mdrender blake3:afc508246787415cb4204c3501a3edffd4b248affc20c3b9afcc52be3c1617ea; and the regenerated iniparse and caldate sets if both are
stored and verified when the dataset is built (their ids go into the run record, and no set stored later is added). The dataset is built by a splinter that
contains cf3f02d2 and pins brain cd8df8266: `dataset build --view sft-step --pack --drop-leaking --min-strength executable --max-block 16384 --control
outcome-verified`, never `--serialise-calls`; a trajectory whose context was compacted is dropped by the view and counted. The count of records, the exclusions by
reason (failed, over block, compacted, exposed) and the per-record pairing check of A14.1 are reported.

**A19.2 Base, precision, recipe.** `splinter train --from local:Qwen/Qwen3.8-27B-FP8 --base-precision nf4`, rank 16, alpha twice the rank, learning rate 2e-4, seed 1,
all targetable linears, monitoring share 0.2 held out by whole family, held-out scoring on, and the best monitored evaluation kept (`--keep-evaluations`). A record
in the packed view is a trajectory, so B = 2 records per step and the steps are ceil(2R/B) for the R the trainer reports (A2.7, no extra steps). The monitoring loss
is read eight times over the budget (`--eval-every ceil(steps/8)`) and training stops after two evaluations without improvement (`--patience 2`). Memorisation
guard: a run whose final training loss is below half its first-step value while its monitoring loss is above its minimum is reported as memorised and its adapter
is not evaluated. Fewer than 40 verified trajectories means no run (A2.7). The step time, row tokens per second and peak VRAM are recorded (one `card --card 1 --gb
66` job) and the loss curve is reported.

**A19.3 Serving.** The adapter is installed into the adapter bank of the Q8_0 GGUF teacher server (all targetable linears, rank cap 16, brain build named in the run
record), so base and adapter share one resident base. The adapter is trained against the NF4 weights of the FP8 checkpoint and applied to the Q8_0 weights; the
numerics differ, and the served-against-in-process agreement of G8 is the check, not an assumption.

**A19.4 Evaluation.** The base 27B and the 27B with the adapter solve the qualification halves of A8.2 (HALF_A, then HALF_B), one arm at a time on one server and one
splinter binary, under the same solve protocol for both arms. That protocol is the one of the solve-protocol amendment (recommended sampling, bounded thinking, the
coding-agent prompt, repetition feedback) and the arms are not started until that amendment has been pushed; if it has not been, the evaluation waits and no other
protocol is substituted. The primary test is that of A2.3 on family outcomes; the comparison keeps the pilot label of A2.3 until its sizing is met.

### Amendment 20, 2026-10-20, 08:56 CEST: the solve protocol of the arms: recommended sampling, bounded thinking, a coding-agent prompt, repetition feedback

Made before any arm is run under it. It applies to every evaluation arm (the untrained 8B and every adapter, and
the 27B where it is an arm) started after the revisions named in A20.5 are in use, and replaces A2.2's greedy
one-attempt decoding and "thinking off" for them. Results of arms run before it (CE1 pass 1, the greedy S1
pilots) stand as results of the protocol they ran under and are never pooled with arms of this one. It
does not change the tasks, partitions, verifiers, the exposure rule (A13.2), the training recipe (A2.7, A14.2) or
the teacher passes recorded for training data (A12, A16), except where A20.7 says.

**A20.1 Sampling.** Each arm samples with the model family's recommended settings instead of greedy decoding:
with thinking off, temperature 0.7, top-p 0.8, top-k 20; with thinking on, temperature 0.6, top-p 0.95,
top-k 20. The primary measure of an arm is pass@1 averaged over k registered seeds (k and the seed list
k = 3, seeds 16022413115216019911, 3017204225152849161, 16861570844197835729, each drawn from /dev/urandom as a u64 on 2026-10-20 (command in runs/ce/protocol-seeds.txt), no fixed literal; a family-clustered sign test on the per-family mean of the three, fixed here before any arm runs; the seed of an attempt is part of its record).
A per-task outcome is the fraction of the k attempts verified, a family's outcome is the mean over its tasks as in
A2.5, and the primary test and its uncertainty are computed on those family outcomes. The request carries the seed (splinter's served_config) and brain echoes the seed it used, which is recorded; measured on brain 3c5bf0b1a, the same seed twice gave the same 8-token reply. A2.2's reason for greedy decoding (no seedable sampling) no longer holds
when brain f2eab2a43 is the server build; if a server cannot seed, the arm is not run.

**A20.2 Thinking, bounded.** Thinking is allowed, with a server-enforced budget of 1024 thinking
tokens per model response (splinter `--thinking --thinking-budget 1024`, sent as brain's `thinking_budget`; a response that exhausts it is closed by the server and
continues in the answer, and the number of such responses is reported per arm). The budget is the same for both arms of a
comparison; changing it is an amendment. Thinking text is part of the recorded trajectory and is not training data
for an arm that did not produce it.

**A20.3 The coding-agent system prompt.** `solve` gives the solver a coding-agent system prompt in place of the
generic chat prompt ("answer directly and concisely, using a tool when one is offered"). The prompt is part of the splinter
revision named in A20.5 (`crates/core/src/prompt.rs`, id `coding-agent-1`, recorded with its digest in every trajectory's solve profile) and is the same for every solver, base or adapter. Its content, fixed here:
  You are a software engineer working in a git repository through tools. You are given a report of behaviour
  that is wrong. Work in this order. First look: list and search the repository and read the code and the tests that
  concern the report before you change anything. Then reproduce the symptom from the report with a short command
  or test and read what it prints. Then change the code (edit the source, not the tests the report relies on).
  Then run the repository's tests and the reproduction again and read the results; if they fail, read the failure
  and fix the cause. Never repeat a call whose result you have already seen: if you are about to run the same call again, do
  something different (read other code, change the call, or change the code). Stop when the behaviour is
  fixed and the tests pass, and say in one or two sentences what you changed.
Changing the prompt is an amendment; the arms and the training records of a given comparison use one prompt text (a record
trained under another prompt is rendered again under this one before training).

**A20.4 Repetition feedback.** The harness (sven d5e68b75) appends a note to the result of a call that is identical to an earlier call of the run and returns the same result,
saying the result is unchanged and a different action is needed (`agent.note_repeated_results`, switched on by `solve`); a shell call is bounded by
the ceiling of 1800 s, and a stop from outside carries its label. The run still stops at the repetition bound the build records in the solve profile of each trajectory. The repetition stop and its count remain
reported per arm; a stop remains a failure of the attempt.

**A20.5 Builds and serving.** Splinter 68ba6a8c4 linked against sven d5e68b75 (>= 8b10610c, B10: a run that ends in the
middle of a thought concludes `stopped_without_conclusion`, not `success`, which changes the follow-up rounds and is
registered solve behaviour) and brain cd8df8266. The 8B is served with `--qwen-ctx 40960` (A2.6's 32768 replaced), all other A2.6 / A5.1
flags unchanged. The 27B teacher server flags are those of A4.7 on brain f2eab2a43; the 8B server is brain f2eab2a43 as well. Both arms of a comparison use one server
instance, one splinter binary and one prompt, as before (A2.2, A4.5).

**A20.6 Variants and the primary arm.** Two registered variants, run for both arms of a comparison: thinking off (temperature 0.7, top-p 0.8, top-k 20) is the
primary arm, and thinking on with a 1024-token budget (temperature 0.6, top-p 0.95, top-k 20) is a secondary arm reported beside it; neither replaces the other
and the primary test is on the primary arm. The runs use the arm script `runs/verifier/arm-seeds.sh` with the seeds of `runs/ce/protocol-seeds.txt`
(`--sampling-seed`), the 8B base arm first on HALF_A of A8.2. A training record is rendered under the same system prompt
(`dataset build --system-prompt`), so an adapter is trained again under this protocol before its arm is run; an adapter trained under another prompt is not an arm of it.

**A20.7 Recording passes.** The teacher passes recorded for training data (A12.1, A16.1) keep their greedy settings and
their build unless a later amendment names this protocol for them; sets recorded under the sven revision of A20.5 are
named with it in their run record (A4.5), and the `stopped_without_conclusion` count is reported per set.

**A20.8 What is reported.** Per arm: pass@1 mean over the seeds with its family-clustered interval, the verified
count per seed, repetition stops, thinking-budget hits, tool-call mix and mean agent steps; for a comparison, the primary test of
A2.3 on the family outcomes. Pilot labelling (A2.3) is unchanged.

### Amendment 21, 2026-10-20, 10:17 CEST: the learning rate of the CE8 training

Made before any CE8 training run and before any CE8 outcome exists. It changes the one value of A19.2 that a smoke measured to be unstable and nothing else of A19.

**A21.1 Measurement.** A 10-step NF4 training of the 27B on the same 16 records with the same seed, at five learning rates (`runs/b27/lrsweep/<lr>.err`, step losses): at 1e-9, a control in which the adapter does not move,
the loss at step 10 is 0.744 (record-to-record variance, 0.473 at step 1); at 2e-5 it is 0.741; at 5e-5 0.750; at 1e-4 0.826; at 2e-4, the value of A2.7 and A19.2, 4.786. The loss at 2e-4 diverges; the
control shows that the rise to about 0.74 at the lower rates is the variance of the records and not training. No rate was chosen for its result on any task.

**A21.2 Value.** The CE8 learning rate is 5e-5 (A2.7's 2e-4 is the registered value for the 8B recipe and is unchanged there). Every other setting of A19.2 stands: the recipe, the packed
view, B = 2 records per step with ceil(2R/B) steps, the monitoring split by whole family, `--keep-evaluations`, patience 2, the memorisation guard. The dataset is `blake3:d246dbb54ca3c8a3eb7b8755d5c7590b8e8a0779747a9c0cc1cab238062a2594` (80 packed records
from the seven sets of A19.1, built by splinter c70ab627 linked against brain cd8df8266 and sven f0cfb82c, a local build; excluded: failed 196, compacted 27, over block 12).

### Amendment 22, 2026-10-21, 01:30 CEST: CE8 on the bf16 base and corrected records, and a teacher pass under the solve protocol

Made before any CE8 training run, before any CE8 outcome and before any trajectory of the pass of A22.3 exists.

**A22.1 Base precision.** CE8 trains its LoRA on the bf16 base of the FP8 checkpoint in place of A19.2's NF4 base (native precision where the card holds it).
Measurement, before this amendment: a 10-step bf16 training on the 16 smoke records at lr 5e-5 tracked an lr 1e-9 control within 0.03 at every step (0.480 -> 0.750 against 0.480 -> 0.755; record-to-record variance, no divergence),
8.5 s per step at block 8938, 73.6 GB peak on one card (`runs/b27/bf16check/`). Every other value of A19.2 and A21 stands.

**A22.2 The records.** The CE8 dataset is built again from the seven sets of A19.1 with the first splinter revision of origin/main that contains three corrections, named with its hash in the run record (A5.3):
(a) the exposure audit pairs parallel tool results with their calls by call id, not by position, and judges a path by what the call did with it, so a traceback line, a refused scratch write or a non-existent invented path is not exposure (a read-only re-check of the 35 passes A13 excluded: 23 of them looked at nothing hidden; genuine probing of hidden paths stays excluded);
(b) a record is rendered under the system prompt its solve recorded, the host's login name and the harness's scratch paths are removed from it, and the tail of identical calls that ended in a repetition stop is not a training target;
(c) nothing else of A14 or A19 changes. The dataset of A21.2 (`blake3:d246dbb5...`) is not trained on.

**A22.3 A teacher pass under the solve protocol.** A20.7 names the protocol for one new teacher pass: the 27B solves the eight CE1 development sets again with the secondary variant of A20.6 (thinking on with a 1024-token budget, temperature 0.6, top-p 0.95, top-k 20),
the first seed of `runs/ce/protocol-seeds.txt`, one attempt per task, `--solve-deadline 3600`, concurrency 8 on one server, the splinter revision of A22.2, served by the brain build that closes a budgeted thought with the model family's early-stopping sentence (brain 2390f4207; binary `bin-brain-0cc4bf64c`, the same change before its rebase) at a context of 65536, in the set order of A11.2.
Its passes, admitted and projected as in A22.2, form a second CE8 dataset (CE8-T), trained with the recipe of A22.1/A22.2 into a second candidate. Why thinking: an unregistered pilot on 46 HALF_A tasks (two per family, pick seed 1400606320) gave thinking off 23/44, 22/46, 22/46 and thinking 1024 26/46, 29/46 on seeds 1-2; it is labelled a pilot and is no result of this study.
How CE8 and CE8-T are evaluated is registered by a later amendment before any of their evaluation attempts runs; neither candidate is evaluated on a CE1 development task.

### Amendment 23, 2026-10-21, 13:05 CEST: the CE8 training after the gradient fix: learning rate, stopping rules, invalid runs

Made before any valid CE8 training run and before any CE8 or CE8-T evaluation attempt. It corrects the basis of A21 and A22.1 and restates A19.2's stopping rules; no value is chosen from any task outcome.

**A23.1 The defect and what it invalidates.** From brain f3c83d307 (2026-10-19) to 8ad0ab962 the 27B's chunked MLP backward added into a buffer that was never cleared: the LoRA gradient norm read 1.3e5-2.8e5 instead of about 1, and clipping scaled the true gradient by about 4e-6. Every 27B training before 8ad0ab962 ran on that gradient. Therefore:
(a) A21.1's learning-rate sweep and A22.1's bf16 check measured a broken gradient and are no basis for any value; their numbers stand only as a record of the defect;
(b) the CE8 run `ce8/logs/train-a22-bf16-r2.*` (lr 5e-5) trained on the broken gradient and is invalid; it also ran with splinter's default stopping values (evaluation every 5 steps, patience 4, 80 steps) in place of A19.2's;
(c) the run `ce8/logs/invalid-lr2e-4-train-a22-bf16.*` used lr 2e-4 by a script error and is invalid.
No candidate of these runs is evaluated or reported as a result.

**A23.2 Diagnostic and the learning rate.** A 40-step bf16 training of the 27B on the 80 records of the A22.2 dataset (64 trained, 16 monitored; rank 16, alpha 32, 2 records per step, seed 1), at the two rates registered so far (5e-5 of A21.2, 2e-4 of A2.7/A19.2), on brain f6c52a39b (`runs/b27/ce8diag/summary.md`): grad-norm median/p90/max 1.35/5.3/13.7 at 5e-5 and 1.18/3.2/8.5 at 2e-4; training loss 0.745 -> 0.127 and 0.745 -> 0.119; neither diverges. Rule fixed before reading it: keep the registered rate unless the diagnostic shows it unstable. Both are stable, so **A21.2's 5e-5 stands**. The monitored losses (0.095 and 0.107 at step 40) were not used to choose and are not a result.

**A23.3 Build.** CE8 trains with splinter 3330c206c (the revision of A22.2) linked against brain 77e510fbf (contains 8ad0ab962, the best-copy spill 741b96c5a and f6c52a39b, which computes a one-sequence batch only up to its last supervised position: loss and gradients equal those of the full row by test) and sven 3e1f6b4d, a local build named with these hashes in the run record.

**A23.4 Stopping rules, passed explicitly.** For the R records the trainer reports and B = 2: steps ceil(2R/B), `--eval-every` ceil(steps/8), `--patience 2`, `--keep-evaluations`. For the A22.2 dataset (`blake3:75b4734dc80d0dec909549aa55b8c3f677837c86d3a0b7cb998c7787e5035cb0`, R = 65) that is 65 steps with an evaluation every 9 steps. The run script passes these values and writes them into the run record. Every other value of A19.2, A21.2 and A22 stands, including the memorisation guard; CE8-T (A22.3) is trained the same way with its own R.

## Provenance

The common definitions restate, without change of substance, the protocol fixed on
2026-10-13 for a separate draft paper on tuning a local model to a repository (units, arms,
primary test, retention, exclusions and controls); that draft contained no measured run and was
folded into this paper, and its directory removed, on 2026-10-16 before any run.
