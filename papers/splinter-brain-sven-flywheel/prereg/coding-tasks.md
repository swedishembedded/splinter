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

## Provenance

The common definitions restate, without change of substance, the protocol fixed on
2026-10-13 for a separate draft paper on tuning a local model to a repository (units, arms,
primary test, retention, exclusions and controls); that draft contained no measured run and was
folded into this paper, and its directory removed, on 2026-10-16 before any run.
