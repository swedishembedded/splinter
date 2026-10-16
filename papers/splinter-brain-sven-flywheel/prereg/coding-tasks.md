<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of the coding-task experiments CE1, CE4, CE6 and CE7

Written and committed on 2026-10-16, 18:04 CEST, before any run of these experiments
exists. It is separate from `protocol.md`, which registered the first campaign (E1-E3
there) and is not amended by this file. Experiment names are prefixed `CE` here because
`protocol.md` already uses E1-E4 for the first campaign; the master plan's names are in
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

(none)

## Provenance

The common definitions restate, without change of substance, the protocol fixed on
2026-10-13 for a separate draft paper on tuning a local model to a repository (units, arms,
primary test, retention, exclusions and controls); that draft contained no measured run and was
folded into this paper, and its directory removed, on 2026-10-16 before any run.
