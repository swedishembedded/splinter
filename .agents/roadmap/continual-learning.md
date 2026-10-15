# Fast continuous learning roadmap

**Status: planned. Nothing below has been started.** Update this file in the same commit
that finishes or changes an item. The grounded analysis behind it (literature check, tiers,
latency budget, evaluation protocol) is the planning section "Fast continuous learning" of
the cross-project plan; this file is the Splinter share of it. Items for brain
(`continual-state.md`) and sven (`experience-memory.md`) are named by id.

## Goal

Learn from experience at four speeds with one promotion policy, and prove it with a measured
protocol rather than a story: in-context (seconds), memory and skills (seconds after a
verifier finishes), adapters (minutes), generations (nightly). Splinter decides which
experience qualifies, when something graduates to the next speed, and when it is retired,
forgotten or rolled back. sven records and retrieves; brain computes.

## What exists (read from the code, not from claims)

- Evidence: verdict strength ordering, claim rulings (admitted with supersession,
  reinforced, forgotten, refused), the agent's words never counting as evidence.
- Gate: paired one-sided sign test, per-release retention bound, frozen anchor bound, serve
  check; consolidation gate that asks for no gain but loses nothing retained.
- Generations: `absorb` retrains an adapter on the live claims, `consolidate` takes them into
  the weights, `plan_rebuild` is the only honest forgetting for a consolidated claim.
- Curriculum: mastery per concept per release, queue, quotas.
- Apparatus: `samples/generations` and `samples/absorb-story` with criteria and predictions
  declared before any run. This review found no recorded result of either; run them before
  claiming anything about consolidation.
- Not present: Forgetting as defined by Chaudhry et al., true Forward Transfer against the
  base, learning-gain curves, persistence-after-restart, poisoned-experience and rollback
  byte-equality tests, an outcome-linked memory policy.

## Left to do

- [ ] CL-P1 (M) A `continual` module in the eval crate and `splinter eval matrix`.
  - A matrix A[k][i] of pass rates with explicit unmeasured cells (n, ties, withheld,
    environment failures), pure functions for ACC, BWT, Forgetting (maximum taken from the
    step at which suite i was first learned), Forward Transfer against the untouched base
    (not against a fresh-trained control), learning gain and the curve G(n) after n
    experiences. Golden tests with hand-computed matrices, including a case where the score
    peaked after the learning step so BWT and Forgetting differ.
  - BWT is brain's definition (one implementation); the test pins equality with brain's on
    shared fixtures. Every report prints the one-answer baseline and the shortcut ceiling.
  - Acceptance: `make test check`; `splinter eval matrix` fills A from existing `eval` runs
    per release and reports unmeasured cells as such.
- [ ] CL-P2 (M) Qualification and withholding controller (rules R0, R1, R2, D1).
  - R0 admit a unit as active-eligible only if the outcome is a pass with strength at least
    executable (formal for factual probes), provenance trusted, no contradiction (a
    contradiction takes the supersession path). Weaker or untrusted units are hints only.
  - R1 randomized withholding with probability 0.25, logged; a unit becomes active when over
    at least 5 non-tied families the used arm wins a one-sided exact sign test at 0.05.
  - R2 queue a unit for adapter training when active and retrieved at least 10 times across
    at least 3 families with injection cost over a budget, or after 2 reinforced user
    corrections, or when it targets the weakest family in the mastery report. Trainable
    records are executable or formal passes, absent from the sealed exam, with teacher-only
    information excluded.
  - D1 retire a unit when its success over its last 10 uses falls more than 0.10 below its
    withheld success, when a newer executable verdict contradicts it, or when its environment
    fingerprint changes.
  - Thresholds are pre-registered configuration, never tuned on the sealed exam.
  - Acceptance: unit tests per rule with the boundary values (4 and 5 non-tied families;
    0.10 harm exactly and just over); a poisoned fixture (agent-words-only evidence, untrusted
    provenance, judged-only verdict) that never reaches active. Depends on CL-S1, CL-S2,
    CL-S3. `make test check`.
- [ ] CL-P3 (M) Post-release monitor and rollback trigger (D4).
  - After a release, re-evaluate on freshly drawn tasks (never training or sealed exam
    families). A drop of more than 0.05 on any earlier release's suite, or more than 0.02 on
    the anchor suite, or later identification of a poisoned training record, moves the alias
    back along the lineage (compare-and-set) and quarantines that release's training records.
  - Acceptance: a test with a deliberately harmful candidate passed through a test-only gate
    with the retention bound removed: the alias returns, and the probe-set responses are
    byte-identical before and after (depends on brain CL-B5). `make test test/e2e/brain-sven`.
- [ ] CL-P4 (S) Fixtures for persistence, poison, rollback and forget.
  - T-persist (restart with only on-disk state; digests identical; refuse to serve on a
    mismatch), T-poison (10 percent poisoned experiences in four forms; zero reach active or a
    training set; one-sided 95 percent lower bound on the clean-suite change above -0.05;
    typed refusal reasons), T-rollback (CL-P3), T-forget (exact at tier 1, exact after
    retrain at the adapter tier, flagged held-in-base at the weights tier until
    `plan_rebuild`).
  - Acceptance: each is a deterministic test with a stated pass criterion. `make test`.
- [ ] CL-P5 (L) Task families for the cumulative-skill experiment.
  - The frozen catalog holds one family. Add at least four more (mutation-seeded from
    repository history), each with witness, negatives and a verifier the policy cannot reach,
    so that 4 batches of at least 8 exam families can be drawn without reuse.
  - Acceptance: each family passes the catalog's own audit (witness solves, negatives
    rejected, answer not readable from the workspace). `make test check`.
- [ ] CL-P6 (M) Experiment B driver and pre-registration (after CL-P2, CL-P5).
  - Arms: no memory, retrieval-only, skill memory, adapter, combined; placebo memory;
    adapter on all trajectories regardless of verdict; adapter without replay.
  - Criteria fixed before the first run: best memory or adapter arm beats no memory by the
    one-sided exact sign test at 0.05 over at least 5 non-tied exam families; skill memory
    cuts median tool calls per solved task by at least 20 percent on tasks both solve; the
    placebo is not better than no memory beyond noise; the retention lower bound on earlier
    batches is above -0.05; CL-P4 tests pass. Failing the first is reported as "not shown".
  - Acceptance: the pre-registration is committed before any measured row exists; results
    land in the results table with unmeasured arms printed as not run.
- [ ] CL-P7 (S) Tier-1 baseline driver on the invented-facts apparatus.
  - Inference only, on `samples/generations` facts and sealed probes: closed book, facts
    retrieved by BM25 into the prompt in stream order, and placebo retrieval, on the 8B and
    4B policies. Small enough to run in minutes of device time; it measures the tier-1
    effect on a controlled stream and prices the harness.
  - Acceptance: a table of recall by arm with the one-answer baseline; criteria declared
    before the run. `make test` for the driver's grading code.

## Order and dependencies

CL-P7 and CL-P1 first (no new capability needed). CL-P5 in parallel. CL-P2 needs sven
CL-S1 to CL-S3. CL-P3 needs brain CL-B5. CL-P6 last. Run `samples/generations` and price one
`absorb` update before scheduling it on a shared device; abort an update that exceeds the
budget declared in advance.

## Out of scope

Training a neural-memory architecture (HOPE, Titans, TTT variants) from scratch, and
world-model learning: neither has a measured path on the hardware and models this project
runs. Revisit only if brain's persisted-state work shows a measurable benefit.
