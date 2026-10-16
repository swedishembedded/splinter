# research-to-stack - splinter share

**Status: planned. Nothing below has been started.** Update this file in the same commit that
finishes or changes an item. Evidence and citations are in the cross-project research note;
item ids U1-U15 refer to it. Brain items RS-B1..B5 and sven items RS-S1..S5 are named by id.
Order: U4, U1, U2, U3 first (they protect everything else), then U5, U9, U10, U7, U8, U14,
U12, U11, U15.

## Goal

Make sure what Splinter learns from is true, that a release is judged by an exam strong enough
to see the effect, and that selection and curriculum spend rollouts where they teach. A pass is
evidence only of "these checks were not violated"; this roadmap adds the checks that were
missing: test strength, canaries, coverage, power.

## Left to do

- [ ] RS-P1 (S, U4) Power-first planning and small-n statistics.
  - Done: exact sign-test power, minimum detectable lead and the fixed alpha spending schedule
    in `splinter-eval::power`; `exam-set power` prints the sign-test power and the smallest lead
    seen; the gate takes `alpha_spending` (opt-in) and `target_effect` and reports a miss on an
    underpowered exam as inconclusive (finding 047). Left: PILOT label under three seeds, Wilson
    intervals below 100 families, a CLI switch for the two gate settings, making spending the
    default after a decision on the first-generation level, the 200-generation null simulation
    through the real gate.
  - `splinter exam power` prints, for the actual exam, the power table (families, observed
    discordance, win share), warns when the minimum detectable effect exceeds the effect the
    run is expected to produce, and labels results PILOT with fewer than three training seeds.
    Interval pieces use brain's Wilson/exact functions (RS-B1) at fewer than 100 families.
    The gate spends alpha across generations (pre-registered schedule).
  - Acceptance: golden table equals the research note's within 0.01; a null simulation over 200
    generations has a false-release rate of at most 0.10 with spending. `make test check`.
- [ ] RS-P2 (M, U1) Canary tasks and shortcut audit.
  - A canary generator in the workspace task family inverts one hidden assertion so any pass
    is a cheat; the gate gains a cheating-rate check; trajectory facts from RS-S2 feed the
    exposure audit (protected-path reads, version-control commands, network attempts).
  - Acceptance: a scripted cheating agent is flagged on at least 90% of 20 canaries and a clean
    one on none; the gate refuses a candidate whose canary pass rate is above 5%. `make test`.
- [ ] RS-P3 (M-L, U2) Test strength at admission and a differential check of passing patches.
  - Done: each mutation-seeded task's hidden-suite kill rate on the faults seeded within three
    lines of its own is measured at admission, kept on the task's check (`hidden_strength`) and a
    task below `MIN_HIDDEN_KILL_RATE` (0.5, with at least two measured variants) is refused as
    `weak_hidden_tests` in the admission report (finding 048). Left: the release gate weighing a
    pass by the recorded strength, the differential check of passing patches, the AUC acceptance
    on 50 tasks with seeded wrong patches, the floor calibrated on that data.
  - Record each admitted task's hidden-suite kill rate on mutants of the reference (the mutation
    operators exist) and refuse below a pre-registered floor; demote a pass whose behaviour
    differs from the reference's to a hint-only strength.
  - Acceptance: on 50 admitted tasks with 10 seeded plausible-but-wrong patches each, the
    kill-rate floor predicts which tasks pass a wrong patch with AUC at least 0.8. `make test`.
- [ ] RS-P4 (S-M, U3) Coverage guard.
  - A fifth gate check: pass@8 and the number of distinct solved exam tasks may not fall by
    more than a bound while pass@1 rises; every generation mixes sealed original data and the
    teacher data, never its own outputs alone. Uses brain's pass@k estimator (RS-B1).
  - Acceptance: a synthetic candidate that memorises half the tasks and loses diversity on the
    rest is refused on coverage and passes today's gate. `make test`.
- [ ] RS-P5 (S, U5) SFT trajectory filters and per-task cap.
  - Filters for degenerate runs (repeated identical action, edits to protected paths, submit
    without a test run, step cap hit, fewer than five turns), a configurable per-task cap
    (test 1 against 2), preference for trajectories that fit the training context whole.
    Each filter logs its count; a bypass exists.
  - Acceptance: three-seed ablation on the sealed exam; adopt the filtered arm only if it is not
    worse (paired sign test p above 0.2) with fewer tokens. `make test`.
- [ ] RS-P6 (S-M, U9) Learnability by disagreement, k, and cross-round skipping.
  - First measurement k=8, then k=2 for tasks already on the frontier; persist the previous
    round's class and skip Always and Never tasks with probability 0.8 while re-testing 20%;
    student-versus-teacher disagreement becomes a recorded class; RL bands are pre-registered
    per stage, SFT needs no band.
  - Acceptance: on a replayed round the policy keeps at least 95% of the tasks that would have
    been on the frontier while cutting rollouts by at least 30%. `make test`.
- [ ] RS-P7 (S, U10) Origin-aware verification strength.
  - A reference-overlap verifier (line-level recall of the student patch against the reference)
    of strength below executable, admitted only for teacher-origin records; self-origin records
    can never enter a soft view (assertion test).
  - Acceptance: teacher-only SFT sets, hard against soft against none, three seeds: soft within
    one standard deviation of hard; recall correlates with test resolution at 0.9 or more on
    our tasks. `make test`.
- [ ] RS-P8 (M, U7) Generator diversity and quotas.
  - A feature-add generator beside mutation; generator and repository as quota groups next to
    concept, kind and strength in the diversity quotas.
  - Acceptance: at equal trajectory counts the mixed-generator set beats mutation-only on the
    sealed exam (paired p below 0.1) or the generator is dropped. `make test`.
- [ ] RS-P9 (M, U8) Higher-order tasks from the student's failed attempts.
  - Apply a failed attempt to the reference tree; if the hidden tests fail reproducibly it is a
    new task whose request is composed from symptoms and the critic's summary, never the tests.
  - Acceptance: yield of at least 20% valid tasks per failed attempt (abandon below 5%); a round
    with half higher-order tasks beats a mutation-only round at equal rollouts (three seeds).
- [ ] RS-P10 (S, U14) Contamination probe for exam families.
  - Per family, the base model's accuracy at naming the file to change from the request alone,
    compared with a family created after the base model's data cutoff; reported, not auto-excluded.
  - Acceptance: on a set where half the families are identifier-renamed, the gap is larger for
    the unrenamed half. `make test`.
- [ ] RS-P11 (S, U12) Skill and memory policy defaults.
  - Candidates only from executable passes; at most three injected (RS-S4); 25% withheld arm;
    delete by realised utility after at least five retrievals; never delete or promote on a
    judged verdict. Extends CL-P2 rules R0, R1, D1.
  - Acceptance: with a paired arm on at least 60 tasks, a unit is retained only if the used arm
    wins an exact sign test at 0.05 or cost falls at least 20% at equal pass rate.
- [ ] RS-P12 (L, U11) Best-of-k with a hybrid selector, then distil.
  - Selector order: regression tests, then the agent's reproduction test, then a learned
    verifier scored through brain's log-probability call (RS-B4); winners feed the SFT view.
    Selector accuracy is reported with Wilson intervals and an expected calibration error.
  - Acceptance: best-of-8 beats a single attempt by at least 5 points absolute on the exam and
    closes at least 40% of the pass@8 gap. Depends on RS-S1, RS-B5.
- [ ] RS-P13 (L, U15) Learned critic with calibration and segment-level selection. Shelve unless
  the critic's calibration error is at most 0.1 and the segment-masked subset is not worse than
  the full set on the sealed exam in three seeds (needs RS-B3).
- [ ] RS-P14 (S, U6) Run the student-recipe sweep: rank 16, 64, 256; learning rate from the
  ten-times-full-fine-tune rule; bf16 against 4-bit base; three seeds; record the decision in
  the experiment file defaults. Uses RS-B2.
