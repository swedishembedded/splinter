<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Validation

A green suite proves an algorithm only when its tests hold it to its specification, not to its own implementation.
Twice a defect hid behind green tests: a policy-gradient run whose optimiser barely moved, and importance ratios
from tokens scored against the wrong reference. This page maps every algorithm Splinter owns to its kind, its
specification and the test that proves it, and says how the proof is run.

```bash
make test/conformance              # per-PR tier: every per-PR row below, and the evidence file (seconds)
make test/conformance/integration  # integration tier: the integration rows, then every mutant (minutes)
make test/methods                  # integration tier: learning effectiveness on a toy model (needs BRAIN_BIN)
```

## Three kinds of correctness

- **Numerical**: a number agrees with an independent reference: values derived by hand or analytically, or golden
  fixtures computed in exact rational arithmetic by a generator committed beside them (Python's standard library at
  generation time only; tests never need Python).
- **Algorithmic**: the method behaves as specified on constructed cases: which records are written, which turns are
  supervised, which groups are compared, which candidate is promoted.
- **Learning**: a tiny model acquires the capability, held-out included (`make test/methods`).

Each check of a core algorithm is also run against deliberate defects. In-test mutants (a two-sided test passed off as
one-sided, arms resampled independently, a sample variance) live in the conformance tests themselves and must be
caught on every run. Source mutants (`scripts/conformance/mutants.tsv`) are applied to production code one at a time
by `make test/conformance/integration`, which counts a mutant killed when the named tests fail and restores the source
after each.

## Evidence

`make test/conformance` builds the proving tests once, runs every per-PR row and writes
`target/conformance/evidence.jsonl` (`make test/conformance/integration` writes the integration rows to
`target/conformance/evidence-integration.jsonl`): one line per row with the algorithm, its kind, the backend and
precision proven, the tests with how many ran and passed, and the commit. A filter that matches no test fails its row.
Splinter's algorithms here are host arithmetic, so the evidence is `cpu`/`f64`. Model computation (losses, gradients,
advantages from rewards, sampling) is brain's and is certified there, per backend and precision; a pass here certifies
none of it.

## The map

A proving test is written `package/target/filter`: target `lib` (the crate's unit tests), `all` (every test binary
of the package) or an integration test binary; filter a substring of the test names. `make test/conformance` and
`make test/conformance/integration` read this table. Both tiers are also part of `make test`.

| Algorithm | Kind | Spec | Proving tests | Tier | Mutants killed |
|---|---|---|---|---|---|
| Paired sign test | numerical | exact one-sided `P(Binomial(n, 1/2) >= wins)` over the `n` discordant pairs, ties dropped, `1.0` with none; golden values from `crates/model/tests/fixtures/conformance` | `splinter-model/it/statistics_conformance` `splinter-model/lib/stats::` | per-PR | two-sided, ties as losses, strict tail (in-test); `sign-test-two-sided`, `sign-test-strict-tail`, `sign-test-ties-counted` |
| Sign test from counts (powered exam, release task level) | numerical | the same tail from `(wins, losses)`, one implementation (`splinter_model::stats::one_sided_p`) | `splinter-model/it/statistics_conformance` `splinter-pipelines/lib/powered::analysis` | per-PR | `powered-sign-test-two-sided` |
| Paired percentile bootstrap | numerical | the interval of the mean per-item difference, items resampled with both verdicts together; each bound is the exact quantile of the multinomial bootstrap distribution within five Monte Carlo standard errors | `splinter-model/it/statistics_conformance` | per-PR | arms resampled independently, one-sided bound (in-test); `bootstrap-one-sided-tail` |
| Family-clustered bootstrap (powered exam, release task level, promotion measurement) | numerical | whole families resampled, the statistic the mean over every task drawn (a family weighs what its tasks weigh); 2.5%, 5% and 97.5% bounds are exact quantiles of the enumerated family bootstrap | `splinter-pipelines/lib/powered::analysis` | per-PR | tasks resampled alone (in-test); `family-bootstrap-equal-weights` |
| Competence intervals | numerical | quality, autonomy and cost resampled by family, pooled ratio; bounds are exact family-bootstrap quantiles | `splinter-eval/it/golden::` `splinter-eval/it/competence::` | per-PR | item-level resample, families weighted equally (in-test); `competence-item-level-resample` |
| Exact power of the sign test | numerical | the chance the gate's own test rejects when each unit is a win with `(d + e) / 2`, a loss with `(d - e) / 2`; critical wins from the handed-in test; spent alpha never sums past the level; hand-derived values (`0.9^5`, `9/256`) | `splinter-eval/lib/power::` | per-PR | |
| pass@k of a samples arm | numerical | a task is solved when any of its `k` graded attempts passed, which is the unbiased `1 - C(n - c, k) / C(n, k)` at `n = k`; the rate arm is `c / n` | `splinter-pipelines/lib/improve::arms` | per-PR | `pass-at-k-as-rate` |
| Cluster units of the gate's sign test | algorithmic | one unit per family: a win when the candidate won more of its items, a tie (dropped) when equal | `splinter-eval/lib/paired::` | per-PR | |
| Strict and shaped reward | numerical | a pass earns 1; a failure `ceiling * sum(w_i m_i)`, ceiling below 1, metrics in `[0, 1]` or refused; the over-length penalty is DAPO's ramp | `splinter-data/lib/views::rewards` `splinter-data/it/policy_gradient::` | per-PR | `pass-earns-zero`, `shaped-reward-sign-reversed` |
| `shaped-partial` reward | numerical | a failure earns `c * max(0, (f - f_naive) / (1 - f_naive))`, nothing at `f_naive = 1`, a missing floor is zero and counted | `splinter-data/lib/views::rewards` `splinter-data/it/policy_gradient::shaping` | per-PR | `naive-floor-ignored`, `partial-credit-unclamped` |
| Groups, zero-variance drop, baseline-only trials | algorithmic | a group is one task version by one policy; a group whose rewards do not differ writes nothing; a written trial nothing of which renders still sets its group's baseline as one record that trains nothing | `splinter-data/it/policy_gradient::` | per-PR | `zero-variance-group-kept`, `groups-merged-across-policies` |
| Learning signal | numerical | `LS = V / mean cost`, `V = (n s2 + 2 v0) / (n + 2)` with `s2` the population variance and `v0` the Beta(1, 1) posterior's `p (1 - p)` from clamped rewards; informative only with two or more differing rewards; drawn by `sqrt(V) / cost`, never when measured flat | `splinter-eval/it/golden::` `splinter-eval/lib/campaign::learning_signal` | per-PR | sample variance, no shrinkage, flat groups informative (in-test); `sample-variance`, `no-shrinkage`, `flat-group-informative`, `flat-group-drawn` |
| Supervision mask | algorithmic | only assistant actions are supervised, never a user turn, tool reply or system prompt; unpacked records supervise their last message only; packed supervises the same actions | `splinter-data/it/supervision_mask` | per-PR | tool reply or user turn supervised (in-test); `context-message-supervised`, `user-turn-supervised` |
| `--only-actions`, `--repository-history` | algorithmic | only the named tools' calls and the final answer are supervised, the rest is context; a history read is dropped (counted, kept as context), kept, or ends the demonstration | `splinter-data/it/supervision_mask` `splinter-data/it/sft_step_curation` | per-PR | `history-read-supervised` |
| Rendered record against the served request | algorithmic | a conversation that recorded the requests its model was sent is rendered from them as sent; the policy's own prompt and sampled ids are written one to one with their action, and a context the scrub or a rebuild changed carries no ids | `splinter-data/it/sft_step_requests` `splinter-data/it/policy_gradient::exact_tokens` | per-PR | |
| `--sampled-only` | algorithmic | an action without the policy's own sampled tokens is left out and counted; its trial still sets its group's baseline | `splinter-data/it/policy_gradient::exact_tokens` | per-PR | `sampled-only-ignored` |
| Verified-only admission, exposure exclusion | algorithmic | an sft dataset admits only experiences decided pass at the minimum strength (step labels aside); the unverified control never keeps an exposed attempt; a failed verdict is no evidence whatever the solver claimed | `splinter-data/it/sft_admission` `splinter-data/it/sft_step_controls` | per-PR | |
| Held-out splits | algorithmic | a group (task, family) is held out whole wherever its samples sit; seeded, stratified, reproducible | `splinter-data/lib/holdout` | per-PR | |
| Exam seal | algorithmic | an attempt or action touching a reserved family is refused, and a build lists what it trained on | `splinter-pipelines/it/dataset_seal` | per-PR | |
| Promotion gate | algorithmic | promoted only when the dev gain's two-sided 95% family interval is above zero over enough families, retention's one-sided bound is above the margin and a fresh sealed slice verifies | `splinter-eval/it/promotion_gate` `splinter-pipelines/lib/promotion::` | per-PR | |
| Campaign dev gate | algorithmic | the greedy gate advances on a net dev gain and rolls back on none or on a retention loss past its margin, reporting significance at the alpha it may spend; the evidence gate advances only when its e-process crosses the round's level | `splinter-eval/lib/campaign::gate` `splinter-eval/it/evidence_gate` | per-PR | |
| Solve concurrency | algorithmic | a remote model is asked up to its width at once, a served one as wide as its slots; one experience per attempt in task order (read with a barrier, no clock) | `splinter-pipelines/it/solve_concurrency` | per-PR | |
| Rationalisation (STaR's rationalise step) | algorithmic | with scripted model output: a verified, clean attempt is kept without the hint, a leaking or insufficient one is rejected with its verdicts, only kept trajectories reach the dataset, a rerun changes nothing, an eval-side task refuses the run (filters as implemented; see below) | `splinter-pipelines/lib/bench::repair::hinted` `splinter-pipelines/it/bench_rationalise` | integration | |
| Generations (expert iteration) | algorithmic | each generation trains on the admitted experience from the initialisation its replay choice names (a fresh adapter on the base for a rebuild, the champion otherwise), is measured on every protected task and released only through the gate | `splinter-pipelines/it/generations` | integration | |
| Learning methods | learning | each method moves a held-out score on a toy model | `make test/methods` | integration | |

## Not yet certified

- brain's advantage normalisation, losses, gradients and updates on CUDA, bf16 and quantised paths: brain's
  validation, not Splinter's.
- pass@k from more attempts than `k`: Splinter has no unbiased estimator for `n > k`. Its one pass@k count (the
  improvement experiment's samples arm) is any-of-the-graded-attempts, exact only when all `k` attempts were graded;
  an ungraded attempt makes it any-of-fewer.
- Rationalisation: the user's rule that a kept trace is rejected only when it holds the actual answer before it was
  observed (no word lists) is not what the filters do; they also reject on a list of leak terms and require a first
  read before any write and a sufficiency rerun.
- Near-duplicate removal and per-task caps of a curated set (item 24b's "at most 3 passes per task, least reasoning
  first") are done by campaign scripts: Splinter's dataset build has no such option, so nothing here proves them.
- Results computed outside Splinter (the paired task bootstraps, sign tests and pass@k tables of campaign ledgers
  produced by analysis scripts) are not proven by this map.
