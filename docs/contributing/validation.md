<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Validation

A green suite proves an algorithm only when its tests hold it to its specification, not to its own implementation.
Twice a defect hid behind green tests: a policy-gradient run whose optimiser barely moved, and importance ratios
from tokens scored against the wrong reference. This page maps every algorithm Splinter owns to its kind, its
specification and the test that proves it, and says how the proof is run.

```bash
make test/conformance              # per-PR tier: every per-PR row below, and the evidence file (seconds)
make test/conformance/integration  # integration tier: every source mutant (minutes)
make test/methods                  # integration tier: learning effectiveness on a toy model (needs BRAIN_BIN)
make test/contract/brain           # integration tier: the records Splinter writes are the records brain's trainer reads (needs BRAIN_BIN)
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
`target/conformance/evidence.jsonl`: one line per row with the algorithm, its kind, the backend and
precision proven, the tests with how many ran and passed, and the commit. A filter that matches no test fails its row.
Splinter's algorithms here are host arithmetic, so the evidence is `cpu`/`f64`. Model computation (losses, gradients,
advantages from rewards, sampling) is brain's and is certified there, per backend and precision; a pass here certifies
none of it.

## The map

A proving test is written `package/target/filter`: target `lib` (the crate's unit tests), `all` (every test binary
of the package) or an integration test binary; filter a substring of the test names. `make test/conformance` and
`make test/conformance/integration` read this table (the integration tier's one row, the learning methods, is
`make test/methods`; the tier itself runs the source mutants). The per-PR tier is also part of `make test`.

| Algorithm | Kind | Spec | Proving tests | Tier | Mutants killed |
|---|---|---|---|---|---|
| Paired sign test | numerical | exact one-sided `P(Binomial(n, 1/2) >= wins)` over the `n` discordant pairs, ties dropped, `1.0` with none; golden values from `crates/model/tests/fixtures/conformance` | `splinter-model/it/statistics_conformance` `splinter-model/lib/stats::` `splinter-model/it/external_statistics` | per-PR | two-sided, ties as losses, strict tail (in-test); `sign-test-two-sided`, `sign-test-strict-tail`, `sign-test-ties-counted` |
| Sign test from counts (powered exam, release task level) | numerical | the same tail from `(wins, losses)`, one implementation (`splinter_model::stats::one_sided_p`) | `splinter-model/it/statistics_conformance` `splinter-pipelines/lib/powered::analysis` | per-PR | `powered-sign-test-two-sided` |
| Paired percentile bootstrap | numerical | the interval of the mean per-item difference, items resampled with both verdicts together; each bound is the exact quantile of the multinomial bootstrap distribution within five Monte Carlo standard errors | `splinter-model/it/statistics_conformance` | per-PR | arms resampled independently, one-sided bound (in-test); `bootstrap-one-sided-tail` |
| Family-clustered bootstrap (powered exam, release task level, promotion measurement) | numerical | whole families resampled, the statistic the mean over every task drawn (a family weighs what its tasks weigh); 2.5%, 5% and 97.5% bounds are exact quantiles of the enumerated family bootstrap | `splinter-pipelines/lib/powered::analysis` | per-PR | tasks resampled alone (in-test); `family-bootstrap-equal-weights` |
| Competence intervals | numerical | quality, autonomy and cost resampled by family, pooled ratio; bounds are exact family-bootstrap quantiles | `splinter-eval/it/golden::` `splinter-eval/it/competence::` | per-PR | item-level resample, families weighted equally (in-test); `competence-item-level-resample` |
| Two-sided sign test, Holm, unbiased pass@k | numerical | two-sided `min(1, 2 min(P(X >= wins), P(X <= wins)))`; Holm's step-down `max_{j <= i} min(1, (m - j) p_(j))` in the order given; per task `1 - C(n - c, k) / C(n, k)`, undefined for `k > n`; cross-checked against scipy, statsmodels and the HumanEval estimator | `splinter-model/it/statistics_conformance` `splinter-model/it/external_statistics` | per-PR | one-sided as two-sided, upper tail doubled, Bonferroni, step-down without the running maximum, any of the attempts, the rate whatever k (in-test); `two-sided-as-one-sided`, `holm-as-bonferroni`, `pass-at-k-any-of-n` |
| Arm comparison (`experiences compare-arms`) | numerical | per arm: trials, graded (timeouts included and counted apart), infrastructure errors in no denominator, pass@1 and unbiased pass@k over the tasks with at least k graded attempts; per arm against the reference over the tasks graded in both: mean per-task difference, task and family bootstrap intervals, better/worse/tied, one- and two-sided sign tests, each Holm-adjusted across the arms; all against an exact golden comparison of three arms | `splinter-pipelines/lib/arm_compare` | per-PR | biased pass@k, one-sided as two-sided, no Holm (in-test); `compare-arms-timeout-ungraded`, `compare-arms-infra-counted-as-failure`, `compare-arms-unpaired-tasks` |
| Exact power of the sign test | numerical | the chance the gate's own test rejects when each unit is a win with `(d + e) / 2`, a loss with `(d - e) / 2`; critical wins from the handed-in test; spent alpha never sums past the level; hand-derived values (`0.9^5`, `9/256`) | `splinter-eval/lib/power::` | per-PR | |
| pass@k of a samples arm | numerical | a task is the unbiased `1 - C(n - c, k) / C(n, k)` of its graded attempts by the proven estimator (`splinter_model::stats::pass_at_k`), absent with fewer than `k` graded; the rate arm is `c / n` | `splinter-pipelines/lib/improve::arms` | per-PR | `pass-at-k-as-rate` |
| Intervention lower bound (`splinter improve`) | numerical | `mean - t * se` over the arm's units, `t` Student's one-sided quantile at the pricing's `confidence` with `n - 1` degrees of freedom; at least two units; the t distribution against scipy.stats.t | `splinter-eval/lib/intervention::` `splinter-eval/lib/student::` `splinter-eval/it/external_student` | per-PR | the normal quantile in place of Student's (in-test) |
| Wilson score interval (pilot shares, environment band) | numerical | two-sided 95% `(c + z^2/2 ± z sqrt(n p (1 - p) + z^2/4)) / (n + z^2)`, ends clamped to `[0, 1]`, none for no units; one implementation (`splinter_eval::pilot::Share::wilson`), against statsmodels' `proportion_confint(method="wilson")` | `splinter-eval/it/external_wilson` | per-PR | |
| Cluster units of the gate's sign test | algorithmic | one unit per family: a win when the candidate won more of its items, a tie (dropped) when equal | `splinter-eval/lib/paired::` | per-PR | |
| Strict and shaped reward | numerical | a pass earns 1; a failure `ceiling * sum(w_i m_i)`, ceiling below 1, metrics in `[0, 1]` or refused (a verifier metric outside it makes the trial a verifier error, never a metric-less grade); the over-length penalty is DAPO's ramp | `splinter-data/lib/views::rewards` `splinter-data/it/policy_gradient::` | per-PR | `pass-earns-zero`, `shaped-reward-sign-reversed` |
| `shaped-partial` reward | numerical | a failure earns `c * max(0, (f - f_naive) / (1 - f_naive))`, nothing at `f_naive = 1`, a missing floor is zero and counted | `splinter-data/lib/views::rewards` `splinter-data/it/policy_gradient::shaping` | per-PR | `naive-floor-ignored`, `partial-credit-unclamped` |
| Groups, zero-variance drop, baseline-only trials | algorithmic | a group is one task version by one policy; a group whose rewards do not differ writes nothing; a written trial nothing of which renders still sets its group's baseline as one record that trains nothing | `splinter-data/it/policy_gradient::` | per-PR | `zero-variance-group-kept`, `groups-merged-across-policies` |
| Learning signal | numerical | `LS = V / mean cost`, `V = (n s2 + 2 v0) / (n + 2)` with `s2` the population variance and `v0` the Beta(1, 1) posterior's `p (1 - p)` from clamped rewards; informative only with two or more differing rewards; drawn by `sqrt(V) / cost`, never when measured flat | `splinter-eval/it/golden::` `splinter-eval/lib/campaign::learning_signal` | per-PR | sample variance, no shrinkage, flat groups informative (in-test); `sample-variance`, `no-shrinkage`, `flat-group-informative`, `flat-group-drawn` |
| Supervision mask | algorithmic | only assistant actions are supervised, never a user turn, tool reply or system prompt; unpacked records supervise their last message only; packed supervises the same actions | `splinter-data/it/supervision_mask` | per-PR | tool reply or user turn supervised (in-test); `context-message-supervised`, `user-turn-supervised` |
| `--only-actions`, `--repository-history` | algorithmic | only the named tools' calls and the final answer are supervised, the rest is context; a history read is dropped (counted, kept as context), kept, or ends the demonstration | `splinter-data/it/supervision_mask` `splinter-data/it/sft_step_curation` | per-PR | `history-read-supervised` |
| Rendered record against the served request | algorithmic | a conversation that recorded the requests its model was sent is rendered from them as sent; the policy's own prompt and sampled ids are written one to one with their action, and a context the scrub or a rebuild changed carries no ids | `splinter-data/it/sft_step_requests` `splinter-data/it/policy_gradient::exact_tokens` | per-PR | |
| `--sampled-only` | algorithmic | an action without the policy's own sampled tokens is left out and counted; its trial still sets its group's baseline | `splinter-data/it/policy_gradient::exact_tokens` | per-PR | `sampled-only-ignored` |
| Pass curation (`--final-answer-only`, `--distinct-calls`, `--max-passes-per-task`) | algorithmic | in this order: a pass whose last agent step calls a tool or says nothing is left out; passes of a task with the same tool calls (name and arguments, keys sorted, whitespace normalised) are one, the least reasoning kept; at most N passes per task, the least reasoning per action first; each counted; the manifest records the curation | `splinter-data/it/pass_curation` | per-PR | `curation-final-answer-ignored`, `curation-duplicates-kept`, `curation-whitespace-not-normalised`, `curation-cap-by-total-reasoning` |
| Verified-only admission, exposure exclusion | algorithmic | an sft dataset admits only experiences decided pass at the minimum strength (an action labelled good aside, which the outcome-verified control turns off); no control, step labels included, keeps an exposed attempt; a failed verdict is no evidence whatever the solver claimed | `splinter-data/it/sft_admission` `splinter-data/it/sft_step_controls` | per-PR | |
| BM25 lexical ranking | numerical | Lucene's idf `ln(1 + (N - n + 0.5) / (n + 0.5))`, `k1 = 1.2`, `b = 0.75`, a query word once, no shared word no hit, ties to the earlier passage; against an independent implementation of that formula, itself checked against rank_bm25's saturation and length normalisation | `splinter-knowledge/it/external_bm25` | per-PR | |
| Skill graph order | algorithmic | prerequisites first in rounds of ready skills in the graph's order (NetworkX's topological generations); a cycle refused naming exactly the skills on or behind it, never an empty order | `splinter-knowledge/lib/skills::` `splinter-knowledge/it/external_skill_order` | per-PR | |
| Held-out splits | algorithmic | a group (task, family) is held out whole wherever its samples sit; seeded, stratified, reproducible; a record whose side is not `train` or `held_out`, or that is not a JSON object, is refused, never placed as a free sample | `splinter-data/lib/holdout` | per-PR | |
| Exam seal | algorithmic | an attempt or action touching a reserved family is refused, and a build lists what it trained on | `splinter-pipelines/it/dataset_seal` | per-PR | |
| Promotion gate | algorithmic | promoted only when the dev gain's two-sided 95% family interval is above zero over enough families, retention's one-sided bound is above the margin and a fresh sealed slice verifies | `splinter-eval/it/promotion_gate` `splinter-pipelines/lib/promotion::` | per-PR | |
| Factory release gate (`promotion`) | algorithmic | a candidate is promoted only when every required purpose's gate is met: target and transfer by their evaluation's decision, retention by a development plan's sign test (no significant loss, mean loss within the margin) or an examiner's one-sided bound above `-margin`; a retention that measured neither is inconclusive and meets nothing; a protected verdict without its signature meets nothing | `splinter-factory/lib/release::` | per-PR | missing statistics read as retained, signed retention always retained (in-test) |
| Harness search selection (`harness.search`) | algorithmic | each candidate is tested against the baseline at `alpha / candidates`, the best improving one kept, so `alpha` bounds the chance a useless candidate wins | `splinter-factory/it/harness_search::` | per-PR | |
| Dev-validation gate | algorithmic | the greedy gate advances on a net dev gain and rolls back on none or on a retention loss past its margin, reporting significance at the alpha it may spend; the evidence gate advances only when its e-process crosses the round's level | `splinter-eval/lib/campaign::gate` `splinter-eval/it/evidence_gate` | per-PR | |
| Solve concurrency | algorithmic | a remote model is asked up to its width at once, a served one as wide as its slots; one experience per attempt in task order (read with a barrier, no clock) | `splinter-pipelines/it/solve_concurrency` | per-PR | |
| Learning methods | learning | each method moves a held-out score on a toy model | `make test/methods` | integration | |
| Brain data contract | algorithmic | the datasets the sft-step, sft-final, preference, feedback and policy-gradient views write are accepted by a served brain's own dataset check, which reads them as its trainer does; the supervised tokens of each record equal the assistant's actions counted from their characters (never a system prompt, a user turn or a tool observation), a policy-gradient record trains exactly its sampled ids, and a hand-broken record (log-probability count, id mismatch) is refused with its rule; brain's tokenization and numerics are not claimed | `make test/contract/brain` | integration | |

## Not yet certified

- Learning effectiveness (`make test/methods`): the smokes cannot run against the current brain, whose toy checkpoint
  moved to its qwen3 sample and whose server no longer serves a local, card-less checkpoint by its store name; the
  row has had no live evidence since brain's capabilities became actions (`validation/gaps.yaml`, G-NEW-06).
- brain's advantage normalisation, losses, gradients and updates on CUDA, bf16 and quantised paths: brain's
  validation, not Splinter's.
- Results computed outside Splinter (the paired task bootstraps, sign tests and pass@k tables of experiment ledgers
  produced by analysis scripts before `experiences compare-arms`) are not proven by this map.

## Comparing arms

Report an experiment's arms with `experiences compare-arms`, never with a script of its own: it computes every number
with the statistics this map proves.

```bash
splinter experiences compare-arms --reference base \
  --arm base=jobs/e2a-base --arm gen1=jobs/e2a-gen1 --arm gen2=jobs/e2a-gen2 \
  --k 16 --json
```

An arm is a Harbor job directory (`--arm NAME=DIR`) or the lines of a trials file (`--trials FILE`, one
`{"task", "arm", "family"?, "attempt"?, "pass" or "reward", "status"?}` per line, status `graded`, `timeout` or
`infra_error`). `--families FILE` (a JSON object of task to family) adds the family-clustered interval. A Harbor
trial passes on a reward of exactly 1; an agent timeout that was still graded is graded and counted as a timeout;
any other exception, or no reward, is an infrastructure error and in no denominator. The report states the sign
test's convention; the Holm adjustment is over the arms compared in that run. The interval seed is drawn and printed
unless `--interval-seed` names a pre-registered one.
