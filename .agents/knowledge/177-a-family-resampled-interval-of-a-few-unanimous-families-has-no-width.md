<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 177. A family-resampled interval of a few unanimous families has no width

**Lesson:** a percentile bootstrap that resamples families says nothing about how many families
there are when they all agree: every resample of unanimous families has the same mean, so the
interval collapses to a point. Two families, both won, give the interval [1, 1], above zero. A
rule "the dev gain's 95% interval is above 0" on such an interval decides on as few as two
families, where the exact sign test over families needs five (`p = 1/32`).

Found 2026-11-18 moving the release gate onto the one promotion rule (`promotion::decide_unexamined`
on `splinter_pipelines::promotion::measurement`). The spec
`release::variants_of_the_trained_facts_join_the_exam_and_are_reported_apart` grades a candidate
on three held-out records, all won: the sign test reports `p = 0.125` (not significant), and the
promotion's dev measurement is 3 items with the interval (1.0, 1.0), so the release is promoted.
Under the sign test the same candidate was refused until nine variants were added.

**Fix (2026-11-18, lane G):** the rule counts a dev gain only over at least
`MIN_GAIN_FAMILIES = 6` families (`splinter_eval::promotion`), derived from `ALPHA`: the least `n`
with `2^-n <= ALPHA / 2`, the fewest unanimous families the exact sign test finds significant at
the two-sided interval's tail (1/64 <= 0.025 < 1/32). Below it no outcome of the exam could be a
gain shown at the stated level, so none counts; the decision records `too_few_families` with the
count and the minimum. A verdict whose section reports no `families` is refused the same way.

Why a minimum and not a sign test beside the interval: the examiner's verdicts carry aggregates
and a family count, never the family signs, and the gate must not recompute or override the
examiner's interval. A minimum reads the same on both paths (`decide` on bench's verdicts,
`decide_unexamined` on `pipelines::promotion::measurement`, which now carries `families`); a sign
test on the unexamined path alone would hold Splinter's own measurement to a stricter rule than
bench's. The spec `release_variants::variants_of_the_trained_facts_join_the_exam_and_are_reported_apart`
refuses the three-record candidate again; with the nine variants it promotes.

Still open: above the minimum a percentile bootstrap (and bench's cluster-t interval) over 6 to
about 15 families is approximate and can be narrower than its level (a mixed 6-family exam may pass
where the sign test would not); retention's non-inferiority bound has no family minimum, so a few
tied families give a bound of no width at 0. Read a gate report on a handful of families with the
sign test beside it.
