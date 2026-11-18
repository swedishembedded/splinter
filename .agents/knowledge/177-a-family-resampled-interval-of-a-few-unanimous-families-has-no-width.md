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

What would hold the rule to its stated level on small exams: a least number of families for an
interval (the examiner's verdict carries `families`), or an interval valid at small n (an exact
or Wilson bound on the family win share). Until then a gate report on a handful of families is to
be read by its family count, which the sign test beside it shows.
