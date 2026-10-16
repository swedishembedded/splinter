# 047. Ten gates at five percent release a null candidate four times in ten

The improvement check holds each candidate to a one-sided sign test at 0.05,
and a lineage is a sequence of such candidates. A candidate no better than its
champion passes each test with probability up to 0.05, so across ten
generations the chance that at least one is released on luck is up to
1 - 0.95^10 = 0.401. The exact test is also discrete: with few independent
families the achievable levels jump, and five families need unanimity
(p = 1/32 = 0.031), six too (1/64), eight tolerate one negative (7 of 8 is
9/256 = 0.035), and four can never reach 0.05 (1/16).

Two consequences were measured on the fixtures:

- A fixed spending schedule alpha / (g (g + 1)) for generation g sums to less
  than alpha however long the sequence runs (200 generations: 0.0498 of 0.05).
  It is strict: generation 1 may spend 0.025, so five unanimous families pass
  at the plain level and are refused as a first generation; six pass the first
  and fail the third (0.0042). The gate makes it opt-in (`alpha_spending`),
  because it halves the first release's level and a gate that silently changes
  what releases is worse than one that says what it holds.
- A miss on a small exam is not evidence of no effect. Six families with half
  discordant cannot show a lead of 0.3 with four chances in five (and ten
  families with half discordant cannot show any), so when a target lead is
  named the check reports such a miss as inconclusive, not as a failure. A
  rejection stays a pass: the p-value is valid at its level however few units
  there are.

How to avoid it: size the exam by its exact sign-test power over families
(`exam-set power`), not by the number of tasks, and spend alpha across a
lineage.
