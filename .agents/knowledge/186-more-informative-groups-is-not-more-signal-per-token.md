<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 186. More informative groups is not more signal per token

**Lesson:** the share of groups whose rewards differ (knowledge 185) counts where an advantage exists, not how much
there is. Partial credit for failures makes more groups informative, but it also lifts every failure towards the
pass, which shrinks the spread of the groups that already had one. Judge a reward or a selection by the reward
variance it buys per token (`sum Var(r)` per million completion tokens), beside the informative share, and never by
the share alone.

Measured 2026-11-19 on lane R's practice exports (9 runs, 1,312 trials of A2s1, 160 task groups of 2+ graded attempts,
47 lineages), with `splinter experiences capabilities`: the strict pass leaves 83/160 groups informative and
1.155 `sum Var(r)` per million completion tokens; `shaped-partial` at `f_naive = 0`, `c = 0.5` makes 110/160 informative
(+27) but gives 0.903 per million, 22% less. Drawing the same groups by variance per cost (`sqrt(V) / mean cost`,
uninformative groups dropped) gives 3.333 per million under the strict pass and 2.290 under `shaped-partial`: the
selection buys about three times the variance per token, the reward does not.

What follows: with z-scored advantages brain rescales each group anyway, so the shaped reward's gain is the groups it
opens, not their size; whether those small spreads teach is for a fresh round to say (ledger item 19), on strict pass
over paired held-out tasks. The selection figures are in-sample: the variance of a task drawn for being high regresses
towards the mean in the next round.
