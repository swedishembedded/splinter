<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 185. A strict-pass reward wastes the verifier's partial credit

**Lesson:** a group-relative advantage is zero in every group whose rewards are all equal, so the share of groups with
a reward spread is the learning signal a rollout batch buys. Bench verifiers already compute weighted partial credit
(`functional_correctness`), but the default `RewardShape::unshaped()` gives every failure 0, which throws that spread
away: about half of the groups were all-pass or all-fail and carried no advantage.

Measured on lane R's practice exports (1,306 trials; 160 groups of 6-8 attempts, 31 single trials skipped): 81/160
groups have a strict-pass spread, 110/160 a `functional_correctness` spread, and every pass-spread group is among them.
Rewarding failures by partial credit makes 29 more groups informative, from 51% to 69%, at no extra GPU cost.

What to do: shape the reward (`r = 1` on a pass, else a capped partial credit above bench's naive floor, or raw
`functional_correctness` through `--reward-metric` and `--fail-ceiling` until bench exports the floor), keep the gate on
strict pass, and track over-listing and abstention drift. Count informative groups per batch as a standing metric.

Found 2026-11-19, expert review; the ranked items are in `.agents/roadmap/experience-to-competence.md`, section
"Learning signal per GPU-hour".
