<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 206. "Not supported" hides two results: a target ruled out, and an interval that still holds it

**Lesson:** judge an arm's interval against the effect the hypothesis claimed, not only against zero. An upper bound
below the target *falsifies* the hypothesis; an interval that holds both 0 and the target is *inconclusive* and says
the experiment was too small. The ledger reported both kinds as "not supported", "STOP" or "no go", which invites
retiring an idea that was never tested at the needed power, or re-running one that already was. `splinter campaign
problems` records the two apart and moves a belief only on a falsification.

## Evidence (experience-to-competence items 21 and 25, 2026-11-20/21; the campaign's `problems` spec replays them)

| Arm | difference [95% interval] | target | verdict by the rule | the ledger's word |
|---|---|---|---|---|
| item 25 E1b, pooled v1 - base | +0.016 [-0.009, +0.040] | +0.05 | falsified: the upper end is below the target | not supported |
| item 21 S1, A2 (c) - (a) | -0.002 [-0.038, +0.035] | +0.06 | falsified | STOP |
| item 21 S1b, (g) - (a) | +0.029 [-0.050, +0.105] | +0.06 | inconclusive | inconclusive |
| item 21 S1b, (h) - (a) | +0.044 [-0.012, +0.102] | +0.06 | inconclusive | no go |
| item 21 S1b, (f) - (a) | +0.010 [-0.045, +0.065] | +0.06 | inconclusive (upper end 0.005 above the target) | does nothing |

- (h) and (f) were read as negative results; their intervals cannot exclude the go margin, so they rank as
  inconclusive and keep their belief, while (c) and E1b lose 0.20 each.
- A real gain smaller than the target (an interval above 0 whose upper end is below the target) is also falsified:
  the hypothesis named an effect size and that size is ruled out.
- The rule's code and specs: `splinter_eval::campaign::problems` (`classify_effect`), validated by the mutants
  `problem-inconclusive-effect-falsified` and `problem-inconclusive-as-falsified`.
