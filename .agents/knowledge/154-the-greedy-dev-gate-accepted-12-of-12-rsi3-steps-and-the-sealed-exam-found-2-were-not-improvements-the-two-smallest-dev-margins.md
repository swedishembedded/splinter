<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 154. The greedy dev gate accepted 12 of 12 RSI3 steps; the sealed exam found 2 were not improvements, the two smallest dev margins

**Lesson:** the lineage's dev gate advances on any net gain over the same 400 dev units every generation (significance is
reported, not required), so from generation 2 on it is the paper TRUST's "greedy gate" under full validation reuse. Measured
against bench's sealed exam, it accepted every RSI3 step, and the two steps the exam found not to be improvements were the two
with the smallest dev margins. A gate that controls the false evolution rate (the evidence gate, `gate = evidence`) is now an
opt-in; with a tolerance set on the scale of our pass-rate changes it would have kept the incumbent at both steps most of the
time, at a cost in accepted true gains. The paper's own tolerance (eps = 0.10) would have accepted all 12: on a dev set where
95 percent of pairs are ties, a tolerance of a tenth of the difference scale makes ties evidence of noninferiority.

## Evidence

Recomputed from `~/project/shared/rsi3/lineages/r{0..3}/rounds.jsonl` (the dev decisions) and
`~/project/shared/rsi3/handoff/verdicts/r{0..3}/{1,2,3}.json` (bench's signed verdicts, plan `rsi3-registered`, items no compared
model trained on):

| Step | Dev wins/losses of 400 | Dev p | Exam wins/losses | Exam p | Exam holds |
|---|---|---|---|---|---|
| r0 g1 | 67/32 | 0.0003 | 58/0 | 3e-18 | yes |
| r0 g2 | 22/5 | 0.0008 | 78/14 | 3e-12 | yes |
| r0 g3 | 13/8 | 0.19 | 29/28 | 0.50 | **no** |
| r1 g1 | 74/26 | <1e-4 | 72/0 | 2e-22 | yes |
| r1 g2 | 16/7 | 0.047 | 41/12 | 4e-5 | yes |
| r1 g3 | 7/5 | 0.39 | 18/28 | 0.95 | **no** |
| r2 g1 | 96/36 | <1e-4 | 67/0 | 7e-21 | yes |
| r2 g2 | 23/8 | 0.0053 | 38/19 | 0.0082 | yes |
| r2 g3 | 16/9 | 0.11 | 48/13 | 4e-6 | yes |
| r3 g1 | 143/24 | <1e-4 | 91/0 | 4e-28 | yes |
| r3 g2 | 27/14 | 0.030 | 42/15 | 2e-4 | yes |
| r3 g3 | 12/5 | 0.072 | 47/23 | 0.0028 | yes |

All 12 advanced. FER_0 (accepted steps that were not improvements, by the exam's registered rule) = 2/12 = 0.17; strict
regressions (exam losses above wins) 1/12 = 0.08 (r1 g3). Net dev margins of the two: 2 and 5; every other step had 7 or more.

What the evidence gate would have done on the same dev counts (`decide_evidence` arithmetic, replayed in a scratch script over
2000 random draw orders per step; zeta = 0 at generation 1 and 1 after, since the same 400 units are reused; q = 0.3, horizon
3 rounds per lineage, each step at the level it gets when every earlier step was accepted, which is what happened - a rejected
step would have changed the later candidates, so this is a single-step counterfactual, not a replay of a lineage):

| eps, beta | Expected accepted of 12 | Of which exam non-improvements | Acceptance at r0 g3 / r1 g3 |
|---|---|---|---|
| 0.10, 0.01 (the paper's) | 12.00 | 2.00 | 1.00 / 1.00 |
| 0.02, 0.01 | 7.85 | 0.13 | 0.10 / 0.02 |
| 0, 0 | 7.86 | 0.12 | 0.09 / 0.03 |

The cost: the exam-held steps r1 g2, r3 g2, r2 g3 and r3 g3 are accepted only 0.17-0.71 of the time, because 400 dev units with
a few percent discordance hold little evidence at a level of 0.05-0.1. With the evidence gate a generation should run several
rounds (the champion stays on rejection), and the dev pool should be larger than one round's draw.

The property tests in `crates/eval/tests/it/evidence_gate.rs` show the mechanism on simulated data: selecting the best of 8 null
candidates on a fully reused 200-item set, the greedy gate accepted 0.36, the evidence gate without overlap correction 0.19 at
alpha 0.1, and with beta = 0.06 0.03.
