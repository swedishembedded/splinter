<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Registration of RSI4: cumulative self-taught generations retrained from the base, at the policy's frontier

Written on 2026-11-13, before any run of RSI4. The design, exam, criterion and analysis below are fixed by this commit;
builds, digests and the settings resolved from the defaults named here go into `rsi4-launch.md`, committed before
generation 1. Later changes are dated deviations beside the outcome, which is recorded whatever it is.

## Why a fourth registration

RSI3 (`rsi3.md`) improved every lineage at generations 1 and 2 on bench's sealed exam, but at generation 3 one lineage
held flat (29 wins / 28 losses) and one fell (18 / 28) although its own dev gate accepted it; only one lineage improved
three times. Two mechanisms fit the records: (1) the learning signal ran out: the policy mispredicted 568 of 600 of its
own proposals at generation 1 but only 229 of 554 at generation 2, so later generations train mostly on what they
already know; (2) each generation continued training the previous adapter, so drift and forgetting compound. Published
expert-iteration results (ReST-EM) retrain from the base on the accumulated self-generated data each iteration, and
curriculum work keeps examples at the frontier of the current policy. RSI4 changes exactly these two things.

## Design

As RSI3 (policy Qwen/Qwen3-4B, thinking off; four lineages differing only in seed; three generations; brain only
through its API; the policy proposes Lua 5.1 programs from the manual and the interpreter labels them; no stronger
model, teacher or human answer; dev and retention sets only from the learner's own proposals and generator, powered),
except:

- **Cumulative retraining from the base:** generation g's candidate is a fresh adapter trained from the base on the
  union of every accepted generation's verified data so far (generations 1..g), not a continuation of generation g-1's
  adapter. The data of generation g is still proposed and answered by generation g-1's champion.
- **Frontier selection:** the current champion predicts each of its proposals k times (sampled; k fixed in the launch
  record); a program enters training only when the champion's predictions are not all correct (mispredicted at least
  once), and the interpreter's output is the target. Programs the champion always predicts correctly are not trained
  on (they stay available to replay only as the union above already contains earlier generations' frontier data).
- **Proposal budget:** proposals per generation are raised until the frontier set reaches the size fixed in the launch
  record, within a hard cap of requests also fixed there.

## Exam, criterion, analysis

Unchanged from RSI3: bench alone holds, serves and grades the sealed 889-item Lua exam on its pinned build (direct-
completion protocol), excludes from each step every item matching any program either compared model was trained on,
and returns signed aggregate verdicts. A step (generation 1 vs base, 2 vs 1, 3 vs 2) holds when wins exceed losses and
the one-sided exact sign test gives p < 0.05; a lineage succeeds when all three steps hold; **RSI4 meets its criterion
when at least three of the four lineages succeed.** Pooled per-step tests, exclusions, frontier sizes, dev and
retention trajectories and GPU-hours are reported beside it.

## Outcome (2026-11-15): criterion met, three of four lineages improved at every generation

Bench examined every champion on its pinned build (bench `390b4f12`, brain d11e91340, f32 KV, the exam's registered direct-completion
protocol, sealed 889-item exam, training-overlap exclusion on) and returned signed aggregate verdicts; every signature verifies and its
signed plan verdict reads `plan_met: true` (3 series succeeded, 0 failed). The learner never read the exam or a verdict. Items right
(base 12 in every lineage), with each step's wins/losses and one-sided sign-test p:

| Lineage | gen 1 vs base | gen 2 vs gen 1 | gen 3 vs gen 2 | succeeds |
|---|---|---|---|---|
| r0 | 92 (82/2, 1.8e-22) | 138 (69/23, 8.4e-7) | 192 (93/39, 1.5e-6) | yes |
| r1 | 131 (121/2, 7.2e-34) | 181 (79/29, 8.0e-7) | 225 (99/55, 2.5e-4) | yes |
| r2 | 118 (109/3, 4.5e-29) | 169 (79/28, 4.2e-7) | 241 (95/23, 6.9e-12) | yes |
| r3 | 121 (110/1, 4.3e-32) | 187 (93/27, 5.6e-10) | 218 (67/36, 0.0015) | yes |

Pooled over the lineages examined: gen 1 vs base 422/8 (p 1.0e-113), gen 2 vs gen 1 320/107 (p 4.8e-26), gen 3 vs gen 2 287/117 over
three lineages (p 6.9e-18). One exam item matched a training program in r2 and was excluded from r2's comparisons; no other overlap and
no integrity flag. Unlike RSI3, no lineage stalled at generation 3: the generation-3 steps are the largest absolute gains of two lineages.
The learner's own dev gate was a weak predictor of the exam at generation 3 (r1: dev 25/19, p 0.23, exam 99/55, p 2.5e-4). Deviations
(all before any verdict; see `rsi4-launch.md`): the examiner's bench build moved to `390b4f12` (base identity by content, server lifetime),
and the examiner served on gpu3 instead of gpu1. What this establishes: on this task (predicting Lua 5.1 program output), with Qwen3-4B
and no stronger model or human answer, three independent lineages of self-taught generations, each retrained from the base on the
interpreter-verified frontier of its own proposals, improved at each of three generations on an exam the learner never saw. What it
does not establish: transfer to other tasks or models, or behaviour beyond three generations.

- 2026-11-15 ~06:50 (r3 generation 3, examined after the outcome above was written): 218 vs 187 (67/36, p 0.0015), the step holds, 0
  excluded, no flag. The signed plan verdict now reads 4 series succeeded, 0 failed; pooled gen 3 vs gen 2 over four lineages 354/153
  (p 1.1e-19). All four lineages improved at every generation.
- 2026-11-15 (correction to the outcome text above): "the generation-3 steps are the largest absolute gains of two lineages" is wrong.
  The gains per generation are r0 +80/+46/+54, r1 +119/+50/+44, r2 +106/+51/+72, r3 +109/+66/+31: generation 1 is every lineage's
  largest step; the generation-3 step exceeds the generation-2 step in two lineages (r0, r2). The table and the criterion are unchanged.
