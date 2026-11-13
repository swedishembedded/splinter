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
