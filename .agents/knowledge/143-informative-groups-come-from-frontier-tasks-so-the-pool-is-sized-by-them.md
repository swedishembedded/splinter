<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 143. Informative groups come from frontier tasks, so a pool is sized by how many it holds

**Lesson:** On a verifiable family the base answers deterministically per prompt (Qwen3-4B, in process, temperature 1,
no reasoning, Lua output prediction), a task's pass rate over 8 attempts is almost always 0 or 1, so only about one task
in ten sits on the frontier where a group has a reward spread. The share of informative groups in a batch is therefore
set by how many frontier tasks the pool holds, not by how many attempts a group gets, and a small pool is exhausted by
the refill that over-provisioning asks for.

## Measured (campaign controller, calibration at k = 8, groups of 8)

- Per-level mean pass rate of the base over 20 tasks per level (sum, product, string, loop, sort, long arithmetic):
  0.90, 0.45, 0.41, 0.00, 0.31, 0.02. Of the 120 tasks, 12 were on the frontier (a rate strictly between 0 and 0.9),
  72 never solved, 36 retired at 0.9 or more.
- Informative share of the groups collected for a batch of 12 to 16: 4.4 percent from a 64-task pool, 13.1 percent from
  120 tasks (11 of 84 groups, the batch 5 groups short), 38.9 percent from 240 tasks (14 of 36, batch full), 31.0
  percent from 360 tasks (18 of 58, batch full). The audit's earlier real-weight runs had 10.7 to 14.7 percent.
- A defect found on the way: a refill drew the same prompts again, so a small pool filled its batch with repeated
  zero-spread groups. The controller now attempts a task once per batch (`campaign::stages::collect`, spec
  `a_task_is_attempted_once_per_batch_however_often_the_batch_is_refilled`).

## What this does and does not establish

One model, one family, no reasoning, answer-only completions: the bimodality is a property of that setting. It
establishes that the informative-group fraction is a gate input worth logging (it was the first number that explained
why the runs could not see an effect) and that calibration plus a pool of several hundred tasks is the cost of a batch
that has signal. Evidence: runs of `samples/campaign-lua` on one GH200 (2026-11-13), the ledger of each in the
campaign's pack.
