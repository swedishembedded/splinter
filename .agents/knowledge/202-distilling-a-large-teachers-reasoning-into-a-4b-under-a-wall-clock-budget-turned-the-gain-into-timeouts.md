<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 202. Distilling a large teacher's reasoning into a 4B under a wall-clock budget turned the gain into timeouts

**Lesson:** a Qwen3-4B fine-tuned on the 27B teacher's verified trajectories learned the teacher's long reasoning
along with its actions. On tasks of the trained families this paid off. On harder exam families with a 900 s
wall-clock agent limit, the student thought longer than the base model and ran out of time. The training fit then
showed up as timeouts instead of passes. Control either the reasoning or the budget: curate concise trajectories, or
cap thinking per turn, or give the agent a budget in turns or tokens (bench 072).

## Evidence (lane VSFT, item 25, 2026-11-21)

- **Students:** two seeds of a rank-16 LoRA (A2 recipe, 230 steps) on `vim-headless-mastery-sft` v1, which is 576
  verified 27B trajectories (knowledge 199). Held-out loss fell from 2.143 to about 0.81.
- **Training fit** (168 train tasks x 2):
  - pass@1 rose from 0.265 to 0.497 and 0.446, pooled +0.207 [+0.137, +0.275];
  - on the 16 held-out validation tasks the pooled gain was +0.25 [+0.09, +0.42].
- **Exam (14 exam families x 16, at the baseline's load of 72 trials in flight):**

| arm | pass@1 | timeouts | model turns per trial |
|---|---:|---:|---:|
| base | 0.049 | 46 / 224 (21%) | 6.5 |
| v1 seed 1 | 0.067 | 172 / 224 (77%) | 10.7 |
| v1 seed 2 | 0.0625 | 188 / 224 (84%) | - |

- The pooled exam gain was +0.016 [-0.009, +0.040], sign test p = 0.29: not supported.
- One timed-out student turn reasoned for 36k characters and made no tool call. Students' completions averaged
  2.4k-2.7k tokens against the base's 2.0k.
- In v1, 106 of the 576 records were trials the teacher's own agent limit stopped before a final answer. They taught
  acting without concluding.
- Follow-ups pre-registered in the experience-to-competence ledger, item 25:
  - E1c: an equal per-turn thinking budget for all arms;
  - v2c: the shortest verified trajectories per task, with reasoning per turn capped and the cut-off records dropped.

**Superseded in part (2026-11-21, knowledge 209):** under a token and turn budget, on 168 held-out-rung tasks, the same
recipe transfers to families it never trained on (+0.118, family interval [+0.021, +0.220]). The null above measured the
wall clock and a 14-task exam.
