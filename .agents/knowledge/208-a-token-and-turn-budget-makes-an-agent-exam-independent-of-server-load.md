<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 208. A token and turn budget makes an agent exam independent of server load, and shows a distilled student's length as failures

**Lesson:** with the trial's length budgeted in completion tokens and turns (B = 1.5 x the base's p90 of passing
trials: 15,114 tokens, 9 turns) and brain's per-turn thinking budget closing every think block (2,048 tokens), pass
rates on the same tasks agree at 4 and 32 trials in flight per arm. On the 16 vim validation tasks x 16 the
differences were base +0.023 [-0.023, +0.070], v1s1 +0.012 [-0.035, +0.055] and v1s2 -0.004 [-0.086, +0.086], every
Holm p above 0.5 (generic-learning E0, 2026-11-21). Under the wall-clock limit of item 25 E1, the same students had
timed out in 71-96% of trials depending on load (knowledge 202). The budget also makes the cost of the teacher's long
reasoning visible: the thinking-mode 27B distillation v1 exhausts B in about half of its trials (49-52%, base 8-9%),
and its in-family gain over base (+0.24 to +0.26) is what survives inside the budget.

**How to apply:** examine agents only under a token and turn budget, the same for every arm, with wall clock as an
infrastructure backstop. Report each arm's exhaustion share beside its pass rate, because a student that copies a
teacher's length fails by exhaustion, not by error. Choose the load for throughput, not for comparability.
