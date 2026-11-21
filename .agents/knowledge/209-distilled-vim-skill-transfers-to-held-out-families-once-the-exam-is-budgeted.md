<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 209. Distilled vim skill transfers to held-out families once the exam is budgeted and wide enough

**Lesson:** a Qwen3-4B LoRA trained with v1's recipe on the 27B teacher's verified thinking-mode passes of some
difficulty rungs raised the pass rate on rungs it never saw. Leave-families-out, three folds over the 14 rungs, 168
held-out tasks x 8, budget B = 15,114 tokens / 9 turns: base 0.235, students 0.353, **+0.118**, family-clustered 95%
interval [+0.021, +0.220], Holm p 0.0003, both seeds above 0 (generic-learning E1, 2026-11-21). Item 25 had read
"no transfer" from the 14-task exam under a wall-clock limit (+0.016 [-0.009, +0.040], knowledge 202). That reading
measured the timeouts and a floor, not transfer. The students still exhaust B in 56-59% of trials (base 10%).

**How to apply:** test transfer by holding out whole families from a calibrated pool (reference arm between 0.15 and
0.85, at least 14 families) under a token and turn budget, never from a small fixed exam. A null on a 14-task or
wall-clock exam is not evidence against transfer. The instrument detects +0.05 effects, which unblocks method
experiments (E4, E5) on it.
