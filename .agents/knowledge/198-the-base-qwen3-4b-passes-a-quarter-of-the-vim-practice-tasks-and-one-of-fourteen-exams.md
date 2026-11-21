<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 198. The base Qwen3-4B passes a quarter of the vim practice tasks and one of fourteen exams

**Lesson:** the vim benchmark (`vim-headless-mastery`, 168 train + 14 eval Harbor tasks, one MCP tool that runs Ex
commands in a headless Vim, graded byte for byte) has room on both sides for a 4B: base pass@1 0.263 on train with
73 tasks never solved in 5 attempts, and 0.071 on eval, where the only success is one exam passed 5 of 5. The eval
set is 14 tasks, so it can only show large gains (bench 068).

## Run (2026-11-21, one GH200)

- Model `Qwen/Qwen3-4B` (base, no adapter), brain-5e25d5b82 `--max-batch 256` (one instance; brain 538), sven
  1db749d9 with its **default** profile (the compact profile offers no MCP tool: sven note
  a-compact-profile-with-an-mcp-only-allow-list-offers-no-tool), bench's Sven adapter through `bench gateway`,
  sampling T 1.0 / top-p 0.95 / top-k 20, per-trial seeds from run seed 20261121, agent limit 600 s.
- Oracle on train: 168/168 full credit. During the agent phase `/tests` and `/solution` are empty and the host is not
  visible (probe agent; Harbor uploads tests only for the verifier).

| side | trials | pass@1 | pass@5 | agent timeouts | infra errors |
|---|---:|---:|---:|---:|---:|
| train | 840 (168 x 5) | 0.263 | 0.566 | 0 | 0 |
| eval | 70 (14 x 5) | 0.071 | 0.071 | 0 | 0 |

- Train by difficulty (pass@1): 1: 0.17, 2: 0.39, 3: 0.22, 4: 0.36, 5: 0.40, 6: 0.30, 7: 0.00, 8: 0.00, 9: 0.33,
  10: 0.60, 11: 0.24, 12: 0.00, 13: 0.32, 14: 0.14. Difficulty is not monotone in it: rungs 7 (registers and macros),
  8 (visual editing) and 12 (refactoring capstones) score 0 in 140 trials, and rung 1 (starting from zero) only 0.17.
- 8 train tasks passed every attempt, 73 none.
- Wall: the first attempt of all 168 took 14 minutes; four more attempts (672 trials at 224 at once) 25 minutes.
  The median trial spent 174-251 s starting its sandbox and 67-71 s in the agent, so sandbox start, not the model
  server, bounds the run (brain 538).
