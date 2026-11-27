<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 244. A generation whose own records come only from tasks it already passes stalls

**Lesson:** a loop's records did not explain its stall until its state was read per generation. `splinter observe`
(seed 7) on the Qwen3-4B source-to-transfer loop (generations 1-3) and the Qwen3-8B one (generations 1-2), read-only:

| Generation | Own share of training records (validation) | Own records from tasks the champion fails | Own / teacher pass rate on the new tasks | Frontier share | Epochs at the selected step | Own held-out difference [95% interval] | Exam difference |
|---|---|---|---|---|---|---|---|
| 4B 1 | 0.078 (0.092) | 0 of 98 | 0.031 / 0.728 | 0.972 | 1.891 (stopped early) | +0.176 [+0.108, +0.255] | +0.043 |
| 4B 2 | 0.399 (0.397) | 0 of 904 | 0.223 / 0.674 | 0.897 | 0.970 (the 800-step cap) | +0.034 [-0.030, +0.098] | +0.005 |
| 4B 3 | 0.408 (0.398) | 0 of 862 | 0.227 / 0.663 | 0.887 | 1.299 (stopped early) | +0.033 [-0.042, +0.111] | +0.005 |
| 8B 1 | 0.063 (0.085) | 0 of 79 | 0.029 / 0.728 | 0.972 | 1.922 (stopped early) | +0.245 [+0.152, +0.338] | +0.055 |
| 8B 2 | 0.449 (0.462) | 0 of 1115 | 0.275 / 0.674 | 0.896 | 0.970 (the cap) | +0.011 [-0.061, +0.083] | +0.018 |

From generation 2 on, about 40% of what the candidate imitates (and of the validation that picks its checkpoint) is the
champion's own behaviour on tasks it already solves at least sometimes; every record of a task the champion fails is a
teacher's (555 of 2266 records at 4B generation 2, 460 of 2117 at generation 3). The new tasks are as hard for the
teacher as at generation 1 (0.67 against 0.73) and far harder for the champion than the old ones were for the student
it replaced, so the frontier did not move with the champion. The previous generation's held-out set read higher
(+0.049 and +0.086 at 4B, +0.069 at 8B) than the generation's own.

**How to apply:** read `splinter observe` before a generation is examined. When the champion's own share is large and
none of it comes from the tasks it fails, its records teach what it can already do: steer the frontier (keep the
records of tasks it fails or only sometimes passes, `.agents/roadmap/loop-control.md`), and check the epochs at the
selected step, which a step cap halves as replay grows (knowledge 235).
