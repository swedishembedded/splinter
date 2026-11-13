<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 147. A learning rate of 6e-6 for 24 steps moves Qwen3-4B 0.075 on a 400-task dev exam with a KL of 0.0035

**Lesson:** Through the served trainer (brain's training job, step guard, clip-higher) the first-step blow-up of finding 144 is
not seen: at lr 1e-6 to 1e-5 the mean KL to the start of the run stays at or below 0.018 and no step is skipped by the guard.
The recipe is chosen on the dev exam only, which holds no exam item.

## Measured (one arm from the base, dev = 300 generator + 100 proposed programs, base 0.4875, greedy, neutral protocol)

Batch 24 informative groups of 8 attempts (calibration k = 8), 8 records per step, rank 16, clip 0.2 / 0.28, guard KL 0.5,
gradient norm 10, KV f32. The informative share of the groups collected was 0.68 to 0.79.

| lr | steps | dev rate | wins / losses | mean KL | clip fraction | retention losses (of 300) |
|---|---|---|---|---|---|---|
| 1e-6 | 12 | 0.4925 | 11 / 9 | 0.00012 | 0.0001 | 5 |
| 3e-6 | 12 | 0.5325 | 28 / 10 | 0.00054 | 0.0028 | 6 |
| 1e-6 | 24 | 0.5250 | 19 / 4 | 0.00018 | 0.0006 | 1 |
| 3e-6 | 24 | 0.5375 | 29 / 9 | 0.00123 | 0.0051 | 1 |
| 3e-6 | 48 | 0.5550 | 29 / 2 | 0.00151 | 0.0062 | 5 |
| 6e-6 | 24 | 0.5625 | 39 / 9 | 0.00349 | 0.0061 | 0 |
| 6e-6 | 48 | 0.5650 | 42 / 11 | 0.01304 | 0.0145 | 5 |
| 1e-5 | 24 | 0.5550 | 35 / 8 | 0.01790 | 0.0138 | 0 |

One seed per cell: the differences between the upper rows are not resolved. 6e-6 for 24 steps gave the most dev gain at the
lowest KL among the top rows.

## What this does and does not establish

That a single arm moves the dev exam without the numerical trouble of 144 when training goes through the service. Not
established: transfer to the sealed exam (the examiner's), behaviour over three generations, or the serve/train gap of these
rollouts (brain's finding 446 measured 0.0099 mean absolute log-probability gap for this model with an f32 KV cache and 0.0245
with int8).
