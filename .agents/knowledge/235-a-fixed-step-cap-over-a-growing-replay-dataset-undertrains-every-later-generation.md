<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 235. A fixed step cap over a growing replay dataset undertrains every later generation

**Lesson:** the source-to-transfer loop trains each generation's LoRA fresh from the base on a dataset that grows with every
generation (the new records plus all earlier generations' as replay), with a fixed cap of 800 optimizer steps of 4 records.
The cap turns into fewer passes over the data at every generation. Measured at the selected step (validation-loss early
stopping, patience 4, every 40 steps):

| LoRA | Training records | Selected step | Epochs | Stopped by |
|---|---|---|---|---|
| Qwen3-4B generation 1 | 1100 | 520 | 1.89 | validation plateau |
| Qwen3-4B generation 2 | 3134 | 760 of 800 | 0.97 | the cap, still improving |
| Qwen3-4B generation 3 at the cap | 5045 | 800 | 0.63 | the cap |
| Qwen3-8B generation 1 | 1082 | 520 | 1.92 | validation plateau |
| Qwen3-8B generation 2 | 3299 | 800 | 0.97 | the cap |

Generation 2 did not beat generation 1 at either size (`.agents/roadmap/source-to-transfer.md`); half the passes per record
is one of the measured differences.

**Fix:** `model.train` takes `params.epochs` in place of `steps` (`docs/guides/training.md`, Parameters): the factory
resolves it into `ceil(epochs * draws / records_per_step)` steps, the draws of an epoch being the dataset's training records
plus the replay (the plain union) or the dataset's over `1 - replay_share`, and a step `grad_accum` records on each
data-parallel rank. The resolved budget is part of the stage identity and is written into `report.json` and
`training.json`. Use `E` the first generation's effective epochs at its selected step (4B 1.891, 8B 1.92); validation-loss
early stopping stays on; no other hyperparameter changes; exams play no part. The out-of-tree `gen-scaled.py` is retired.

**How to apply:** any loop that retrains from the base on a cumulative dataset must size its budget in epochs, not steps, and
record the epochs at the selected step in each generation's report, so a cap that binds shows before the generation is
examined.
