<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 237. Validation drawn from the new generation alone selects checkpoints that imitate the champion

**Lesson:** `model.train` takes its validation split only from the stage's `dataset`; replay datasets contribute none. In the
source-to-transfer loop the new generation's dataset mixes the teacher's records with the champion's own, so validation
followed that mix: 41% of the generation-2 validation characters are the champion's (3% at generation 1). Checkpoint selection
by validation loss then rewards imitating the champion, whose records are shorter and simpler (34% exactly two calls, 48%
without an error, against 8% and 13% of the teacher's). It did not bind at generation 2 (loss still falling at the step cap);
it binds once the budget allows convergence.

**Fix:** project teacher and champion records separately and make the teacher-only projection the training `dataset` (so
validation is teacher-only), with the champion's records in replay.

**How to apply:** check what the validation split is drawn from whenever a training set mixes policies; selection follows the
validation distribution, not the training one.
