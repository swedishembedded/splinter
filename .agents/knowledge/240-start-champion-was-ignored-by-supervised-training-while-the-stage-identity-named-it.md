<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 240. `start: champion` was ignored by supervised training while the stage identity named it

**Lesson:** `model.train` with `start: champion` on an `sft` or `cpt` dataset trained a fresh adapter from the base.
The brain trainer passed `request.start` to brain only for preference, feedback and policy-gradient jobs; the
supervised `FineTune` was built with `continue_from: None`. The stage identity includes the start's digest, so the
recipe, the lock and the run all read as a continuation of the champion while the job never received the adapter
(brain's job document had no `start_adapter`). The documentation stated the limitation, so nothing warned.

**Fix:** the supervised job continues `request.start` like every other regime (brain receives `data.start_adapter` and
no rank or alpha, keeping the adapter's). Spec: `crates/factory/tests/it/brain_trainer.rs`,
`a_variant_is_a_resumable_brain_job_whose_adapter_is_served_before_it_counts`.

**How to apply:** a parameter that is part of a stage's identity must reach the computation in every branch that
computes the stage; test each branch's request for it, not only the identity. Candidates trained before this fix
with `start:` on supervised data were trained from the base.
