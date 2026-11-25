<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 220. brain writes every step figure as a float

**Lesson:** a streamed step's `pg` map is brain's metrics map, where every figure is an `f32`: `policy_tokens`,
`staleness_max` and `dropped_stale` arrive as `64.0`, never `64`. The learner read them with `as_u64`, which is
`None` for a float, so every step reported 0 tokens: the report's `tokens_trained`, `experience_utilisation` and
`useful_tokens_per_second` were 0 after five trained steps, and the controller's step estimate (the token
allowance) never learned from a report. The scripted double wrote integers, so no test saw it. A count is now read
as a non-negative whole number in either spelling, and the double writes brain's floats. Spec: splinter
`crates/factory/tests/it/online/brain.rs` (`trained_tokens` 20 over two steps of 10) and
`crates/model/tests/served_stream.rs`.

## Evidence (2026-11-25, lane agrpo-e2e2)

The CPU stack's first complete run: `groups.committed` 5, `learner.steps` 5, `learner.trained_tokens` 0,
`tokens_trained` 0; with the fix a later run reported 2601 tokens trained of 9580 generated.
