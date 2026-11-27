<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 242. A retention gate with no counted pairs passed as retained

**Lesson:** the release's retention gate read only `wins`, `losses`, `difference` and `alpha` of a development plan's
sign test: `retained` unless the loss was significant or the point difference was below `-retention_margin`. An
evaluation with zero pairs (every slot uncounted) has no wins, no losses and a difference of 0, so it was
`retained`; so was one with most of its slots uncounted. `max_unpaired_share` applied only to the target path, so a
release could promote on a retention suite that measured nothing.

**Fix:** retention is `inconclusive` with no counted pair, and when the uncounted share exceeds `max_unpaired_share`
(any uncounted slot when it is unset, the target rule); a significant loss or a point difference below
`-retention_margin` on the pairs that were counted is still `regressed`. Spec: `crates/factory/src/run/release.rs`,
`retention_spec`.

**How to apply:** a non-inferiority gate must refuse absence of evidence explicitly; "no loss found" passes
vacuously on an empty sample. Hold every purpose of a gate to the same completeness rule.
