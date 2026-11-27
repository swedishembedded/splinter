<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 241. `champion:RUN` read the stage called `release`, whatever generation it was

**Lesson:** a champion was always read from `RUN/release/release.json`. A loop recipe holds one promotion per
generation (`release`, `release-g2`, `release-g3`), so a champion bound to a generation-2 run that promoted at
`release-g2` loaded generation 1's champion: the next generation would have been measured against, and continued
from, the older adapter, with nothing in the lock to show it.

**Fix:** `champion:RUN` resolves to the release the run's recipe declares as `outputs.release`, else its only
release, else its newest release that promoted a candidate (by the product's `created_at`); a run of several
releases none of which promoted is refused, and `champion:RUN#STAGE` names a release explicitly. A run of one
release reads the same bytes as before, so its champion digest and every lock that pinned it are unchanged
(checked on the source-to-transfer runs `loop-g1`, `loop-g1-8b`, `loop-g1-8b-rel`). Spec:
`crates/factory/tests/it/champion_release.rs`.

**How to apply:** a reference to "the result of a run" must say which product when a run can hold several of that
kind; resolve it from what the run declares, never from a conventional key.
