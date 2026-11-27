<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 243. One failed item of a foreach stopped a three-generation authoring run

**Lesson:** a `foreach` stage failed on its first failed item and then started no item, and the stage engine started
no other stage. One of 264 proposal items of the source-to-transfer authoring run failed with "no response headers
from brain after 600 s" (a slow first byte from the 27B author) and failed `proposed-g2` and `proposed-g3`, stopping
the run. The model-call retry (`crates/agent/src/retry.rs`) gives up after about 7.5 s and does not retry a timeout,
too short for a server restart or a slow first byte.

**Fix:** an item that fails transiently (a timeout, a dropped or refused connection, a 5xx answer) is asked again,
up to 5 times, 30 s, 60 s, 120 s and 240 s apart (deterministic, cancellable). A failed item stops nothing else:
every other item runs and commits, and the stage then fails naming the failed items, which a resume computes again
(only them). `tolerate_failed_items: N` (or a share such as `0.01`) on a `foreach` stage records up to that many
failed items as rejected (`failed: ...`) instead; the default stays strict. Neither changes an item's identity.
Spec: `crates/factory/tests/it/agent_op.rs`.

**How to apply:** a fan-out over hundreds of model calls must treat a transient failure as the normal case: retry
with a budget sized to the service's recovery time, and let one failure cost one item, not the run.
