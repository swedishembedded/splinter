<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 134. Value coincidence is no evidence that two phrases are one

Two ways to demand more evidence than phrase similarity before aligning fact keys, both measured with served Qwen3-Embedding-0.6B on 5 hard corpora (seeds 7000-7004). (1) Require that the two attribute phrases state the same value about the same entity: only 2, 5, 2 and 0 of 4, 20, 28 and 16 paraphrased keys were aligned, because in the corpus the paraphrased current document is the only copy of the true values (the old edition has stale ones, the forum copy wrong ones), so a true pair has nothing to agree on. (2) Align whole keys (`attribute | entity` as one text) with agreeing values and a count of linking statements: 5 to 9 wrong merges per corpus, all between topics' units with the same number and the same attribute (`maximum pressure | noisk fan 1` with `maximum pressure | habalt valve 1`), because small value ranges collide and two such collisions between the same documents satisfied the support rule. In the client-behaviour form, where values come from a closed set of 16, coincidence is common.

Rule: value agreement corroborates a pair only where the truth is stated twice; do not make it a precondition, and never let it be the only check against the wrong merges. Evidence: the diagnosis runs of 2026-11-12 (`--alignment-only` of `goal-selection`), `crates/knowledge/src/goal/align.rs`.
