<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 124. Phrase embeddings alone cannot tell a paraphrase from a neighbouring attribute

Served Qwen3-Embedding-0.6B cosine: `service interval` and its paraphrase `maintenance period` 0.66, but `maintenance period` and the different attribute `warranty period` 0.72; `the rent is late` and `payment is overdue` 0.68, but `payment is overdue` and `an overdue notice arrives` (another situation) 0.79. A threshold cannot separate them. What worked for the property form: greedy one-to-one pairing, strongest first, forbidding a pair that any document uses together (a document that states two things in two phrasings means two things): 4 of 4 attributes right, 0 wrong merges over 3 corpora. For client-behaviour keys (situations and type names that no document uses together) it did not: on corpora with no paraphrase 4 keys were still merged wrongly, and on 5 real corpora (seeds 6000-6004) alignment lowered held-out from 0.52 (unaligned) to 0.40, recall from 0.78 to 0.60, because a false merge turns two different facts into a conflict that a header then rules. A guard that drops unsettled flags across an alignment was tried and was worse (0.28).

Rule: alignment by phrase similarity is safe only where a structural check (co-occurrence) can refuse the wrong pairs; where it cannot, leave the keys apart and let a provenance-aware reader settle the disagreement. Evidence: `crates/knowledge/src/goal/align.rs`, `~/project/shared/goal-selection/round3/` (behaviour-aligned, behaviour-unaligned, behaviour-guard-tried-worse).

Update: the structural check that works for identified things and fails for classes is finding 135.
