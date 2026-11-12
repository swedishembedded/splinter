<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 123. A fact key grounded in its quote never meets its paraphrase, so stale statements were chosen

The knowledge-object admission requires a fact key's words to be in the quote it rests on, so `The maintenance period of alder pump 3 is 250 hours` and `The service interval of alder pump 3 is 400 hours` are two facts, and no header ruling ever compared them. On the hard world (half the current documents paraphrased) the selector chose the superseded or contradicted sections whose current counterpart was paraphrased: 30 outdated sections of 5 corpora (scripted), real 27B reader: selective recall 0.71 and precision 0.71 with the budget (four rounds of 300 words) spent on them, held-out 0.54 (seeds 3000-3004).

Fix: align the keys (an embedder proposes, code checks: cosine at least 0.6, mutual best by a margin, no document uses both phrases, no digit in an entity name) and let the documents' headers and the stated values rule across the alignment (`objects::resolve_aligned`). Scripted, 12 corpora: selective held-out 0.74 to 1.00 (property), 0.79 to 1.00 (behaviour). Real reader and served Qwen3-Embedding-0.6B, fresh seeds 6000-6004, property: aligned selective 0.97 [0.95,1.00], precision 0.99, recall 0.99; the same corpora unaligned 0.70 [0.54,0.89], precision 0.84, recall 0.84. Evidence: splinter f876c910a, ledger `acquisition.md`, `~/project/shared/goal-selection/round3/`.
