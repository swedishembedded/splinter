<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 114. Selecting material by words and by meaning each miss what the other finds

Relevance to a goal was first lexical (shared words with the open gaps). On the hard world
(paraphrased current documents, header-less near-copies, decoys), scripted reader, 12 corpora per
cell: lexical selection recalled 0.75 of the sections of a client-behaviour form, semantic (an
embedder standout over the corpus) 1.00; on an equipment-property form with invented topic names it
was the reverse, lexical recall 0.98 against semantic 0.72, because an embedder knows nothing of an
invented name. The fused score keeps both recalls and pays precision (decoy and stale sections get
in). Served Qwen3-Embedding-0.6B raised behaviour recall from 0.80 to 0.95 against the local one;
the held-out score was unchanged at n=5.

Rule: select by both signals fused, never by one, and judge a selector on a corpus of each kind (an
invented-name form and a paraphrase form). Evidence: splinter e7c204c3b, ledger `acquisition.md`.
