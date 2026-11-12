<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 117. Best-of-N sampling adds little to greedy decoding of an under-confident model

On the toy checkpoint (a three-letter cipher, held-out exact match, CPU) greedy decoding scored 0.39 and
one draw at T 0.6 scored 0.10. Best-of-16 sampling against a check the model can run itself (the cipher
undone must give the prompt back, no answer key) scored 0.35, below greedy; greedy plus best-of-48 scored
0.46; the same at T 1.0 scored 0.41. The prompts greedy gets wrong are the ones whose right answer the
model gives almost no probability, so more or flatter draws find few of them.

Rule: compare a search against greedy decoding of the same model, with the greedy answer among the
candidates; a search compared with a single sample overstates its gain (0.10 to 0.35 there). A search
that must win needs a proposer that can reach answers sampling does not (iterative repair); that is not
built. `search.rs` in `tests/methods`; the bar was lowered from +0.10 to +0.05 after the +0.07 result
and the row says so.
