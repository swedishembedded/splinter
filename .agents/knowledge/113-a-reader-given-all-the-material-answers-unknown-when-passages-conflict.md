<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 113. A reader given all the material answers unknown when passages conflict

Goal-directed selection was meant to save words. It turned out to decide accuracy: a model handed
every document of a corpus that contains outdated, contradicting and misleading passages abstains.

Real Qwen3.8-27B-FP8 reader, held-out questions never used to select, 3 corpora per form (6 in all):
selective 1.00 on 6/6 corpora (equal to the relevant-only oracle), closed book 0.00, random
same-budget 0.15 / 0.25, everything as it comes 0.00. Giving the same everything each section's
provenance header (authority, date, version) restores 1.00 / 0.92 but at 5x the words. Ablation of
the loop: without the conflict check 0.15 (harmful share 0.75), against 1.00 with it; without the
novelty check the same accuracy with more words; without the objective, lower precision.
Second round on a harder world (paraphrase, near-copies, partial staleness), 5 corpora per cell:
selective + provenance 0.70 [0.64,0.77] vs all + provenance 0.68 [0.62,0.76] at a fifth of the words,
vs everything as it comes 0.12. So selection wins on cost, and the conflict ruling (headers decide:
authority, then date, then version) is the load-bearing component.

Rule: never hand a reader all the material and call it the baseline for "learning from data"; the
fair control is everything with provenance. Rule out conflicts by code before the reader sees them.
Limits: few corpora; no adapter trained on the 27B comparison yet. Evidence: splinter 40e190e53,
2ccdf016d, 230c14039; table in `.agents/roadmap/acquisition.md`, evidence kept outside the repo.
