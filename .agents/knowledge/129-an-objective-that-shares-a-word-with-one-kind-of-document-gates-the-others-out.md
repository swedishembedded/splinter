<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 129. An objective that shares a word with one kind of document gated the others out

The goal's objective names its domain: a section that mentions none of its rare words is capped below the relevance floor. For the multi-hop corpora the objective was `Service the installed equipment correctly.`; `installed` occurs only in the site registers, so the unit manuals were off-domain and the loop learned the registers (30 sections) and then nothing: held-out 0.00 with and without gap expansion. With `Service the equipment correctly.` (no word that one kind of document has) the expansion found the manuals and the pass rate rose 0.00, 0.06, 0.33, 0.56, 0.61 over rounds. Rule: an objective's words gate, so they must be the domain's, not a vocabulary of the documents that happen to be found first. Evidence: `crates/knowledge/src/goal/synthetic.rs`, test `a_learned_link_opens_the_gap_for_the_facts_it_names`.
