<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 125. A corpus ordered current-first made a first-wins reader look right

In the first synthetic corpora the current manual, then the old one, then the forum copy were written in that order. A reader that takes the first passage that states the fact (and a persona built by concatenating statements in corpus order) scored 1.00 on the `all` arm, hiding that reading everything is harmful: after shuffling the documents with the corpus seed the same arm scored 0.00 to 0.50 (client-behaviour form, fidelity of the persona). Rule: a generated corpus must not carry the answer in its order; shuffle documents and test the reader against the shuffled corpus. Evidence: `crates/knowledge/src/goal/synthetic.rs` (`w.rng.shuffle(&mut documents)`), splinter 703a4a560.
