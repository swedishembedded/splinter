<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 122. Words only one probe asks with outweighed the topic they were asked about

Relevance of a section to a gap was the share of the gap's words (rarity-weighted) the section has. On the toy cipher world (12 dev probes, `What is the cipher of aaa?` and so on) every cipher table scored relevance 0.15, under the 0.5 floor, so the loop chose no section: selective held-out exact match 0.00, oracle 0.46 (`tests/methods/selection.rs`, first run). Cause: the probes' prompts (`aaa`, `aab`...) also occur in the corpus as cipher *values*, so each is a rare word with weight 11 and support 1/12; twelve of them summed to about 0.9 against 0.11 for the topic word `cipher`.

Rule: in a gap of three or more probes a word that fewer than two probes use says which probe it was, not what the gap is about, and is dropped from relevance. After the change the loop chose exactly the 25 current tables (precision 1.00, recall 1.00) and held-out went 0.00 to 1.00 on the toy model. Evidence: splinter 230c14039, `crates/knowledge/src/goal/select.rs`.
