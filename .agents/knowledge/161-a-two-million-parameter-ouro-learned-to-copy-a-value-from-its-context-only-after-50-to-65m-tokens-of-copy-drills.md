<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 161. A two-million-parameter Ouro learned to copy a value from its context only after 50 to 65M tokens of copy drills

**Lesson:** in-context copying (a code out of a listing, a file name from a request into a call) is a
capability a small model acquires abruptly, after a token count that a quick run does not reach; below
it, more fine-tuning, rehearsal or better formatting moves only the tasks that need no copying. Measure
the capability directly while pretraining (exact match of a copied value on held-out prompts) and size the
pretraining past the jump before judging any later stage.

Found 2026-11-16 with the minimind retrieval probe (`samples/minimind/src/retrieval.rs`, 12 listing and 12
request-to-path prompts) on gpu3, brain 16c2db09e. The smoke-size Ouro (hidden 256, 4 layers, 2 passes,
vocabulary 1024 learned without digits) pretrained on a world of 100000 copy drills only, 2000 steps of
64 x 512 tokens in four segments: retrieval 0 of 24 after 16, 33 and 49M tokens, then 10 of 24 (4 listing,
6 path) after 65M. The development probes of every smoke lever (about 20M pretraining tokens, a fifth of
them drills) stayed at 0 of 8 for the tasks that need copying. The 4-pass, vocabulary-4096 default model
reached 1 of 24 at most within 66M mixed tokens at 23k tok/s (the 2-pass model trains at 50k tok/s).
