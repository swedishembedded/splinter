<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 166. A small model trained only on made-up names could not copy a real-word name from its context

**Lesson:** copying is learned for the strings training shows. When every name a model copies in
training comes from one generator (made-up syllable words), it copies those and nothing else: it writes
a name of that generator's shape instead of the one in front of it. Draw names from every kind an
examiner may use (made-up words, random letters, natural words) before blaming capacity or the objective.

Found 2026-11-17 in minimind (default profile, seed 7, 5M-param Ouro, brain 2784b15ab, sven 0d7d6643,
bench's sealed workspace-tools exam): with requests, sampling and protocol identical to the exam
(knowledge 165), the model's exam episodes listed the workspace (`ls -R`) and then read a file that the
listing did not show in all 1041 of its reads (v13), writing a made-up word where the task named a real
one. Half of all names as random letters (v14) gave the first non-zero tiers (0.03); adding a third
drawn from the common words of a fetched English corpus (TinyStories, 22 MB; v16) raised the pretrained
checkpoint to 6 of 12 development and 5 of 12 held-out probes and 0.13 on tier a. Scaling width to
hidden 384 at the same tokens (v12) changed nothing measurable.

A second finding in the same runs: the fine-tune undid what pretraining taught. Over 6000 steps at lr
3e-4 on about 20000 records the development pass rate fell from 0.50 (1000 steps) to 0.08 and the exam to
0; 1000 steps at 1e-4 kept 6 of 12 and 0.04/0.03 on tiers a/b. At lr 5e-5 the fine-tune under-fit (loss
stayed near 0.2). Measure the probe after each fine-tune segment and stop where it peaks.
