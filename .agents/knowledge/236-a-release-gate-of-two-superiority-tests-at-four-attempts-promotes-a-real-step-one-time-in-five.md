<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 236. A release gate of two superiority tests at four attempts promotes a real step one time in five

**Lesson:** the source-to-transfer loop's release required the candidate to beat the champion on its own held-out authored
tasks (264 paired slots) and, because the forgetting evaluation declared no purpose and so defaulted to a target, also on the
previous generation's held-out tasks (204 slots), each with a one-sided sign test at 4 attempts per task. For a true
1.25-fold improvement the estimated power is about 0.54 and 0.35, about 0.19 jointly (normal approximation); at 16 attempts
per task one superiority test has about 0.86-0.98. The 8B generation 2 was held back with its forgetting evaluation reading
+0.069 and its own held-out test inconclusive (35 won, 32 lost).

**Fix:** declare a forgetting evaluation's purpose (retention, a non-inferiority bound) so the release does not require two
superiority tests, and size attempts per task to the effect the loop expects a generation to make (16 at a 1.2-1.3-fold step).

**How to apply:** before a gate is registered, compute its power for the smallest step worth releasing, over every test the
release requires jointly; a gate below about 0.8 holds back real progress and re-anchors the next generation on the old champion.
