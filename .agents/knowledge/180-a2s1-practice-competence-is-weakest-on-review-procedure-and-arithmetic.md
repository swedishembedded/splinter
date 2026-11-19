<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 180. A2s1's practice competence is weakest on review, procedure and arithmetic, not on the domains

**Lesson:** read competence per capability of bench's taxonomy, not per collection. On A2s1's G=8 practice the
domains (support, compiled, legal, accounting) mix capabilities the policy has nearly mastered with ones it fails
more than half the time; three capabilities (apply-rules, locate-in-workspace, structured-answer) label almost every
task of these collections and only restate the overall rate. The narrow ids are the ones that tell a curriculum where
to spend: design-review and professional-procedure 0.40, exact-arithmetic 0.44, against abstention 0.89 and
stateful-service 0.87.

Found 2026-11-19 (lane K): `splinter experiences capabilities` over lane R's five G=8 practice runs of A2 seed 1
(1280 trials, 160 tasks, frozen brain, adapter blake3:762427...), labelled with `--labels` from the same four draws
re-staged by bench at 56b6502c with capability labels (content_sha256 equal to lane R's for all four; all 160
source digests equal). Graded passes / graded, 95% Wilson interval, bands frontier/cliff/retired, weight:

| capability | tasks | passed / graded | rate [95%] | F/C/R | weight |
|---|---|---|---|---|---|
| design-review | 21 | 67 / 168 | 0.399 [0.328, 0.474] | 13/5/3 | 0.486 |
| professional-procedure | 21 | 67 / 168 | 0.399 [0.328, 0.474] | 12/8/1 | 0.471 |
| exact-arithmetic | 65 | 228 / 520 | 0.438 [0.396, 0.481] | 37/18/10 | 0.440 |
| locate-in-workspace | 120 | 454 / 958 | 0.474 [0.442, 0.506] | 57/34/29 | 0.388 |
| structured-answer | 120 | 454 / 958 | 0.474 [0.442, 0.506] | 57/34/29 | 0.388 |
| apply-rules | 160 | 732 / 1278 | 0.573 [0.545, 0.600] | 83/34/43 | 0.325 |
| stateful-service | 40 | 278 / 320 | 0.869 [0.827, 0.901] | 26/0/14 | 0.110 |
| abstention | 26 | 185 / 208 | 0.889 [0.840, 0.925] | 9/0/17 | 0.077 |

Two trials were stopped by the gateway and are in no denominator. Most failures are `wrong_content` (353 of the
apply-rules failures), then `resource` (88) and `tool` (65); `resource` concentrates in professional-procedure
(40 of 101 failures) and exact-arithmetic (71 of 292). Dev-set and sealed results are not implied: this is the
policy's own practice material.
