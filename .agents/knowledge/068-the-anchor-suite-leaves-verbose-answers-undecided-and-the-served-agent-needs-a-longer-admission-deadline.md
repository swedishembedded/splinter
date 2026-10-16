<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 068. The anchor suite leaves verbose answers undecided, and the served agent needs a longer admission deadline

Measured 2026-10-16 with splinter 6ca8b3f, brain 465448ae for `brain serve`, one GH200.

Anchor-200 frozen as version 1, digest
`blake3:88b3abdb84dadc7fa16bdce36b1737799c7d70809c8b19872e624fa99b9c0035` (200 tasks).
`splinter eval --suite anchor` without a judge decides only the answers the code
verifiers can: base Qwen3-1.7B decided 169 (164 right), Qwen3-4B decided 179 (175 right),
the served 27B Q8_0 decided 119 (118 right). The rest are undecided, not wrong: the 27B
answers "There are **3,000** millilitres" and "The chemical symbol ... is **K**". The
reported accuracy (97.0, 97.8, 99.2 %) is over decided tasks; passed over 200 is 82.0,
87.5 and 59.0 %. An admission threshold of 70 % is therefore decided by the reading, not
by the model, for the 27B.

P1-THRU, 10 exprcalc development tasks, one attempt each, `solve` with `remote:brain/...`:
Qwen3-8B 351 s at concurrency 1 and 425 s at 4 (0 of 20 verified, no edits made); the 27B
Q8_0 2191 s at 1 and 903 s at 4 (8 of 20 verified, 13 of 20 hit the sandbox wall), card
peak 15.5 GiB and 52.9 GiB. `brain serve` answered 429 "request could not be admitted
within the deadline" to the third and fourth concurrent agent request until
`BRAIN_ADMIT_DEADLINE_MS` was raised; no log line is written.

How to avoid it: give the anchor gate a judge or a terse-answer prompt before reading its
accuracy, and report decided and undecided counts together. Raise the admission deadline
when serving an agent at concurrency above 2.
