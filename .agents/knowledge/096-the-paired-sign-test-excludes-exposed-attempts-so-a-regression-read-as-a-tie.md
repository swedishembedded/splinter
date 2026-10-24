<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 096. The paired sign test excludes exposed attempts, so a regression read as a near-tie

On the 46-family development benchmark D46 (one task per family, greedy decoding), a candidate
verified 15 of 46 tasks while its own base verified 21, and it failed the hidden checks after
passing the visible ones (exposed) on 19 attempts against the base's 9. The paired comparison
of the same two sets reported 23 paired families, 5 wins, 3 losses, 15 ties, p = 0.363 - a
near-tie by the registered statistic, and a 6-task regression by pass@1.

Both readings are what the tool says. `experiences compare` leaves exposed and harness-failed
attempts out of the pairing, as it documents ("Exposed and harness-failed attempts are left out
and counted"). An attempt that passes the visible tests and fails the hidden ones has no
decision, so its family cannot pair. A candidate that turns correct fixes into plausible ones
therefore removes its own worst outcomes from the paired test: 19 of the candidate's 46
attempts never enter it, against 9 of the base's, which is why 46 tasks pair only 23 times.

The competence block beside it does not correct the difference either: quality is
verified/attempted-among-graded, 15/27 = 0.556 against 21/37 = 0.568, so the two candidates
look alike once the exposed attempts are removed from both denominators too.

Consequence for a release gate: the paired sign test alone must not carry the decision. A
promotion needs pass@1 over all attempts and the exposed count and rate reported beside it, and
a candidate whose exposed rate rises must be refused regardless of its paired result. On this
measurement the adapter is a regression that the registered statistic cannot see.

Evidence: `runs/g8/pilot-d46-base.verify.json` (passed 21, exposed 9),
`runs/g8/pilot-d46-pc1-.verify.json` (passed 15, exposed 19), and
`runs/g8/pilot-compare-d46-pc1-.json` (paired 23, 5W-3L-15T, p = 0.363).
