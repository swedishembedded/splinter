<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 103. A sign test's p-value was read from the reverse tail

The three-decoding-seed protocol read of D46 was reported as "decisive wins 20 families against
the base's 13 (13 tied), one-sided p=0.081". The exact one-sided sign test for 20 wins in 33
discordant families is **P(X>=20 | n=33, p=1/2) = 0.148**; 0.081 is **P(X>=21)**, which is also
`1 - P(X>=13)` - the value `protocol-summary.py` prints when the same two arms are named in the
opposite order, inverted by hand. The correct reading of the three-seed data is 20W-13L-13T at
p=0.148; the 0.081 figure came from the summary's other direction, not from a stronger result.

The exact values, for checking: P(X>=19)=0.2434, P(X>=20)=0.1481, P(X>=21)=0.0814 at n=33. The
other comparisons reported the same day are correct - 19W-15L gives 0.3038, 12W-12L 0.5806,
16W-1L 0.000137, 8W-4L 0.1938, 3W-13L 0.9979 (i.e. 0.0021 the other way) - so the error is not
in the counting or in the script but in reading one line of it.

Two guards, both cheap: name the hypothesis in the printed line ("A greater than B"), and read
the tail from the arm that is the candidate. A p-value that halves a result's significance is
exactly the kind that gets quoted, so the summary now prints both arms' directions and the
per-family table behind them (`runs/g8/protocol-summary.py`).

Evidence: `runs/g8/protocol-summary.py dec-proto base-proto` prints
"20W-13L-13T, exact one-sided p=0.148103"; the same command with the prefixes swapped prints
"13W-20L-13T, p=0.918622". Recorded in `runs/plan-2026-10-18/NEXT-SESSION-PROMPT.md`'s
successor and `~/code/progress/end-to-end-coding-model.md` (lease 14 section).
