<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 100. At this budget rehearsal protected nothing and cost nine tasks

Generation 2 continued the decisive seed-1 champion (D46 23/46, S1 16/23) on S1's verified
successes (`blake3:4f9ba02d`, 25 records) for 120 steps at lr 5e-5, seed 1, on the packed-bf16
base. The two arms differ only in `--rehearsal blake3:92252209 --rehearsal-share 0.25` (615 fit
records, 154 monitored):

| arm | benchmark | pass@1 | exposed | paired against the base, exact one-sided sign test | training wall |
|---|---|---|---|---|---|
| no rehearsal | D46 | **28/46** | 7 | 32 paired, 8W-4L-20T, p = 0.19384765625 | 2299 s |
| rehearsal | D46 | **19/46** | 5 | 34 paired, 4W-7L-23T, p = 0.88671875 | 2538 s |
| no rehearsal | S1 | 12/23 | 5 | 13 paired, 2W-2L-9T, p = 0.6875 | - |
| rehearsal | S1 | 11/23 | 1 | 15 paired, 1W-0L-14T, p = 0.5 | - |

The direct same-start contrast on D46, rehearsal against no rehearsal, pairs 37 families with
3 wins, 13 losses and 21 ties (p = 0.9979095458984375 with rehearsal as the candidate; the other
direction, 13W-3L, is p = 0.0106). The rehearsal arm is nine tasks worse than the same update
without it.

At this budget the new corpus did not make the model forget D46 - the no-rehearsal arm adds five
tasks over its 23/46 start - so there was nothing for rehearsal to protect, and it cost nine
tasks instead. Acquisition on S1 over the champion's 16/23 is not demonstrated by either arm (12
and 11), although both are above the S1 base of 8. The reading is a development one: the corpus
is a new benchmark, the arms are one seed each and greedy, the comparisons are against the same
base and share the 46 families, so they are not independent tests. The nine-task gap is larger
than the run-to-run variation a single greedy arm carries (knowledge 101 measures about 2.5
tasks of standard deviation on D46), but the arms themselves are single runs.

What this establishes: with a new, unrelated corpus at this budget, rehearsal on an earlier
corpus bought no retention and cost tasks. What it does not: that rehearsal is harmful in
general, or that the no-rehearsal update is better than the champion - its own comparison with
the base is p = 0.194 and the S1 comparison p = 0.6875.

Evidence: `runs/g8/pilot-gen2-gen2-noreh.verify.json` and `runs/g8/pilot-gen2-gen2-reh.verify.json`
(the D46 counts), `runs/g8/pilot-gen2s1-gen2-noreh.verify.json` and
`runs/g8/pilot-gen2s1-gen2-reh.verify.json` (S1),
`runs/g8/pilot-compare-gen2-gen2-noreh.json` (32 paired, 8W-4L, p = 0.19384765625),
`runs/g8/pilot-compare-gen2-gen2-reh.json` (34 paired, 4W-7L, p = 0.88671875),
`runs/g8/pilot-compare-gen2s1-gen2-noreh.json` and `-reh.json` (S1),
`runs/g8/compare-gen2reh-vs-noreh.json` (37 paired, 3W-13L, p = 0.9979095458984375), and
`runs/ce8/logs/train-gen2-noreh.time` (2299 s) and `train-gen2-reh.time` (2538 s).
