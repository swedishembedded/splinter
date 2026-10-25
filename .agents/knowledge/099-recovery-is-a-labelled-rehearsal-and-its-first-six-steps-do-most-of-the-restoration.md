<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 099. Recovery is a labelled rehearsal, and its first six steps do most of the restoration

The all-action adapter `win1` had been damaged to 12/46 on D46 (knowledge 097). The recovery
corpus is the base's own verified successes on the same 46 families (`blake3:df000e8f`,
14 records), trained at lr 5e-5 from the damaged adapter, and measured on D46 at greedy decoding:

| arm | steps | pass@1 | exposed | paired against the base, exact one-sided sign test | training wall |
|---|---|---|---|---|---|
| rcv-win1 | 6 | **23/46** | 2 | 36 paired, 4W-6L-26T, p = 0.828125 | 303 s |
| rcv-win1 | 18 | 20/46 | 5 | 34 paired, 6W-8L-20T, p = 0.78802490234375 | (measured 2026-10-24) |
| rcv-win1 | 36 | **28/46** | 7 | 31 paired, 10W-5L-16T, p = 0.15087890625 | 815 s |
| undamaged control (decisive seed 1) | 18 | 24/46 | 10 | 30 paired, 7W-2L-21T, p = 0.08984375 | 578 s |

The first six steps - about five minutes of training - recover the damaged adapter from 12/46 to
23/46, which is the base's own level (21/46) and nominal parity with the undamaged decisive
adapter (23/46). Longer continuation on the same corpus buys more tasks (28/46 at 36 steps) but
raises exposed attempts from 2 to 7, and no arm's paired comparison reaches the registered
alpha. The undamaged control moves a solved 23/46 to 24/46 while its exposed count rises to 10,
so the gain at 36 steps is not separable from the corpus: it is the same 46 families the
benchmark scores, a labelled rehearsal of the base's own successes, not a transfer measurement
and not free gain.

What this establishes: within this dose series, most of the restoration happens in the first six
steps, and further steps on the benchmark's own families trade exposure for a nominal task
count. What it does not: the doses are one seed, share the same start, corpus and families and
are not independent tests; the 18-step point was measured on 2026-10-24 under the same corpus
and recipe; the 36-step arm's 28 against the base's 21 is not significant (p = 0.151) and
carries 7 exposed attempts; and no arm is the registered confirmatory query. The exposed counts
are reported beside the paired test for the reason knowledge 096 gives: a paired test that
leaves exposed attempts out can read a regression as a tie. That the damage came from all-action
supervision rather than from learning new data is knowledge 097's result; this entry says what
it takes to undo it.

Evidence: `runs/g8/pilot-rcvA-rcv-win1-6.verify.json` (23 passed, 2 exposed) and
`runs/g8/pilot-compare-rcvA-rcv-win1-6.json` (36 paired, 4W-6L, p = 0.828125);
`runs/g8/pilot-d46-recov.verify.json` (20 passed, 5 exposed) and
`runs/g8/pilot-compare-d46-recov.json` (34 paired, 6W-8L, p = 0.788);
`runs/g8/pilot-rcvA-rcv-win1-36.verify.json` (28 passed, 7 exposed) and
`runs/g8/pilot-compare-rcvA-rcv-win1-36.json` (31 paired, 10W-5L, p = 0.15087890625);
`runs/g8/pilot-rcvA-rcv-decisive-18.verify.json` (24 passed, 10 exposed) and
`runs/g8/pilot-compare-rcvA-rcv-decisive-18.json` (30 paired, 7W-2L, p = 0.08984375);
training walls and recipes in `runs/ce8/logs/train-rcv-win1-6.time` (303 s) and
`runs/ce8/logs/train-rcv-win1-36.time` (815 s).
