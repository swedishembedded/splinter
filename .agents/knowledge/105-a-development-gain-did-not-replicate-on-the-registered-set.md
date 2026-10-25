<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 105. A development gain did not replicate on the registered set

Two reads of the same supervision contrast, one step apart:

| read | benchmark | per family | p |
|---|---|---|---|
| development, six decoding seeds | D46 (46 families, drawn during development) | 22W-11L-13T | **0.0401** |
| registered confirmatory query, three registered seeds | qualification set `blake3:f7fe502f` (46 families, frozen before any candidate existed) | 18W-12L-16T | **0.1808** |

Mean pass@1 moved the same way in both (19.67 to 24.50 on D46; 20.0 to 23.0 on the qualification
set) and the behavioural signature repeated almost exactly (exposed attempts 10 of 138 for the
candidate against 35 of 138 for the base). What did not hold was the significance: the
development set, chosen while the system was being tuned on it, gave a p one step away from the
registered set's.

Two consequences worth keeping:

- **The registration's own thresholds matter, and they are not all about alpha.** It says a
  development gain with no qualification gain at $p>0.2$ is reported as memorisation. This read
  is $p=0.1808$: it meets neither the registered alpha nor the memorisation rule, so the honest
  label is a null whose direction matches the development read. Do not reach for whichever label
  is closer.
- **Outcome ties, not behaviour, are what failed to separate.** Sixteen of the 46 qualification
  families tied on the mean success fraction over three seeds, against thirteen on D46 at six
  seeds. The next registration should be sized from this: with this tie mass, 46 families and
  three seeds cannot resolve a +0.065 fraction difference, and more seeds reduce ties faster
  than more families cost.

Evidence: `runs/g8/protocol-6seed.out` (development), `runs/g8/qual-summary.out` and
`runs/g8/qual-query.json` (registered), paper claims PROTO6 and QUAL1 (splinter `cd2b63f06`).
