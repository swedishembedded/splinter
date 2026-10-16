<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# First flywheel campaign, October 2026 (archived, not a paper result)

Records of the first pre-registered campaign of the self-learning loop, removed from
`papers/splinter-brain-sven-flywheel` so that the paper holds no unreported or failed
registration. The campaign failed its registered criterion: with a 0.6B policy every
candidate learned facts and passed the improvement and retention checks and was refused by
the anchor check alone; a post hoc 1.7B run released its first candidate, and its second
candidate forgot most first-generation facts without the retention check noticing.
Its lessons are in the paper as requirements of the loop (`sections/06-requirements.tex`).

- `prereg/protocol.md`: the registration (E1 corpus-expansion funnel, E2/E3 two generations on
  invented facts) with its dated second-wave amendments.
- `data/runs/e1`, `data/runs/e2`: first-wave raw records; `data/runs/w2`: second-wave runs
  (E1 reruns a-c, E2 registered run, E2b, E2c repaired verifier, E2d 1.7B policy).
- `scripts/`: the drivers (`run_experiments.sh`, `flywheel_cycle.py`, `second_generation.py`),
  the archiver, the rejection classifier and the statistics helpers they used.

The scripts were written for the paper directory and are kept as run; paths inside them
refer to it. Nothing here is a claim of the paper.
