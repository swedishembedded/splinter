<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Protocol, fixed before the first measured run (2026-10-16, 02:05 CEST)

Question: does each stage of the Splinter + Brain + Sven loop do what its specification
says on a real corpus (E1), and does one full turn of the loop, taken twice on a
controlled domain, teach facts the policy was not given, keep what it knew and release,
serve and roll back as specified (E2, E3)? No benchmark claim is made.

## E1: corpus-expansion funnel (descriptive)

- Corpus: three pages of Brain's user documentation (monitoring, security, D-Bus API),
  captured as one directory source. Generator: Qwen3-4B, context 8192. Kinds: recall,
  explain, classify, denoise. The denoise seed is drawn and recorded.
- Reported, no threshold: proposals, admissions and rejections per kind and per reason
  from the task report; text parts covered; wall time. A reason that dominates is
  reported as the finding. If the run does not finish inside a 45 minute wall-clock
  limit, the report states that and uses the kinds that finished.

## E2/E3: two generations on invented facts

- Domain: `samples/generations/generations.py make_facts`, 24 invented facts per
  generation, 3 generations, seed drawn and recorded. Generation 0 and 1 are taught in
  turn by `splinter learn --kinds recall`; generation 2 is never taught.
- Policy Qwen3-0.6B; generator and teacher Qwen3-4B.
- Probes: 4 per fact, built by code, never read by Splinter. A fact is answered when
  at least 3 of 4 probes are answered. Anchor: the sample's 40 general questions,
  frozen as the anchor suite.
- Primary (E2): facts of generation 0 answered after the first release against the
  base, exact McNemar test on facts, alpha 0.05 two-sided. Success needs a release
  and p < 0.05.
- Controls (E2): generation-2 facts answered after the release must not exceed those
  answered by the base by more than the chance level the base shows (we expect 0);
  anchor correct may fall by at most 2 of 40 (the gate's own tolerance); after
  rollback the generation-0 count must equal the count after the first release's
  state before the second release, and the serving manifest must name the first
  release's adapter digest.
- Retention (E3): generation-0 facts answered after the second release may be at most
  1 below the count after the first release; generation-1 facts answered must exceed
  the base's. The gate's own decision on the second candidate is reported whatever it
  is. A refused release is a result.
- A learn that exits non-zero is reported with its exit code and the reason it prints;
  nothing is retried to obtain a better outcome. At most one rerun is allowed, only
  when the binary is replaced by the repaired build, and both runs are then reported.

## Exclusions and post-hoc rule

No item is excluded. Anything computed under a rule changed after the first measured run
is labelled post hoc. All results are pilots: 24 facts per generation.
