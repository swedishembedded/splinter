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

# Second wave, E1 rerun (fixed 2026-10-16, 09:49 CEST, before the runs)

Three defects found by the first wave's failures were repaired in Splinter (a set from one
source text is no longer divided and then refused by its own leakage gate; markup inside a
subject no longer hides it; a candidate's adapter card id carries a tag so `brain serve` accepts
it). E1 is rerun on the repaired binary, once per corpus, nothing retried:

- Corpus A: the same three Brain pages as the first wave (10625 bytes), kinds recall, explain,
  classify, denoise, generator Qwen3-4B, context 8192, denoise seed drawn and recorded.
- Corpus B: three pages of the Sven user documentation (introduction, quickstart,
  troubleshooting), same settings.
- Reported per corpus, no threshold: proposals, admissions and rejections per kind and per
  reason; parts covered; wall time. The rejections of reason `no_subject` are classified by
  reading each against the rule (markup only; words present but apart; subject absent from the
  instruction; subject not in the source) as in the first wave's 82, with the script kept.
- The only directional expectation: no rejection of the markup-only class remains. The
  generator samples, so counts differ from the first wave by chance; no test is run on them.
- E1c (added 2026-10-16 10:17 CEST, after the corpus A rerun showed 75 `no_subject` rejections of
  which 45 name a subject that is absent from the instruction, and before any other run):
  corpus A again with Qwen3-8B as the generator, everything else equal, to test whether the
  rejections follow the generator model rather than the rule or the prompt. The only
  comparison is the share of proposals rejected as `no_subject` by the two generators,
  reported with Wilson intervals and without a test (one run each, the generator samples).

# Second wave, E2/E3 (written 2026-10-16 11:06 CEST, while the run named below was in its first
# `learn` and before any of its results existed)

Changes to the design of the first wave, each forced by a failure or by the pilots:

- A generation's source is a directory with one sheet file per fact (30 facts per generation).
  The first wave gave one file with all facts: Splinter groups records by the source text they
  are grounded in, so the file was a single group, the holdout rule cannot divide a group, and
  training was refused. A held-out measurement needs independent units; here a fact is one.
- Training flags fixed from two pilots (below) on other seeds: `--eval-every 0` (the monitoring
  records are whole unseen families, which for invented facts measure facts the policy cannot
  know; the default picks step 20 of 41 by that loss), `--steps 400 --rank 16
  --records-per-step 2`, `--rehearsal 0.25` (the base's own answers to general tasks, to hold
  the anchor suite). Default learning rate. Seeds are drawn and recorded by the tools.
- The facts are probed in the live policy before and after every step, as in the first wave:
  four probes per fact, answered when at least three are right; a general anchor of 40
  questions. A probe is graded by the in-process verifier of the evaluation command; after
  each release the same probes are also asked over `brain serve` started from the serving
  manifest alone (greedy; the value must appear as a whole number in the reply).

Pilots (not part of any claim): pilot 1, 24 facts in one file per generation's directory,
default training, seed 2393185590908659544: 17 tasks trained, candidate selected at step 20,
7 of 36 paraphrases right, gate refused; pilot 2, 30 facts, seed 5221283959204369301: default
training (no monitoring) 0 of 30 facts answered; 240 steps 8 of 30; rank 16 and 400 steps
13 of 30 with the anchor 39 -> 36 of 40. The registered run is pilot 3 (30 facts, seed
2515273933014230377).

Registered outcomes (E2). Primary: facts of generation 0 answered after the first learn
against the base, exact McNemar test over the 30 facts, alpha 0.05 two-sided; success needs a
release and p < 0.05. A refusal by the gate is reported with its exact reasons and counts as
the outcome "no release", not as a failed run. Controls: generation-2 facts answered after
the learn at most the base's count plus 2; the anchor within the gate's own bound after a
release. If a release exists after the second learn: rollback must move the alias to the first
release and the serving manifest must name the first release's adapter digest, and the
generation-0 answered count after rollback must equal that after the first release.
Retention (E3): generation-0 answered after the second release at most 1 below the count after
the first; generation-1 answered after it must exceed the base's. No item is excluded. All
results are pilots of n = 30 facts per generation; one run, nothing retried; if the run fails
for an infrastructure reason the repaired run is reported beside it.

# Second wave, E2b (post hoc; written 2026-10-16 11:49 CEST, before the run)

The registered run (pilot 3) produced a candidate that passed the improvement, retention and serve
checks and was refused by the anchor check alone (the anchor suite fell by 0.05 against the bound
0.02; candidate probes: 13 of 30 generation-0 facts answered, 0 of 30 of generations 1 and 2).
Whether a lower-forgetting setting reaches a release is a follow-up decided after seeing that
result, so it is post hoc: the same design, a new seed, `--rehearsal 0.5` and `--lr 0.0001`
instead of `--rehearsal 0.25` and the default rate, nothing else changed, one run, nothing
retried. The outcomes and decision rules are those registered above for E2/E3; this run is the
one the release, serving, second-generation and rollback measurements can come from if the gate
releases. The candidate itself is also measured on the probes whether or not it is released.
