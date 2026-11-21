<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 200. Pre-registered leak and observation filters reject nearly all unhinted vim passes

**Lesson:** before a rationalisation filter is trusted to detect what a hint taught, run it on attempts that never
saw a hint. On the vim benchmark two of item 24's pre-registered filters reject almost all of the base model's own
unaided passes, so as written they would stop the arm for reasons unrelated to the hint.
`experiences rationalise` now reports both controls in every run (`own_term_mentions`, `own_real_observations`).

## Evidence (2026-11-21, CPU only, no model run)

The base Qwen3-4B's 221 passed trials of `train-k1` and `train-k4` (840 trials, knowledge 198), read by
`experiences rationalise` (splinter worktree on 2018a67d8) with brain's mock model as the rationaliser:

- **Leak terms.** In its own reasoning and messages, 218 of 221 passes mention `expected` and 99 mention `solution`.
  Every vim instruction has an `Expected - main.txt` heading, and the model says "the expected output" and "the
  solution is" unprompted. Counts for the other terms: `reference` 4, `as given` 1, `hint` 1, `solve.sh` 0. Over all
  840 trials (failed ones included) the counts are 832 for `expected` and 462 for `solution`.
- **Real observations.** 19 of 221 passes pass the read-before-write filter. 155 write the file in their first tool
  call without printing it, because the instruction already shows the starting file. 10 of the 19 pass because
  their task changes no file with content to read first; 9 show the file in their first tool reply.

The two filters work as intended on a real leak. The mock echoes its whole prompt, hint section included, and both
mock attempts were rejected for mentioning `solution`, `expected`, `reference` and `hint` and for repeating the hint
preamble's wording.

## What it means

Under the default term list, at most 3 of 221 unhinted passes clear filter 3, so the 10% sufficiency stop rule would
fire before the hint is tested. Dropping the terms the instruction or the model's own idiom uses (`expected`,
`solution`) is a deviation (`--leak-term` replaces the list), and so is relaxing filter 2. Either is the lane owner's
decision, recorded as a dated deviation of item 24.

The hinted instruction tells the model to read the files before it changes them. Whether that lifts filter 2 above
the base rate is measured by the feasibility gate.
