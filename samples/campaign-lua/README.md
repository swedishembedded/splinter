<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# campaign-lua

Does the campaign controller put an informative, on-policy signal through a
real model, and does the dev gate decide correctly what that signal did?

The sample runs `splinter_sdk::campaign::CampaignController` end to end on a
base checkpoint, in process, against a verifiable task family: Lua output
prediction. A generated program is shown, the policy writes what it prints, and
the Lua interpreter that ran the program is the verifier. Programs come in six
levels (sums, products, strings, loops, table sorts, long arithmetic); the training pool, the dev set the
gate is decided on and the retention set (levels 1 and 2) are drawn from one
recorded seed under different salts and share no program.

## What one run measures

Per arm, printed as JSON and appended to the campaign's ledger:

- the informative-group fraction of the batch (groups whose attempts earned
  different rewards, among all groups collected), after an offline calibration
  at `--calibration-k` attempts per task and permanent retirement of tasks at a
  pass rate of 0.9 or more;
- the KL to the reference, the clip fraction and the entropy of the step when
  the collector records the sampled tokens (the in-process path records none, so
  entropy is absent, never zero);
- the dev gate: dev pass rate of the candidate and of the champion, tasks won
  and lost, the sign-test p-value against the level this gate may spend
  (alpha spent by gate number), and retention losses;
- the arm's hypothesis, configuration digest, GPU-hours, rollouts and verdict.

An arm whose dev exam could not detect `--target-effect` at the level it may
spend is refused before it rolls out anything.

## Running it

```bash
cargo run --release -p splinter-campaign-lua -- \
    --device gpu2 --backend cuda --model <base checkpoint> --out <directory> --arms 4
```

Running the same command again continues the campaign from its saved state; a
changed setting is refused. The seed is drawn and printed when `--seed` is not
given; pass it back to reproduce the programs and the draws. The campaign needs
`lua` on the path (or `--lua`).

## The `lineage` driver: self-taught rounds

`lineage` runs one lineage of generations against the same family without the
campaign controller. Everything goes through a `brain serve` the driver starts
(serving, the policy's proposals, training jobs):

1. The current policy proposes Lua 5.1 programs from the manual, one section
   per request. Sections are chosen so that the library functions the policy
   answers worst on its dev set are asked about most, and a program that
   repeats an earlier proposal is dropped.
2. The policy predicts what each proposal prints (greedy); the interpreter
   runs it and its output is the label.
3. The mispredicted programs are the new information. A supervised training
   job continues the champion's adapter on them, with `--correct-share` of the
   draws on programs the policy already predicted right and `--replay-share` on
   earlier accepted rounds' records.
4. The candidate and the champion answer the dev and retention sets (the
   base's own held-out proposals and the generator's programs; sized by
   power-first planning before anything runs), and the dev gate decides. Only
   an advance changes the champion; the last advance of a generation is its
   champion, handed to an examiner when `--handoff` is given.

The lineage's journals (`rounds.jsonl`, `dev-results.jsonl`,
`predictions.jsonl`, `proposals.jsonl`) hold everything it knows; running the
command again continues from them.

```bash
cargo run --release -p splinter-campaign-lua --bin lineage -- --help
```
