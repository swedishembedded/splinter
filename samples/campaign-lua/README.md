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
   power-first planning before anything runs), and the dev gate decides
   (`--gate`, below). Only
   an advance changes the champion; the last advance of a generation is its
   champion, handed to an examiner when `--handoff` is given.

Two options change steps 2 and 3, both off by default:

- `--train-from base-cumulative` trains each candidate as a fresh adapter on the
  base, on the union of every accepted generation's training records and the
  new ones, instead of continuing the champion's adapter (`--replay-share` must
  be 0: the union already holds the earlier records). `--cumulative-epochs E`
  lets the steps grow with the union: `max(--steps, ceil(E x records /
  --records-per-step))`.
- `--frontier-samples K --frontier-temperature T --frontier-target N
  --frontier-request-cap C` select the frontier: the champion predicts each
  proposal K times, sampled at temperature T (instead of once, greedily), and
  only programs it did not predict right every time are trained on (the
  interpreter's output is the target; `--correct-share` must be 0). Proposing
  goes on in waves of `--proposals-per-round` programs until N frontier
  programs are found, within C proposal requests per round; the round fails
  with the numbers found when C is reached, and running it again with a larger
  C continues from the journalled waves and predictions. The round trains on
  the first N frontier programs, in the order proposed, whose training records
  fit `--max-block` tokens (measured as the trainer measures them); a longer
  one is skipped and listed with its length under `frontier.overlong` in the
  round's record, and the round fails when fewer than N fit.

`--gate evidence` replaces the dev gate of step 4 with the evidence gate,
which holds the false evolution rate (the share of accepted rounds that are
worse by `--gate-eps` or more) to `--gate-q` over every planned round
(`--generations` x `--rounds`). The dev set becomes a pool: each round draws
`--dev-draw` of its programs without replacement (fewer than it holds),
grades the champion and the candidate on them `--dev-batch` at a time (default
`--slots`), and advances only when the noninferiority e-process, with the
overlap correction `--gate-beta` for programs earlier rounds used, crosses the
round's level; otherwise the champion stays and the next round tries again.
Only drawn programs are graded, and the proposer is steered by the champion's
results on those alone. The rounds journal records the e-value, the level, the
budget left, the exposure and the programs spent, and a lineage refuses to
continue under another gate or other gate settings. The four settings have no
defaults (they are fixed before a lineage runs); the default `--gate greedy`
is the gate every registered lineage ran with. How to choose the settings is
in https://swedishembedded.com/docs/splinter/guides/acceptance-gates (the
manual's page on acceptance gates).

`--proposer-state DIR` moves the proposing handle's task records (one file per
record, disposable) off the project disk, for instance to a memory-backed
directory.

The lineage's journals (`rounds.jsonl`, `dev-results.jsonl`,
`predictions.jsonl`, `proposals.jsonl`, and with a frontier `waves.jsonl` and
`frontier.jsonl`) hold everything it knows; running the command again
continues from them.

```bash
cargo run --release -p splinter-campaign-lua --bin lineage -- --help
```
