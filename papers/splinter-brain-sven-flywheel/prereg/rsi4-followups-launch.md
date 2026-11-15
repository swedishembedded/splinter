<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Launch record of the RSI4 follow-ups (written 2026-11-15, before any F, M, M-900 or S candidate was examined)

Resolves the settings the registration (`rsi4-followups.md`, with its dated deviation) leaves to this record. Each arm's part is
committed before that arm's first candidate is handed off. Anything changed after its part is reported as a dated deviation beside
the outcome. Builds, protocol and settings not named here are RSI4's (`rsi4-launch.md`).

## M - matched-data control

- Seeds (from the registration): m0 8785991200924142667, m1 7404742185133534162, m2 811289239880634778, m3 2657151457107420430;
  generator seed 6881244634785368405.
- Script: `~/project/shared/rsi4f/run-control.sh SERIES CARD SEED` (SHA-256
  `7111cf407f4f2c3e0befcf3023982dd380c35bf1aa30ffbb9a42d5b17982e4b5`): RSI4's `lineage-86f13672a` with RSI4's settings, one
  generation from the base, frontier target 2100, request cap 21000.

## M-900 - matched-data control at generation 2's amount

- Seeds: n0 5921382301972984548, n1 3433440566262762178, n2 87529672973948155, n3 158072112317300048; generator seed as M.
- Script: `~/project/shared/rsi4f/run-control-900.sh SERIES CARD SEED` (SHA-256
  `c563d95de66200ed5e797cd9ccc15f8bfb3f0ef5f142c1023e09fab3cc0897f0`): as M with frontier target 900, request cap 9000.

## S - seed null

Each RSI4 lineage r*i*'s generation-3 training job is repeated from the base with a new training seed; every other part of the
request is the one the lineage driver sent. The result is handed off as series s*i*, step 1, with `"seed_null_of": "r<i>/3"`.

**The training set, proven identical.** The job record brain wrote for each RSI4 generation-3 champion
(`~/project/shared/rsi4/lineages/r<i>/adapters/g3r1/training.json`) names its dataset by digest; that digest is the SHA-256 of the
lineage's `datasets/g3r1-mispredicted.jsonl`, and of the artifact brain's store kept of the job's dataset (the store's own key is
not the content digest; the file content is). Each file holds 2100 records with 2100 distinct task ids, and those ids are exactly
the `train`-role ids of RSI4's hand-off list `handoff/candidates/r<i>/3/training-tasks.jsonl` (2800 entries: 2100 train, 400 dev,
300 retention). The held-out file is the job's `g3r1-held-out.jsonl` (64 dev proposals, never trained on), equally the store's.

| Lineage | Dataset SHA-256 (= job record = artifact store) | Store key of the dataset | Held-out SHA-256 | RSI4 training seed |
|---|---|---|---|---|
| r0 | `50e7e028f3885aac4ec4dd2296d00655d672d6dd25688c03996d854b9a4ec801` | `88f7ac91...` | `35a50080...` | 3217800004211404676 |
| r1 | `0379772c2ead60bcb4c4b82a3a469f42fbd93178026a6ca648ae14d8f7078f7f` | `f3c84cd9...` | `5d7944f8...` | 4785861398850725753 |
| r2 | `8049484a13df1f1aff72bbb88e615feb46300a93cf5d8a4652a9fa1be862f7c8` | `024f8cbb...` | `27560cdb...` | 4743887572608504178 |
| r3 | `17ff436924c5cde86fa2dfcf72e7a9325c45390ce492c746f5f870e65674fdcd` | `9d2401c3...` | `b5fd7f7b...` | 6379732533416154941 |

RSI4's training seed is the lineage seed XOR (generation << 32 | round), here XOR (3 << 32 | 1), as the driver
(`samples/campaign-lua/src/lineage.rs`) computes it; each value above is the one the job record states.

**The request, reproduced field by field.** From the driver's request (`lineage.rs`, `train_and_gate`) at RSI4's command line
(`~/project/shared/rsi4/run-lineage.sh`) and confirmed by each job record: SFT LoRA from the base (no adapter continued), rank 16,
alpha 32 (twice the rank), learning rate 3e-5, 8 records per step (`grad_accum` 8), 525 steps (2 epochs over 2100 records), weight
decay 0, longest row 1024 tokens (the recipe's default), frozen base bf16, no replay and no weighted draws (`--replay-share 0`,
`--correct-share 0`), no monitor, `eval_every` 0, patience 0, thinking off, answers not packed, the last step selected. The training
job runs on the same `brain serve` as RSI4's (`brain-796871647`, `--kv-cache f32`, 64 slots, `--training`). `samples/served-items`
fixed the longest row at 4096 and the weight decay at 0; commit `c4a237a4f` lets its `train` step state both, so the request is
the driver's exactly rather than an approximation. The run accepts a result only when brain's record of the new job equals RSI4's in
dataset digest, record count, steps, rank, alpha, learning rate, weight decay, base tier, computed block, base digest and selection,
differs in the seed alone, scores the base on the same held-out records, and the adapter answers a probe through serving with its
own digest and differs from the champion's.

**Seeds** (drawn 2026-11-15 from the operating system's random source):

| Series | Seed null of | Training seed | Card |
|---|---|---|---|
| s0 | r0/3 | 11454757556434858119 | gpu0 |
| s1 | r1/3 | 12758312050340776601 | gpu1 |
| s2 | r2/3 | 7641335182389724810 | gpu2 |
| s3 | r3/3 | 17713819005411514143 | gpu0 |

Each run waits on its card's flock behind the M and M-900 runs queued there; gpu3 (the examiner's) is not used.

**Builds.** `served-items` built from splinter `c4a237a4fec59801170c1b62cafd39b4244bda40` against sven
`3d4508abf1a38630b22e73ebbb65bed953e8097f` (the provider) and the brain crates of `b6e06e82b6f9367a1b2f744e6ab68f9ca15f2d93`
(Splinter's reviewed brain exceptions; the job is computed by the server), binary `served-items-c4a237a4f`, SHA-256
`d7bae1999bf43f9469e517e3adbcff1f2f578b5c0b6d69ff59c3c9c5f76275a6`. The training request path (`splinter-model`'s
`served_train`) is unchanged since RSI4's splinter `86f13672a`. Server and trainer: `brain-796871647`, SHA-256
`a020a8eba00cb699fb1c901ecf1a8f63a2dc82d4ce14658973258e3fbeb3ad26`.

**Script.** `~/project/shared/rsi4f/run-seed-null.sh SERIES CARD SEED LINEAGE` (SHA-256
`082211866a374950d9106ca1232e4982c8d9c6d8d67def46d09c30c9bdae97a5`); records under `~/project/shared/rsi4f/seed-null/<series>/`.

**Hand-off.** `~/project/shared/rsi4f/handoff/candidates/s<i>/1/`: `training-tasks.jsonl` first (RSI4's `r<i>/3` list, copied
byte for byte), `candidate.json` last, in RSI4's schema (series, step, base, base digest, adapter under
`~/project/shared/rsi4f/adapters/s<i>/1/`, adapter digest, serving flags, commits, binary digest) with `seed_null_of` and
`training_seed`; RSI4's `advanced` and `arm_verdict` are absent, since no gate decides on a seed-null candidate.
