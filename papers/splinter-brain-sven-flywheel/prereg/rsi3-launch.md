<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Launch record of RSI3 (written 2026-11-13, before generation 1 of any lineage)

Resolves the settings the registration (`rsi3.md`) leaves to this record. Anything changed after this commit is reported as a
dated deviation beside the outcome. The exam is held by the examiner and was never read; every number below comes from the
learner's own dev tasks.

## Builds

- splinter `5ce8666c5ac1b5466639bf475823c4344fa2c28d` (binary `lineage` of `samples/campaign-lua`), binary `lineage-5ce8666c5`,
  SHA-256 `bbb49a7d4e2d9564e13532369b8bf8533d8daba54c16b0232be110b869276e03`.
- sven `5bcb5f78a28a71f0c56048c12104ec6dee427818` (the provider the requests go through).
- brain `796871647e854a68089631c98842c027cd5c9c7a`, `brain` binary `brain-796871647`, SHA-256
  `a020a8eba00cb699fb1c901ecf1a8f63a2dc82d4ce14658973258e3fbeb3ad26`.
- Lua 5.1.5 interpreter, SHA-256 `4341300318ecc67f4e077b0db1c6b6160a39d39fb5b01eb29c194972130d25c7`.
- Base: Qwen/Qwen3-4B; its digest is written in every candidate (`base_digest`).

## Path

Brain is reached only through its API: one `brain serve --openai --max-batch 64 --kv-cache f32 --serve-adapters DIR --training
--artifact-store DIR` per lineage on its card. Through it: the policy's proposals (typed calls, one manual section per request),
its predictions (greedy, 256 tokens, no system prompt, thinking off), the dev and retention exams and the training of each round as
a brain training job (`objective sft`, LoRA continued from the champion). Nothing runs in the learner's process.

## Generation and round

A generation is one round (the registration's "last accepted policy" is the round's candidate when the gate advances, else the
predecessor). A round: the champion proposes up to 600 programs (800 requests at most, 64 at once), predicts each, the interpreter
labels it; the mispredicted programs are the training set; the programs predicted right are drawn at a share of 0.25 of the
draws; earlier accepted rounds' records (both kinds) at a share of 0.25 (none in generation 1). Target of every record: the
interpreter's output in one fenced block. A proposal that repeats an earlier one (exact program, or instruction text at Jaccard
0.8 over 3-word shingles) or the generator's is dropped.

Proposal diversity (beyond RSI2's 240): one manual section per request; sections chosen with weight 1 + 3w for the dev-weakness
w of the worst library function they mention (functions with at least 3 dev programs), every section before any twice; weak
functions named in the request; each visit to a section takes one of 12 angles and a seeded word and number to include.

## Settings (the same for every lineage and generation)

| Setting | Value |
|---|---|
| Generations, rounds per generation | 3, 1 (six gates would leave the 400-program dev set unable to detect 0.08; three can: minimum detectable 0.0788 at level 0.0042) |
| Dev set | 100 generator programs (levels 1-6) + 300 programs proposed by the lineage's base (held out, never trained on) = 400 units |
| Retention set | 150 generator programs (levels 1-2) + 150 base proposals = 300 units (204 needed for a loss of 0.05) |
| Gate | net dev gain and retention loss within 0.05 of 300; alpha spent by gate number (total 0.05), significance reported; target effect 0.08 at discordance 0.2 |
| Training | SFT LoRA rank 16 alpha 32, lr 3e-5, 40 steps, 8 records per step, bf16 base, max block 1024, weight decay 0 |
| Mix | mispredicted programs the rest of the draws; correct share 0.25; replay share 0.25 |
| Proposals | 600 per round, 800 requests, 64 in flight; held-out dev and retention proposals: 450 from at most 700 requests |
| Server | KV cache f32, 64 slots |
| Seeds | r0 7822578567680965267, r1 2377727685644984238, r2 1380198831415946718, r3 1925759896007873967; generator seed 2644622322983530489 (the same generator programs for all four) |
| Cards | r0 gpu0, r1 gpu1, r2 gpu2, r3 gpu3, each under its card's flock for one generation at a time |

## Pilot (dev only, one round from the base, 400-program dev set, base 0.4025; no exam item involved)

Same 600 proposals and base answers in every cell (training is deterministic: identical settings gave identical candidates).

| Cell | Change from lr 3e-5, 40 steps, rank 16, batch 8, correct 0.25 | Dev | Wins / losses | Retention losses |
|---|---|---|---|---|
| c01 | lr 1e-5 | 0.5400 | 80 / 25 | 2 |
| c02 | (reference) | 0.5775 | 99 / 29 | 4 |
| c03 | lr 1e-4 | 0.5525 | 89 / 29 | 2 |
| c04 | 80 steps | 0.5750 | 94 / 25 | 3 |
| c05 | lr 1e-5, 80 steps | 0.5800 | 97 / 26 | 5 |
| c06 | correct share 0 | 0.5875 | 102 / 28 | 4 |
| c07 | correct share 0.5 | 0.5775 | 98 / 28 | 2 |
| c08 | rank 32 | 0.5525 | 89 / 29 | 4 |
| c09 | batch 16 | 0.5800 | 98 / 27 | 3 |
| c10 | 300 proposals (248 mispredicted) | 0.5700 | 95 / 28 | 2 |

Second round from the c02 champion (replay share 0 / 0.25 / 0.5): dev 0.5925 (14 / 8, retention losses 2) / 0.5950 (11 / 4, 1)
/ 0.5875 (18 / 14, 7). Cell c11 (150 proposals) ran on a different dev set (its seed's own held-out proposals) and is not
comparable. Chosen: the reference, replay 0.25 (fewest retention losses); the differences among the upper cells are not resolved
(one seed per cell). Round 2 added 0.01-0.02 and proposing for it repeated 40-45 percent of its programs, so one round per
generation. Proposal yield of the base, per 320 requests: RSI2's windows saturate at about 240 programs; one section per request
kept 235, with angles 186, with angles and seeded literals 357 (704 requests: 604 new programs after 450 held-out).

## Hand-off

`~/project/shared/rsi3/handoff/candidates/<series>/<step>/` (series = the lineage, step = the generation): `training-tasks.jsonl`
first (every program the candidate's chain trained on, plus dev and retention programs by role), `candidate.json` last. Adapters
are kept at `~/project/shared/rsi3/adapters/<series>/<step>/adapter.safetensors` and never deleted before the verdict exists.
