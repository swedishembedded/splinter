<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Launch record of RSI4 (written 2026-11-14, before generation 1 of any lineage)

Resolves the settings the registration (`rsi4.md`) leaves to this record. Anything changed after this commit is reported as a
dated deviation beside the outcome. The exam is held by the examiner and was never read; every number below comes from the
learner's own dev tasks, except the examiner's base-against-base null replicate, which compares the base with itself.

## Builds

- splinter `86f13672ab8dddfaf1c7785978e567ecbcfb7a01` (binary `lineage` of `samples/campaign-lua`), binary `lineage-86f13672a`,
  SHA-256 `ea233cc6db7c16b54ddd321688f6835bc17c770bfb6a08fd42e0254b63a3ef5d`.
- sven `fc99a2f1aa414701a323d164a7c8c6b518f887fa` (the provider the requests go through, as pinned by that splinter commit).
- brain `796871647e854a68089631c98842c027cd5c9c7a`, `brain` binary `brain-796871647`, SHA-256
  `a020a8eba00cb699fb1c901ecf1a8f63a2dc82d4ce14658973258e3fbeb3ad26` (the learner's serving and training).
- Lua 5.1.5 interpreter, SHA-256 `4341300318ecc67f4e077b0db1c6b6160a39d39fb5b01eb29c194972130d25c7`.
- Base: Qwen/Qwen3-4B; its digest is written in every candidate (`base_digest`).
- Examiner (bench's, never read by the learner): bench `d8a75b30`, serving every model with brain binary `brain-d11e91340`
  (SHA-256 `d11e91340b55687671d45a6ef97a1107eb7a39d2ee56b4a3bf05de9b4cfc85b6`) at `--kv-cache f32`, the exam's registered
  direct-completion protocol with a reply the token cap ended graded as written.

## Path

Brain is reached only through its API, as in RSI3: one `brain serve --openai --max-batch 64 --kv-cache f32 --serve-adapters DIR
--training --artifact-store DIR` per lineage on its card, through which the policy proposes, predicts, is examined on its dev and
retention sets and is trained (brain training jobs, `objective sft`, LoRA). Nothing runs in the learner's process.

## Generation

One round per generation. The champion proposes programs from the Lua 5.1 manual (one manual section per request, as in RSI3)
until the frontier holds the generation's target, within its request cap. The frontier: the champion predicts each proposal 4
times at temperature 1.0; a program enters training when at least one of the 4 predictions is wrong; the interpreter's output is
the target (one fenced block). Generation g's candidate is a fresh LoRA trained **from the base** on the union of the frontier
programs of every accepted generation 1..g (cumulative retraining), 2 epochs over that union. No correct-share or replay draws
(the union is the replay).

**Frontier schedule (resolved here):** the registration names "the size fixed in the launch record"; it is a per-generation
size, doubling: **300, 600, 1200** frontier programs at generations 1, 2, 3, with request caps 3000, 6000, 12000. Reason
(dev pilot below): with a fixed 300 per generation the second step fell to +0.045 on dev; accuracy grows roughly with the log of
the data, so equal steps need geometrically growing data.

## Settings (the same for every lineage)

| Setting | Value |
|---|---|
| Generations, rounds per generation | 3, 1 |
| Dev set | 100 generator programs + 300 base proposals (held out, never trained on) = 400 units, as RSI3 |
| Retention set | 150 generator programs + 150 base proposals = 300 units, as RSI3 |
| Gate | as RSI3: net dev gain, retention loss within 0.05 of 300, alpha spent by gate number (total 0.05); target 0.08, discordance 0.2 |
| Training | SFT LoRA rank 16, lr 3e-5, 8 records per step, 2 epochs over the cumulative frontier union (steps follow: 75, 225, 525 at the schedule's sizes) |
| Frontier | 4 samples at temperature 1.0 per proposal; targets 300 / 600 / 1200; request caps 3000 / 6000 / 12000; 64 in flight |
| Held-out proposals | 450 from at most 1400 requests |
| Server | KV cache f32, 64 slots |
| Seeds | r0 3217799999916437381, r1 4785861403145693048, r2 4743887585493406067, r3 6379732546301056828; generator seed 8599218772143256343 (the same generator programs for all four) |
| Cards | r0 gpu0, r1 gpu1, r2 gpu2, r3 gpu0 (shares gpu0 with r0); each generation holds its card's flock; the examiner serves on gpu1 between generations |

## Pilot (dev only; lineage cells with their own seeds and dev sets; no exam item involved)

| Cell | Settings | Generation 1 dev (wins / losses) | Generation 2 | Generation 3 |
|---|---|---|---|---|
| p1 | fixed 300, K 4, 2 epochs | 0.3875 -> 0.595 (110 / 27) | - | - |
| p2 | fixed 300, K 8, 2 epochs | 0.425 -> 0.5725 (86 / 27) | - | - |
| p3 | fixed 300, K 4, 1 epoch | 0.39 -> 0.5625 (91 / 22) | -> 0.6075 (23 / 5) | stopped |
| g4 | 300 / 600 / 1200, K 4, 2 epochs | 0.385 -> 0.615 (122 / 30) | -> 0.685 (33 / 5) | -> 0.735 (28 / 8, p 0.0006) |

Chosen: g4's recipe. Proposal yield at generation 3 (1200 frontier programs) stayed within its cap. Examiner null replicate
(bench, base against base, the exam's 889 items, f32 KV, brain d11e91340): valid, 12 / 889 right on both sides, 0 discordant,
0 failures. (The stored RSI3 baseline of the same build had the base at 0: it was served at brain's default KV tier; every RSI4
model is served at f32, so the comparison stays matched.)

## Hand-off

As RSI3, under `~/project/shared/rsi4/handoff/candidates/<series>/<step>/` (`training-tasks.jsonl` first, listing every program
the step's candidate trained on, `candidate.json` last); adapters kept at `~/project/shared/rsi4/adapters/<series>/<step>/`.
