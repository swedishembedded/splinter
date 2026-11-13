<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Launch record of RSI2 (written 2026-11-13, before generation 1 of any lineage)

Resolves the settings the registration (`rsi2.md`) leaves to this record. Anything changed after this commit is reported as a
dated deviation beside the outcome. The exam is held by the examiner and was never read; every number below comes from the
learner's own dev tasks (finding 147).

## Builds

- splinter `ee63d908b0c614de98145633a56f61dd976e14ce` (sample `rsi2` of `samples/campaign-lua`), binary
  `rsi2-ee63d908b`, SHA-256 `4ce140353f924167d9550d504e081acfbc4858f5386fc0c418467fed3449d4c8`.
- sven `fc99a2f1aa414701a323d164a7c8c6b518f887fa` (the provider the requests go through).
- brain `796871647e854a68089631c98842c027cd5c9c7a`, `brain` binary `brain-796871647`, SHA-256
  `a020a8eba00cb699fb1c901ecf1a8f63a2dc82d4ce14658973258e3fbeb3ad26`.
- Lua 5.1.5 interpreter, SHA-256 `4341300318ecc67f4e077b0db1c6b6160a39d39fb5b01eb29c194972130d25c7`.
- Base: Qwen/Qwen3-4B; its digest is written in every candidate (`base_digest`: the SHA-256 of the lines `name sha256` of its
  safetensors files).

## Path

Brain is reached only through its API. Each lineage runs one `brain serve --openai --max-batch 32 --kv-cache f32
--serve-adapters DIR --training --artifact-store DIR` on its card. Through it: the policy's task proposals (typed calls of
splinter's sven-backed agent layer, thinking off), the rollouts (one request per attempt, temperature 1, no nucleus or top-k
cut, 256 tokens at most, no system prompt), the dev and retention exams (greedy) and the training of each arm as a brain
training job (`brain/train`, objective `policy_gradient`). A trained adapter is placed in the followed directory and waited for
until the server lists it under its digest. Nothing runs in the learner's process. Not used: `sven rollout` task folders (the
tasks are answer-only programs; the controller's served chat path is the one that exists at this build).

## Settings (the same for every lineage and generation)

| Setting | Value |
|---|---|
| Generations per lineage | 3, each one controller arm from the previous champion; a rolled-back arm keeps the predecessor |
| Training tasks | 160 programs of the seeded generator (source b) + proposals of the current policy over the three manual pages (source a): up to 120 new per generation |
| Dev set | 300 generator programs + 100 programs proposed by the base (held out, never trained on) = 400 units, greedy |
| Retention set | 300 generator programs of levels 1 and 2 |
| Calibration | k = 8 attempts per task by the current policy; retire at pass rate >= 0.9; dynamic sampling keeps groups with a reward spread |
| Batch | 24 informative groups of 8 attempts; at most 4x over-provisioned; replay 0.25 of the batch from earlier arms |
| Update | policy gradient, group-standardised advantages (two-sided), rank 16 alpha 32 LoRA continued from the champion, bf16 base |
| Learning rate, steps | 6e-6, 24 steps, 8 records per step (no warmup beyond brain's schedule) |
| Clip | 0.2 lower edge, 0.28 upper edge (clip-higher); KL weight 0; reference and behaviour = start of the arm |
| Guards | step skipped when mean KL > 0.5 or gradient norm > 10; the controller halts an arm whose mean KL exceeds 0.5 and reports a step starved below 0.25 informative |
| Gate | dev sign test at alpha spent by gate number (total 0.05); target effect 0.08 at discordance 0.2 (exam detects 0.064 at the first gate); retention non-inferiority margin 0.05 of 300 (loss of 15 passes) |
| Protocol | neutral request (the fenced program and the question), thinking off, answer = last fenced block, whitespace-normalised token comparison |
| Completion | 256 tokens, 32 requests in flight |
| Server | KV cache f32 (brain finding 446: 0.0099 mean absolute log-probability gap against 0.0245 for int8), 32 slots |
| Seeds | controller seeds r0 4282888424084496033, r1 2433016285090944943, r2 143096754986046224, r3 1187858343585591204; generator seed 2644622322983530489 (the same programs for all four) |
| Cards | r0 gpu0, r1 gpu1, r2 gpu2, r3 gpu3, each under its card's flock for the length of one generation |

## Behaviour log-probabilities

The records carry no sampler log-probabilities at this build (the wire format has no field for them), so the objective scores
the behaviour policy as the start of the arm. Each arm samples fresh from the policy it starts from, so this equals the
behaviour policy up to the serve/train numeric gap. That gap was not remeasured here: brain's numerics gate (finding 446,
Qwen3-4B, f32 KV, 20 prompts x 4 samples) reports a mean absolute gap of 0.0099 per token, 0.27% of tokens outside the 0.2
clip band and a sequence effective sample size of 0.96. In the pilots the first step's ratio was 1 and the mean KL stayed at
or below 0.018 with no step skipped by the guard. This departs from the registration's wording ("behaviour log-probabilities as
recorded at sampling") and is recorded as a deviation.

## Pilot (dev only, one arm from the base)

Eight cells of learning rate and steps (finding 147); 6e-6 for 24 steps chosen: dev 0.4875 to 0.5625 (39 wins, 9 losses), KL
0.0035, clip fraction 0.006, no retention loss. Lower rates did not resolve a gain at 12 steps; 48 steps and 1e-5 raised the KL
(0.013, 0.018) without a larger gain. One seed per cell.

## Hand-off

`~/project/shared/rsi2/handoff/candidates/<series>/<step>/`: `training-tasks.jsonl` first, `candidate.json` last (series = the
lineage, step = the generation; the base is step 0 and has no file). The adapter of each step is kept at
`~/project/shared/rsi2/adapters/<series>/<step>/adapter.safetensors`, never deleted until the examiner's verdict exists.
