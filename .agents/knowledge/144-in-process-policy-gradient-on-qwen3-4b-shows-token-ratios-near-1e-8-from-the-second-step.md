<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 144. In-process policy gradient on Qwen3-4B shows token ratios near 1e-8 from the second step, whatever the learning rate

**Lesson:** Brain's clipped policy-gradient run on Qwen3-4B (LoRA rank 16, alpha 32, `--behaviour start`,
`--reference start`, no KL term, 96 to 128 records of short answers) has importance ratio exactly 1 at step 1, as it
should, and from step 2 on reports a smallest token ratio of 0.09 to 0.3 on some steps and about 2.5e-8 on others, with
mean KL to the reference of 1e5 to 1e7 on those steps. The same ratio (2.48e-8, 2.44e-8) recurs across steps, which a
policy that moved would not do: it points at a token whose behaviour log-probability (the start-of-run proxy) and the
training forward disagree, not at a diverging update. The campaign controller's KL bound caught it and adopted
nothing.

## Measured (one GH200, bf16 base)

- lr 1e-5, 16 steps, 88 records: mean KL 8.8e6; step 2 ratio_min 5.9e-6, KL 3.1e3; steps 5 to 9 KL 1.4e6 to 2.1e7.
- lr 3e-6, 12 steps, 128 records: mean KL 1.4e5.
- lr 3e-6, 12 steps, 96 records, `--base-precision fp32` asked for (brain trained on bf16 tensor cores regardless): step 2
  KL 0.18 with ratio_min 0.09, steps 4, 5, 7, 8, 9 KL 7e5 to 1.2e6 with ratio_min 2.4e-8 to 2.5e-8, mean KL 3.7e5.
- A step after one Adam step at lr 3e-6 cannot move a token by 17.5 nats; the ordinary steps' ratio_min of 0.09 at
  step 2 is already larger than a sign-sized LoRA update explains.

## What this does and does not establish

Not isolated: which token (answers are 2 to 8 tokens, so the end-of-turn token and the first answer token are
suspects), whether the proxy or the training forward is wrong, and whether `--behaviour recorded` with sampler
log-probabilities removes it. Brain owns the cause (the numerics gate of the design review: a step-0 ratio of 1 is not
enough, a step-1 ratio after a tiny update must also be near 1). Until it is fixed, a policy-gradient arm on this
setting is a measurement of the guard, not of learning: the controller records the halt with the signals.
