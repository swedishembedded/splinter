<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 204. Stable on-policy RL on own rollouts fits the trained tasks and gives up retention

**Lesson:** once the importance ratio and the KL penalty are bounded, a guarded on-policy round on a policy's own
rollouts trains cleanly. It reaches the KL band and takes two passes without spikes. It still fits only the tasks it
practised: their pass rate rises, fresh tasks of the same domains stay flat, and the retention suite drops. A clean
optimiser is necessary, not sufficient. The next lever is the signal, not the stability:
- per-turn credit;
- rehearsal of what must be kept, or a KL to the champion;
- the actions that `--sampled-only` leaves out.

Measured 2026-11-21 (ledger experience-to-competence item 19, round R19, lane R19, n110).

**Setup.** A2s1 practised 80 fresh train-side tasks at G = 8 (640 trials). The policy-gradient view used
`--reward shaped-partial` and `--sampled-only`. `splinter train --recipe on-policy-rl --behaviour recorded --reference
start --dual-clip 3 --max-step-kl 0.1 --max-step-grad-norm 5`, 501 steps x 8 records (2 passes), on brain 7a7869433
(the ratio-weighted KL penalty, brain knowledge 537), seeds 1 and 2.

**Coverage.** `--sampled-only` kept 2,006 records and left out 1,068 as `proxy_tokens`. That is 65% of the records with a
group signal; 1,631 of 2,699 actions were exact (`scrubbed` 522, `context_reconstructed` 546). Splinter has no
turn-level credit: every action of a trial carries the trial's advantage.

**Training.**
- Median KL 0.024 (inside 0.02-0.05). Median clip 0.081-0.082, under the recipe's 0.10-0.20 at lr 1e-4.
- The dual clip bound 0.03% of signed tokens (ratio max 34-54).
- Guard skips 0 and 2 (both on KL 0.11-0.13), gradient norm at most 4.
- About 20 s per step: recorded behaviour skips the start-of-run scoring pass that made round 1 cost 184 s per step.
- On brain 5e25d5b82 (unweighted KL penalty) the same recipe diverged on seed 1 and skipped 16 steps on seed 2.

**Against A2s1** (paired by task, 95% task bootstrap, every arm at the same serving load):

| Measure | Two-seed RL mean | RFT control |
|---|---|---|
| Training tasks, 40 x 8 | +0.064 [-0.013, +0.138] | +0.022 |
| Held-out, 48 fresh x 8 | -0.016 [-0.064, +0.033] | -0.021 |
| Retention, 91 x 2 | -0.077 [-0.151, -0.003] (fails non-inferiority at 0.10) | +0.005 (non-inferior) |

- The RFT control is the sft-step view of the same round's verified passes, matched in steps.
- Sampled-token NLL at T = 1 rose from 0.39 to 0.47-0.59: the policy got broader.
