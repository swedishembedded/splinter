<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 188. Record-driven ratio spikes wrecked an on-policy round; a step guard removed the damage

**Lesson:** on real agent rollouts some records carry tokens whose behaviour log-probability and the trainer's forward
disagree by 10+ nats, so a batch that draws one has an importance ratio in the thousands, a KL in the tens or thousands
and, with a negative advantage, an unbounded loss. A few such steps undo a whole round. Run every policy-gradient round
on rollouts with brain's step guards (`splinter train --max-step-kl`, `--max-step-grad-norm`) until the cause is fixed,
and read a run's KL by its median and its skips, not its mean.

Measured 2026-11-20 (splinter ledger experience-to-competence item 19). A2s1 on brain 1bdd5400c, 640 practice trials on 80
fresh tasks. One `--recipe on-policy-rl` round: 72 steps x 8 records, `--behaviour start` (recorded log-probabilities
where the trace has them, 52% of tokens; the start adapter's elsewhere), `--reference start`.
- **Unguarded (rl1):** 9 of 72 steps spiked.
  - Mean ratio 4.8-484, ratio max up to 1.6e6, KL 31-6,414, one gradient norm 1.4e5.
  - Every other step: KL 0.01-0.05.
  - Against A2s1: held-out pass@1 -0.229 [-0.294, -0.159] on 48 fresh tasks x 8; training tasks -0.191 [-0.263, -0.116];
    retention -0.165 [-0.258, -0.071].
- **Guarded rerun (rl1g):** same data, same seed, `--max-step-kl 0.1 --max-step-grad-norm 5`.
  - It spiked on the same step indices and skipped 13 steps.
  - Held-out -0.031 [-0.081, +0.018], training tasks -0.056 [-0.131, +0.016], retention -0.038 [-0.121, +0.038].
  - Single seed.

Why the spikes come from the records, not from drift: the guarded run reproduced the unguarded one exactly up to its first
skip. At step 6 the policy had moved by KL < 0.008, yet a token's ratio was 1.9e4, and both ratio extremes appear
(minimum about 1e-5). The open question is which records. It is either recorded log-probabilities that do not belong to
the trained ids, or proxy tokens (`scrubbed`, `context_reconstructed`) that the policy never sampled. Each needs a
per-record ratio dump.
