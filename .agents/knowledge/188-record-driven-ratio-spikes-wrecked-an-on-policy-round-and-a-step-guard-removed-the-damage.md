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

## Update 2026-11-20: root cause, start-of-run proxy tokens; fix, `--sampled-only`

**Which tokens.** Every spike was on a proxy token, never on a recorded one.
- The round-1 dataset's 1,027,275 recorded behaviour log-probabilities are all at or above -5.18. The sampler draws
  under top-k 20 and top-p 0.95, so it never records a far-tail token.
- With the trainer's log-probability at or below 0, a recorded token's ratio is at most e^5.18 = 177. Every spike (518 to
  1.6e6) is above that, so each one was a token weighed against the start-of-run proxy.
- Those tokens belong to the actions counted `scrubbed` (652) and `context_reconstructed` (475). Their text is re-rendered,
  not the ids the policy drew, and they can sit far in the tail.
- PPO's surrogate takes the unclipped side when the advantage is negative and the ratio is above 1 + eps. One such token
  then enters the gradient at its full ratio (norm 1.4e5 on step 19).

**Fix.** `splinter dataset build --sampled-only` (b879d37d2) leaves every record whose actions do not all carry their
exact sampled tokens out, counts it as `proxy_tokens`, and keeps its trial in the group's baseline. Training then runs
`behaviour: recorded`.

**Verified.** A 24-step diagnostic on round 1 rebuilt this way (1,902 records, 1,127 left out), with the same recipe, seed
and reference:
- ratio max per step 2.1-12.3, against rl1's 1.9e4, 1.3e5, 1.0e4, 1.6e6 and 3.1e4 on steps 6, 7, 15, 19 and 23;
- KL 0.003-0.029 (mean 0.018), clip mean 0.071, gradient norm at most 7.6, no guard skips.

**Still open.**
- The fix drops 48% of trained tokens. Exact contexts for scrubbed and request-log-rebuilt actions would bring them back.
- Closed 2026-11-20: brain bcd17ba8b bounds the negative-advantage side by dual-clip PPO (default `c` 3, per-step
  `clip_fraction_dual`), so a proxy that remains costs at most `c*|A|` and gives no gradient; splinter passes
  `--dual-clip` (db33cd729).
