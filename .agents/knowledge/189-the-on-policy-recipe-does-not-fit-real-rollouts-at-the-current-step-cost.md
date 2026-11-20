<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 189. The on-policy recipe does not fit real rollouts at the current step cost

**Lesson:** `--recipe on-policy-rl` assumes exact behaviour log-probabilities on every token and four passes over a batch.
On real practice rollouts neither holds, so plan a round by its records and step time before launching it.
- The policy-gradient view writes one record per traced action, not per trial.
- About half the supervised tokens are proxies without recorded log-probabilities, so `behaviour: recorded` is refused.

Measured 2026-11-20 (item 19): 640 trials of A2s1 on brain 1bdd5400c (exact-token build).
- **Records.** The view built 2,913 records (2,628 with an advantage; mean context 6.2k tokens).
- **Proxy tokens.** Only 1,541 of 2,668 traced actions were `exact`; 652 were `scrubbed` and 475 `context_reconstructed`.
  brain refused `behaviour: recorded` because 934,787 of 1,962,062 supervised tokens had no recorded log-probability.
- **Step cost.** The recipe's 4 passes are 1,314 steps of 8 records at 184 s/step (31 s of it start-of-run scoring), or
  66 GPU-h for one round. The round ran capped at 72 steps, 0.2 of a pass.

Next: make the exact path cover scrubbed and request-log-reconstructed actions, and cut the per-action re-encoding of a
shared context (one record per trial with several supervised turns) before more passes are affordable.
