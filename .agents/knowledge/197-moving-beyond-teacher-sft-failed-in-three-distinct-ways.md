<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 197. Moving beyond teacher SFT failed in three distinct ways

**Lesson:** "RL did not beat SFT" hides three different failures with different causes. Before reading a flat or
negative result, read the optimiser's KL and clip, the ratio maxima and the per-step skips, and split held-out results
from training-task results. Each pattern below points somewhere different.

| Pattern | Where seen | Cause | Read it from |
|---|---|---|---|
| The policy never moves | gen-1 GRPO A6 (+0.004 sealed) and the Frontier paper's RL runs, median KL 0.002-0.008 | an under-optimised or stuck update (one offline pass, start-proxy behaviour log-probabilities) | KL and clip far below 0.02-0.05 and 0.10-0.20 |
| It moves, then collapses | item 19 rl1, held-out -0.229, retention fails | records with proxy tokens scored against the wrong reference: ratio up to 1.6e6, gradient norm 1.4e5 (knowledge 188) | ratio max, KL spikes, guard skips |
| Training reward rises, held-out does not | item 19 RFT: training tasks +0.094 [+0.019, +0.172], held-out +0.039 [-0.013, +0.094] | sharpening within the practised families without transfer | training-task and held-out deltas side by side |

A fourth sign showed up as well: pass@8 falling while pass@1 stays flat (rl1g) is early diversity collapse, so track
pass@k and distinct solutions every round. Against all of these, SFT on about 420 verified 27B trajectories gave +0.383
on held-out families (sealed, signed). A strong SFT baseline is hard to beat, and an RL claim needs a run where KL is in
band with no guard skips before its result is read.
