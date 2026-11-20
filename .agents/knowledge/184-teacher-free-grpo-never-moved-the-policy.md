<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 184. Teacher-free GRPO never moved the policy

**Lesson:** a teacher-free arm that shows no gain has not tested the approach unless the policy actually moved. Read the
optimiser's own numbers (importance ratio, KL per step, clip fraction) before reading the exam. Our GRPO arm ran with a
ratio of about 1.002, a KL of about 0.002 per step and a clip fraction of about 1%: 67 steps at lr 2e-5 over roughly 270
informative trajectories, each seen once, with half the records from zero-variance tasks. That is an under-optimised
update, not evidence against self-improvement.

The headroom was there to harvest. On A2's own 1,280 rollouts over train tasks, pass@1 was 0.572 against pass@8 0.781
(legal 0.40 against 0.70, accounting 0.29 against 0.63), so about 0.21 was in support and unused. 35 of 160 tasks were
never solved in 8 attempts; outcome-only RL has no signal there, and only a teacher, a repair or a hint can supply it.

The literature agrees on both halves: RLVR mostly sharpens within the base model's support (arXiv 2504.13837), needs
hundreds to thousands of updates (arXiv 2504.20571), and for small models distillation beats RL (DeepSeek-R1);
teacher-free agentic gains are reported only at much larger scale, with thousands of tasks and many iterations
(arXiv 2508.03501, 72B).

What to do: target a KL of 0.02-0.05 and a clip fraction of 10-20% (recorded behaviour logprobs, token-sum loss,
lr 1e-4, 3-4 epochs per batch), sample tasks by p(1-p) and drop zero-variance groups, and measure the ceiling
(pass@16 - pass@1 on fresh held-out tasks) before spending GPU time. A KL still near zero is a defect, not a result.

Found 2026-11-19, expert review of the A2 and A6 arms. The ranked causes, the decisive experiment and how to read its
outcome are in `.agents/roadmap/experience-to-competence.md`, section "Why teacher-free learning showed no gain".

## Update 2026-11-20: the paper's earlier policy-gradient runs had the same signature

The optimiser logs of "The Frontier Must Grow"'s two negative policy-gradient results read the same way (per-step
`kl` to the base reference, `clip` fraction and mean importance `ratio`, medians over steps):

| Run | Steps x records | lr | KL median (steps >= 0.01) | Clip median | \|ratio - 1\| median |
|---|---|---|---|---|---|
| S2 Lua, Qwen3-4B, lineages r0-r3 | 2 generations x 24 steps | 6e-6 | 0.0025-0.0034 (1-3 of 48) | 0.9% | 0.001-0.002 |
| 27B gen 0 GRPO | 60 x 4 of 476 | 2e-5 | 0.0044 (4 of 60) | 1.9% | 0.002 |
| 27B gen 0 GRPO + self-distill | 60 x 4 of 440 | 2e-5 | 0.0061 (9 of 60) | 2.5% | 0.007 |
| 27B gen 1 frontier GRPO | 120 x 4 of 2379 | 2e-5 | 0.0077 (27 of 120) | 2.6% | 0.012 |
| 27B gen 1 frontier self-distill | 120 x 4 of 2378 | 2e-5 | 0.0064 (24 of 120) | 2.3% | 0.009 |

None reached the KL 0.02-0.05 and clip 10-20% this entry asks for; the few steps above 0.01 are isolated spikes (to
0.34 and 31.9 on single steps). The 27B runs drew each record at most once and saw 20-50% of their datasets. These are
under-optimised updates, not evidence that the method gives no gain. Sources: the training logs in the private HF
dataset `swedishembedded/qwen38-27b-flywheel` (`decisions/gen-{0,1}/generation-files.tar.zst`, `train-*.log`) and
`swedishembedded/splinter-flywheel-evidence` (`rsi2/lineages/r*.log`).
