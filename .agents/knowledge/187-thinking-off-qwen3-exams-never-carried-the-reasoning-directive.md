<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 187. Thinking-off Qwen3 exams never carried the reasoning directive

**Lesson:** brain's served-prompt injection (brain knowledge 531: a 37-id "Reasoning effort ..." system instruction in
every Qwen3 prompt before brain 1bdd5400c) is gated by `enable_thinking`. A request with thinking off renders the
checkpoint template exactly, on old and new builds alike, so direct-answer exams served that way compare across builds;
only results served with thinking on (agent runs, thinking baselines) changed prompt with 1bdd5400c. Read the rendered
prompt ids (`policy_trace.prompt_token_ids`) before assuming a build change moved a result.

Measured 2026-11-20 on gpu2: a deterministic 300-item sample of "The Frontier Must Grow"'s 889-item Lua exam, Qwen3-4B
base and S4 r0/3, the paper's direct-completion protocol (user turn only, thinking off, greedy, 64 concurrent,
f32 KV), each of brain ad44162a7 (injecting) and 1bdd5400c served twice. All four runs gave the same generated ids
and log-probabilities on every item: base 6/300, r0/3 65/300, gain +0.197 (61 wins, 2 losses). Thinking off, both builds
render 58 ids on average with no system turn; thinking on, ad44162a7 renders a system turn starting `Reasoning`
(id 25139), 42 ids longer than 1bdd5400c. The gate is in `qwen3::chat::render_prompt` (no `reasoning_effort` when
thinking is off) and is the same at brain 796871647, the paper's learner build. Co-batched greedy serving of this
workload was bit-reproducible within a build, so its noise band was zero. Details: roadmap
`recursive-improvement.md`, "Paper revision after the experience-to-competence work", item 7.
