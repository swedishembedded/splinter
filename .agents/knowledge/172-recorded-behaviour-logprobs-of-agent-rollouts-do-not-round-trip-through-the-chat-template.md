<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 172. Recorded behaviour log-probabilities of agent rollouts do not round-trip through the chat template

**Lesson:** a policy trace's token ids and log-probabilities belong to the text the model
generated, and a rendered training record holds the text the chat template writes from the parsed
turn. For tool calls they differ, so the sampler's own log-probabilities cannot be attached to a
re-rendered answer: brain's contract v2 refuses a `behavior_logprobs` that does not count the
answer's rendered tokens, and `token_ids` that its text does not render to. Until a record can
carry the generated text itself, a policy-gradient run on such rollouts uses
`--behaviour start` (exact only when the run starts from the policy that sampled them).

Found 2026-11-18 (lane R, A6 arm): Qwen3-4B + the pilot A2 adapter, sven a0235c71 with
`driver = "brain"` through bench's gateway, every agent step carries `completion_token_ids` and
`logprobs` (one more than `completion_tokens`, the end-of-turn marker included). Re-rendering 115
steps of the smoke trials as Qwen3's template writes an assistant turn (`<think>` block, content,
`<tool_call>` with `{"name": ..., "arguments": ...}`, `<|im_end|>`) and tokenizing gave the sampled
ids for 14 and other ids for 101. The difference is the arguments' JSON: the model wrote it compact
(`{"path":"case.md"}`), the template's `tojson` writes `{"path": "case.md"}`; a model that writes a
non-string where the tool takes a string (`"text": {...}`) differs too. So the sft-step and
policy-gradient views supervise tokens the policy did not emit in most tool-call turns, a small
off-policy shift that imitation tolerates and that an importance ratio would not. The check script
is in the lane's run notes (`rt.py`: decode the ids, render the parsed step, compare ids).
