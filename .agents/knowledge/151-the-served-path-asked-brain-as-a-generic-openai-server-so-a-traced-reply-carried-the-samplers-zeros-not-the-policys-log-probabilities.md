<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 151. The served path asked brain as a generic OpenAI server, so a traced reply carried the sampler's zeros, not the policy's log-probabilities

**Lesson:** a model requested from brain endpoints (`ModelSelection::load_served`, every `ServedHost` exam, rollout and
proposal) was built as `openai/<id>`. sven's generic OpenAI driver asks for a trace as `logprobs` plus `return_token_ids`,
which brain answers with the sampler's truncated figure: 0.0 for every token of a greedy reply, and no stop token. Only sven's
`brain` driver asks for `policy_trace` (the policy's own log-probabilities over the whole vocabulary, scored by the serving
model, the stop token included). The request is now built as `brain/<id>`, so a traced served reply is brain's policy trace.

## Evidence

Qwen/Qwen3-4B, `--kv-cache f32`, 48 held-out compiled items, greedy, thinking off (X0 lane B, 2026-11-14). Before: 36
generated ids per reply for the first item (no `<|im_end|>`), log-probabilities `[0.0, 0.0, 0.0, ...]` on all 1212 tokens,
no `chat_template_digest`. After: 1260 ids (the 48 stop tokens included), identical bit for bit to a direct
`policy_trace: true` request on another card of the same build (max |difference| 0.0), with the template digest reported.
Put through brain's numerics gate as rollout records with `token_ids`, the trace's mean |gap| to the trainer is 0.0015 (all
48 records round-trip). The prompt ids the model read equal a direct request's on 48 of 48 items either way: the driver
change alters what a trace reports, not what is asked.

Regression test: `a_traced_request_asks_the_endpoint_for_the_policys_own_trace_not_the_samplers_figure`
(`crates/model/tests/endpoint.rs`), red before the change.
