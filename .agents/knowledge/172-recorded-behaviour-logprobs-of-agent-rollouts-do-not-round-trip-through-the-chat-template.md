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

## Update 2026-11-19 (lane T): the answer was never the obstacle; the context is

Re-measured on 48 of lane R's practice trials (24 support, 24 compiled; 415 traced agent steps, every one `logprob_kind:
policy`, logprobs one per id). The 101-of-115 figure came mostly from the check itself: `rt.py` rendered the arguments
with Python's spaced `json.dumps`, while splinter writes them compact (as the model does). Rendered the way splinter
renders the step (text and reasoning trimmed, arguments re-serialised compactly), 345 of 415 answers round-trip
(83%); the rest are think tags the model wrote malformed (42), whitespace around a call (11), the same text tokenised
another way (15), other (2). Of the 122 traced actions the policy-gradient view writes for those trials, 119 round-trip.

What changed:

- brain b4ffeb71e trains the sampled `token_ids` themselves and admits them when they read back, by the template's
  reply format, as the record's message (122 of 122 of those actions), and takes `prompt_token_ids` beside them,
  refusing a record whose rendered context is not those ids (`prompt_token_ids_mismatch`).
- splinter 5a0c7cd43: the policy-gradient view writes each traced action's `token_ids`, `behavior_logprobs` and
  `prompt_token_ids` and the record's `tokenizer_digest`, and counts every other action as a proxy by reason in the
  manifest's `tokens` report.

Why the ratio is still not exact on these rollouts (each verified through brain's trainer on the view's records):
every recorded prompt differs from the trainer's rendering. The server put a reasoning-effort instruction into the
system turn that the Qwen3 template never writes (brain knowledge 531: 0 of 415 prompts match without it, 364 of 415
with it). Past it, the records' context was rebuilt from the trajectory, not read from the request log: the user turn
is the task's instruction without the harness's time line. So the view now writes ids only for a conversation read
from the requests its model was sent (`context_reconstructed` otherwise); on lane R's exports that is none, and those
records train as before against `--behaviour start`. Exact ratios need both: requests recorded as conversations, and
a served prompt without the instruction (`reasoning_effort: medium`, or the serving default changed with a fresh
baseline).

## Update 2026-11-19 (lane T2): every traced action exact, once the server renders the template

What kept the context inexact, each fixed at its source, measured on 8 of lane R's practice trials (support and
compiled, a pass and a fail of 4 tasks; 48 policy-gradient records, 47 traced actions, one baseline-only record):

- brain served a reasoning-effort instruction the Qwen3 template never writes (brain knowledge 531); brain 1bdd5400c
  serves the template's own rendering (Qwen3-4B pinned token for token) and keeps tool schemas in the request's order.
- a single agent's trial was never read from its request log (only harness conversations were); bench 0380a841
  declares the log's schema in the export manifest, and splinter 6e0ea27a1 reads every single-agent trial whose log
  divides among its agent steps from its requests (the harness's time line included).
- the log does not divide when the model answered a request with nothing (reasoning only; sven records no step and
  asks again) or the trial ended before an answer: now only requests that made a step are divided (2 of the 4
  compiled trials).
- the requests path trimmed the text around a call and re-serialised its arguments: Qwen3's template writes a newline
  before a call only after text, so `"\n\n"` and `""` render different ids (every action after the first differed at
  the first earlier answer). The context now keeps both as sent.
- the scrub renamed bench's sandbox account `agent` in `ls -la` output to `user` (11 compiled actions counted
  `scrubbed`); `agent` is now kept like `root`.

Result: the policy-gradient view counts 47 of 47 traced actions `exact` (before: 0, every single-agent trial
`context_reconstructed`). Through brain's trainer (`answers_as_asked` + `encode_rollout`, Qwen3-4B): 0 of 47 admitted
as recorded, because these rollouts were served by a build that injected the instruction (refused at position 3,
`prompt_token_ids_mismatch`); 47 of 47 admitted with that instruction's 37 ids taken out of the recorded prompt ids,
which is what the fixed server sends. Rollouts collected on brain 1bdd5400c or later are exact as recorded; older ones
are not, and train against `--behaviour start` as before.
