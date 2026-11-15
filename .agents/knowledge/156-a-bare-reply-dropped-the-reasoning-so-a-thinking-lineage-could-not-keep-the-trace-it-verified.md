<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 156. A bare reply dropped the reasoning, so a thinking lineage could not keep the trace it verified

**Lesson:** a reply type that models only the visible text cannot carry a verified chain of thought; keep what the
server streamed apart (reasoning, text, usage) at the one place a reply is collected.

Found 2026-11-15 adding verified reasoning traces to the Lua lineage (`samples/campaign-lua`,
`--reasoning on`).

`splinter_agent::chat::reply_bare`, the one call the served validator asks a model through, matched
only `TextDelta` events of the stream. A reply of a model that thinks carries its reasoning as
`ThinkingDelta` events, which were discarded: with thinking on, the answer was graded but the
reasoning that produced it existed nowhere, so no trace could be journalled or trained on. `Reply`
now has `thinking` and, when the server reports usage, `output_tokens`. Two consequences worth
remembering:

- An answer cut by the token budget has `truncated` set and, when the cut fell inside the reasoning,
  an empty text. A verifier that passes a cut text is not a verified answer: a verified trace needs
  `passed && !truncated`, and that is what the lineage's per-program `correct` counts with reasoning
  on (without it, `correct` counts `passed` as it always did, so old journals are unchanged).
- The Qwen3 chat template renders the last assistant turn after the last user turn with a think block
  whether or not its reasoning is empty (`brain/crates/data/src/chat_template.rs`: `is_last ||
  !reasoning.is_empty()`), so an answer-only record trained in thinking mode (`--train-on
  verified-traces+labels`) is `<think>\n\n</think>\n\n` followed by the answer, and a record with no
  `reasoning_content` renders the same bytes as one with `"reasoning_content": ""`. The record
  meter and the trainer read the line through the same brain code (`Asked::examples_of`), so they
  agree provided both are given the same `thinking` flag: it is the plan's
  (`Plan::train_thinking`), not a constant.
