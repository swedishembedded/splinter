<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 152. A reply the output limit cut was graded as a wrong answer, and one failed call failed a whole exam

**Lesson:** `reply_bare` dropped sven's `MaxTokens` event, so `ServedValidator` graded a generation cut by `max_tokens`
exactly like a finished wrong answer: an infrastructure outcome (the protocol's budget) counted as the model's competence.
And `answer_with` stopped at the first failed request, so an exam with one dropped call produced no per-item record of which
calls succeeded. A bare reply now says it was cut (`Reply::truncated`), carries the prompt ids, the generated ids and the
served identity when a trace is asked (`ChatOptions::trace`), and `ServedValidator::attempt_with` keeps every call's outcome
apart (`CallFailure::{TimedOut, Server, Cancelled}`); `answer_with` still refuses a partial exam, naming the failed task.

## Evidence

Qwen/Qwen3-4B on a GH200 through `samples/served-items` (X0 lane B, 2026-11-14): with an 8-token budget, 48 of 48 replies
are recorded as `truncated` (before: 48 wrong answers). 880 calls at the protocol's 1024-token budget: 0 truncated, 0 failed,
0 malformed, every call recorded. Unit tests: `a_bare_reply_says_when_the_output_limit_cut_it_and_carries_the_trace_asked_for`
(`crates/agent/src/chat.rs`) and
`a_failed_call_is_its_own_outcome_a_cut_reply_says_so_and_a_traced_exam_asks_for_the_trace`
(`crates/pipelines/tests/it/campaign/served.rs`).

The dev gate's `Outcomes` are still booleans: a truncated reply fails its task there, as a reply under a fixed budget is the
policy's; a caller that must not count truncation (a measurement of competence) reads `attempt_with`.
