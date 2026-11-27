<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 251. A model looping inside a tool call's arguments is a counted fail, not an endpoint rejection

**Lesson:** when the model repeats itself inside a tool call's arguments, sven asks once more, then ends the run with
`[sven:agent:error] tool-call loop: ...` and closes the stream; the gateway records that request as HTTP 499. The
worker read only `reasoning loop:` as a loop, and the classifier counted every 4xx up to 499 as `endpoint_rejected`,
a retryable infrastructure failure. The loop is the model's and recurs, so such trials were retried to exhaustion and
ended uncounted: 9 of 400 base-model trials of an 8B vim exam (2026-11-27), which marked its paired test incomplete
(`inconclusive`) although it read 25 wins to 0 losses. No brain server or router returned a 4xx in that window.

**Fix:** the worker reads either loop line as `agent_reasoning_loop`, which the classifier grades `ReasoningLoop`
(counted, failed, not retried); 499 is not an endpoint rejection. Spec:
`an_agent_that_hung_up_on_its_model_was_not_rejected_by_the_endpoint` (splinter-core `it::factory::classify`) and the
worker's `test_a_model_looping_in_its_reasoning_or_in_a_tool_calls_arguments_is_reported_as_a_loop`.
