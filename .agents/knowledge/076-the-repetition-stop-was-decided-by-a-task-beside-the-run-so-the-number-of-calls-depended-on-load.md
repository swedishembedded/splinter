<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 076. The repetition stop was decided by a task beside the run, so how many calls a stagnating worker made depended on the load

A worker that gets the same answer to the same call `max_repeated_results` times is
stopped as going nowhere. The count was kept by the event watcher, a task that reads
sven's event stream and cancels the run when the bound is reached. The run does not
wait for it: under load the run kept asking while the watcher waited to be scheduled
(or while a slow observer held it), so the same scripted worker made 3 calls on an
idle machine and 22 in a full workspace test run
(`a_worker_that_keeps_getting_the_same_answer_to_the_same_call_is_stopped_as_going_nowhere`
passed alone and failed under load, and a worker with a 40 ms observer made 25).
The tests allowed "at most one beyond the bound", which hid how small the margin
was on a quiet machine.

Fix: the stop is decided at the point every further call passes through, the model
request. `RepetitionGuard` wraps the worker's model provider; on each request it
counts, in the turns after the last user turn, how often each (call, answer) pair
has come back, and when one reaches the bound it cancels the send and never
answers the request. The transcript the request carries holds every earlier call
and its answer, so the count is the same on every run. The watcher still counts
calls for the report and enforces the tool-call cap; it no longer decides
repetition. Spec: `crates/agent/tests/work.rs`
(`the_repetition_stop_does_not_wait_for_a_slow_event_consumer`, and the exact
count of 3 in `a_run_that_repeats_one_call_and_gets_the_same_answer_is_stopped_and_says_so`).

Not covered: the tool-call cap (`max_tool_calls`) is still enforced by the watcher
on `ToolCallStarted`, so it can overshoot the same way under load; it needs a
decision before the call runs, which the model request cannot give.
