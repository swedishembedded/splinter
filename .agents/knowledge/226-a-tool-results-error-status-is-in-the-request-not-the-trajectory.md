<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 226. A tool result's error status is in the request, not the trajectory, and the marker alone is no evidence

**Lesson:** to tell which tool calls of a recorded Sven trial failed, compare what the model was shown with what
the tool returned. The ATIF trajectory keeps each tool's raw output and no status (`ObservationEntry` has no
error field, and the headless runner writes `ToolCallFinished`'s output). The request log keeps the stored text,
which Sven writes as `error: ` followed by the output when the tool flagged the result (MCP `isError`), then
maybe truncated, a repeat note and the time-left line. A prefix test on the shown text alone misreads tools
whose own output starts with `error: `, such as a command-line parser's message. Those are shown unmarked,
because the call succeeded. `splinter_data`'s `tool_errors` counts a result as an error only when, after the
marker is removed, the shown text matches the output at least as far as it did with the marker in place. Sven
5.9 also files a result on a step of its own after the step that made the call. So a result is paired with
the latest call of its id, not with its own step.

## Evidence (2026-11-26, transfer-arms)

- Of 677 tool results in 15 recorded trials of an earlier firmware run (`model-requests.jsonl` against
  `trajectory.atif.json`), 6 were shown as `error: ` plus the trajectory's output. 5 more began with `error: `
  only because the tool's own output did (`error: unexpected argument '--board' found`), and the request showed
  that output unchanged. A prefix filter would have counted 11 errors; 5 of them were not errors.
- In 2 of the 677 the shown text did not begin with the output: one kept only an omission marker after
  truncation, and in the other the tool's `\r\n` was shown as `\n`. Neither carried the marker.
- `sven-5.9-reasoning` (factory fixtures) files the observation of the call made at step 2 on step 3.
