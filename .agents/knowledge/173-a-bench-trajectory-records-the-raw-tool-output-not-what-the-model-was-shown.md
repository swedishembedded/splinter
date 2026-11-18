<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 173. A bench trial's trajectory records the raw tool output, not what the model was shown

**Lesson:** to continue or replay an examined attempt, take its observations from the trial's request log
(`model-requests.jsonl`), not from its ATIF trajectory: the headless sven runner writes each tool's raw output
into the trajectory, while the model was shown the stored text (an error marked `error: `, long output truncated,
a repeat note), and each request also ends with a time-left line the history never holds.

Found 2026-11-18 in the first repair smoke (splinter `experiences repair`, the 27B teacher on lane R's A2 practice
failures, sven a0235c71 in the sandbox): every fork whose prefix held a failed read was refused because the replay
produced `error: read error: No such file or directory (os error 2)` where the trajectory recorded
`read error: ...`; the request log shows the model saw the marked text. With the observations taken from the
requests (time-left line stripped) those forks replay. The same gap means a dataset rendered from the trajectory
alone shows a student context that differs from the one it acted on; views that render from the recorded
requests do not have it.
