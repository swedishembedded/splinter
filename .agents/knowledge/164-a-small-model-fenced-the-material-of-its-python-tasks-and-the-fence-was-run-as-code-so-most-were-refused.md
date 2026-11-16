<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 164. A small model fenced the material of its Python tasks, the fence was run as code, and most tasks were refused

**Lesson:** a refusal count by reason (`checks_failed`) hides what failed; before trusting a proposer's
yield, read the reasons of a sample. The task generator ran the model's `material` as written, so a program
wrapped in a Markdown fence (a line "```python" first) was a syntax error and the task was refused as one
whose checks failed. `shown` now takes the code out of a fence that opens and closes the material
(`tasks::shown::unfenced`).

Found 2026-11-16 in the live smoke of the multi-family lineage (Qwen3-4B as the proposer, Python 3 family):
69 of 119 base proposals and 51 of 63 in the next wave were refused as `checks_failed`; with the refusal text
printed, the cases were `SyntaxError: invalid syntax` at the fence line. After the fix the same family
admitted a few times more proposals per request, but `checks_failed` was still about 45% of the Python
requests (Lua about 30%): the other causes are not yet read.
