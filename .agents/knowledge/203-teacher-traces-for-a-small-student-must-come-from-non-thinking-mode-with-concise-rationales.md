<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 203. Teacher traces for a small student must come from non-thinking mode with concise rationales

**Lesson:** capture a teacher's trajectories for a small student in the mode the student should act in.

- Turn thinking off (`chat_template_kwargs.enable_thinking = false`) and use the model's non-thinking sampling
  (Qwen: 0.7 / 0.8 / 20).
- Ask in the system prompt for a one- or two-sentence rationale before each tool call. The visible text before each
  action is then the reasoning trace.
- A thinking-mode teacher (thinking sampling 0.6 / 0.95 / 20) produces long think blocks. The student copies their
  length and not their competence, and runs out its time budget (knowledge 202).
- Check the mode in a recorded request, not in the profile. With sven-1db749d9 a `model:`-section setting such as
  `enable_thinking`, or `driver_options` placed next to a named provider, was dropped silently (bench 073). The prompt
  also named tools the session did not offer (sven 0d94638e). sven-7a9f60cb carries both fixes.

## Evidence (lane VSFT, item 25, 2026-11-21)

| teacher mode | reasoning characters per supervised action (mean / p50 / p90) | exam timeouts of the students (E1b) |
|---|---|---|
| thinking on (v1) | 1,199 / 566 / 2,897; one turn up to 23k | 77-84% (base 21%) |
| thinking off + concise instruction (v2 smoke, 4 trials) | 0 in a think block; the visible text before a call is one sentence | pending (item 25 E2) |

- In non-thinking mode the 27B still explored and sometimes looped: one smoke pass made 102 vim calls before it acted.
  Curate per task for the fewest actions as well.
- The v2 yield and the E2 result are recorded in the experience-to-competence ledger, item 25.
