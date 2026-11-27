<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 238. An authoring prompt whose worked example breaks its own size rule produces tasks at the example's size

**Lesson:** the generation-2 and generation-3 outcome-first prompts (bench `prompts/vim-manual/propose-outcome-g2.md`,
`-g3.md`) ask for files of 15 to 80 lines, but their worked example has files of 12 and 5 lines. The 27B author followed the
example: the median file stayed at 8 lines in both generations and only 15.7-17.1% of jobs have every file at 15 lines or more.
Output was not length-limited (median 1.7k of 16k completion tokens). Generation 2's tasks were measured as large and as hard as
generation 1's (workspace 11.9 against 13.0 lines, teacher pass rate 0.72 against 0.68).

**Fix:** a worked example that meets the specification, and a rejection of undersized jobs in the reproduce operator, so the
specification is enforced by code rather than by prose.

**How to apply:** when a prompt states a constraint, its examples must satisfy it; verify the authored distribution against the
constraint (lines per file, changes per job) before the tasks are used.
