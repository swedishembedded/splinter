<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 087. The coding-agent prompt and the request told the solver to reproduce with tests that are hidden

Two read-only trace audits (the 8B base on half A, three seeds of about 250 failed attempts
each; the 27B teacher over 372 attempts, 205 failed) found the same cause in the instructions.
`coding-agent-1` says "reproduce the symptom from the report with a short command or test", and
the request says each symptom names a test that is not in the tree. The only test a solver can run
is the visible suite, which is green before any edit: 35 of 52 right-file-wrong-fix attempts of
the 8B ran it, saw green and declared the fix verified. Roughly 45% of the 8B's attempts grepped
the symptom words as if they were a string in the code (106 of 253 in seed 2). The 27B
never grepped symptoms, but 180 of its 205 failed attempts never changed a line of source: 111
stopped on repetition (66 re-ran their own script that crashed on its own bug, 30 re-ran a
`find / -name` hunt for "other copies"), 37 hit the tool cap and 20 the wall clock. No
verifier misjudgement was found in either audit.

Fix: `coding-agent-2` (`splinter_core::prompt`, chosen with `--coding-prompt`, default still
`coding-agent-1`, byte-identical and pinned by its digest) states that the tests are hidden and the visible suite proves nothing, asks
for a direct call of the named function (`python3 -c`, `lua -e`), forbids grepping the report
and leaving the repository, and says a crashing script is read and changed, not run again.
The request has a second wording (`--instruction-version 2`, `WorkspacePlan::instruction`):
the source layout (directories and their source files, never tests), that each line is a
hidden test's name written as words, and how the hidden tests are run. The original wording is
unchanged, so no stored task's address moves. The tools append a directory hint to an empty
`find_file`/`grep` result (sven), which removes the `**/*_test.py` dead end (22 attempts of one
seed).

A multi-symptom task is one fault. The generator seeds exactly one textual mutant per task, and
all 249 reference patches of one 8B seed are one file, one hunk. The audit's "fixed `round` but
not `ceil`" (`5c34df8349c5`) was a misreading: the hidden test named `ceil moves toward
positive infinity` also asserts `round(5/2)`, so a test's name can describe less than it checks.
No per-symptom verdict was added; the V2 request says one fault causes all the symptoms and that a
name may describe only part of a test.

Not measured: the effect of any of this on pass@1. Each is an arm to run under a registered
amendment (spec: `crates/core/src/prompt.rs`, `crates/agent/tests/workspace.rs`,
`crates/knowledge/src/tasks/workspace/instruction.rs`).
