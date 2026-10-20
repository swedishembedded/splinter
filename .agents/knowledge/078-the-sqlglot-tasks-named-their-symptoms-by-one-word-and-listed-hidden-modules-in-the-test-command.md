<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 078. The sqlglot tasks named their symptoms by one word and listed hidden modules in the test command

A teacher pass over 112 sqlglot tasks passed none, and a regeneration over 104 passed none.
Every task was valid: the untouched tree fails the hidden checks and the reference fix passes
them (1300 tests in about 22 s). The requests were the cause. A failing test was named by its
method name alone, and the sqlglot tests are named after dialects and areas, so a request read
`databricks`, `time` or `cast` (2.07 words per label, against 5.1 in ttlcache) while every
in-tree test passed and the failing ones were withheld: nothing could be reproduced in a
388-file library. 108 of 112 attempts ended with an empty patch, and 82 chased a stale-bytecode
hypothesis for lack of anything else. The request also gave the whole suite's command, which
names the hidden test modules (every run printed `ModuleNotFoundError` for them, 161 results),
and in 28 tasks the withheld file was the base module every in-tree dialect test imports, so
none of them could run in the solver's tree at all.

Fix (Amendment 15): the command names only the modules in the solver's tree; a label of fewer
than 3 words is rendered from the failed assertion's own message (an equality's two values, or
an error the code raised), never a traceback frame, a diff, a path or the call, and a candidate
that cannot be rendered is refused (`symptom_not_actionable`); a candidate whose withheld module
the kept tests import is refused (`hidden_imported_by_tree`). The shown values are the one
disclosure the task makes: the admission check masks them, and the exposure audit counts the
request as text the solver may see. Spec: `crates/knowledge/src/tasks/workspace/instruction.rs`
(unit tests) and `crates/knowledge/tests/workspace_tasks.rs`
(`a_module_that_tests_in_the_tree_import_is_never_withheld_whole`,
`a_symptom_too_short_to_act_on_is_rendered_from_the_failed_assertion`,
`a_short_symptom_whose_assertion_cannot_be_rendered_refuses_the_task_with_a_count`),
`crates/eval/tests/exposure_audit.rs`. Not covered: only `unittest` failures are read per test
(a cargo runner's failure message is not), so a cargo task with a short name is refused.
