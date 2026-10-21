<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 091. A failed sub-case was read as a second test and the fault was refused as a broken build

In the group-0 sqlglot admission 28 faults were refused as `broken_build`. Running all 240 mutants
against the plain suite and feeding each unittest report to the reader showed 19 truly break the
suite (a module no longer imports, so most of the 1301 tests never run). The other 9 killed
tests normally: 8 deletions in `parsers/sqlite.py` that fail 73 tests and one in
`parsers/bigquery.py` that fails 46. Their failing tests use `subTest(dialect=...)` or
`subTest(value=...)`, and the summary header of a failed sub-case is `ERROR: name (id)
(dialect='sqlite')`. The summary reader took the text after the id's opening bracket up to the
last bracket as the id, so it recorded a second, invented failed test
`id) (dialect='sqlite'.name`. The ran set then differed from the baseline's and the generator
refused the fault as a broken build; the `[message]` form of a sub-case header was ignored.

Fix: the failure summary reads a test as `name (id)` and drops what follows the id's closing
bracket (an id holds none). Spec: `a_test_whose_sub_case_failed_is_one_failed_test_named_by_its_id_alone`
in `crates/eval/src/verifiers/test_report.rs`. After it the replay of the same 224 reports gives
19 broken, 63 equivalent and 142 killed. The cost of the defect was yield, not soundness: a
fault in code covered by sub-case tests could not become a task.
