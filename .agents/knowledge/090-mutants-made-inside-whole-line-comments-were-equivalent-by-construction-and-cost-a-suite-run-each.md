<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 090. Mutants made inside whole-line comments were equivalent by construction and cost a suite run each

The group-0 sqlglot admission (6 files, 40 mutants each, 240 faults) refused 63 as
`equivalent_mutant`. Re-running every one of the 240 mutants against the plain suite (an
independent count: 63 equivalent, 16 over the output cap, matching the generator) and reading the
63 diffs showed 17 of them were a number, comparison or boolean changed inside a `#` comment line
(`# Week truncation follows @@DATEFIRST, which defaults to 7`, a URL, `# WITH x AS (SELECT 1 y)`),
which no test can catch. Each cost two suite runs of 1301 tests before it was refused. The
`delete_line` operator already skipped comment lines; the other three did not.

Fix: `mutants()` changes no whole-line comment (`#` or `//`) with any operator. Spec:
`crates/eval/tests/mutation_oracle.rs`. Not covered, and still counted as `equivalent_mutant`:
numbers and booleans inside docstrings (6 of the 63), deleting `from __future__ import
annotations` (6), and code the suite does not reach (the rest, for example tsql's style-code
table and `WEEK_OFFSET`); the last are test gaps, not equivalences, and the reason name overstates
them.
