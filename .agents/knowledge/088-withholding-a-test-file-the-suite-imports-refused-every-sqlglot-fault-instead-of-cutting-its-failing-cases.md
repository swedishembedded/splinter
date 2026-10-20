<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 088. Withholding a test file the suite imports refused every sqlglot fault instead of cutting its failing cases

The re-admission of sqlglot with the Amendment 15 generator admitted 0 tasks in all ten groups
(group 0: 240 faults, `hidden_imported_by_tree` 128, `equivalent_mutant` 63, `broken_build` 28,
`output_truncated` 16, `flaky` 5). The first admission, before the rule, had admitted tasks
whose solver tree could not run its dialect tests. The cause is the repository's layout: 33
modules under `tests/dialects/` import `Validator` from `tests/dialects/test_dialect.py`, and a
fault in dialect code fails a case of that module too. The generator withheld the whole failing
test file, so every importer failed in the solver's tree; refusing that was right (A15.3), but
it was the only outcome for most faults.

Fix: the failing test files are still withheld whole, and when the tests that stay then do not
all run, a second tree is made in which those files stay and only the failing cases are
removed (`cut_unittest_cases.py`: the parser's line range of each method with its decorators and
the comment lines directly above, `pass` for a class left empty, refused when the result does
not parse or an id is not a method of the module). The overlay is the same (the whole files),
the cut file is protected, not a hidden path (reading it is no exposure), and the hidden text
the self-containment and exposure checks use is what the cut tree lacks. The hidden-strength
measure counts a variant as caught when a removed case, or a withheld file's test, fails.
Spec: `crates/knowledge/tests/workspace_shared_helpers.rs`. Not covered: a cargo runner's
sources are not cut (the refusal stays), and a test inherited from a base class or generated at
import time is not found as a method of its module and is refused.
