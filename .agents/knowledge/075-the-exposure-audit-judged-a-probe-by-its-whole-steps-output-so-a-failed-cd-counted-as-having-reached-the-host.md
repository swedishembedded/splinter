<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 075. The exposure audit judged a probe by its whole step's output, so a failed `cd` counted as having reached the host

`outside_workspace` marks an attempt exposed when a tool call names an absolute path
outside the episode's tree. Each hit was classed (refused, absent, reached) by
`output_of(step)`: the results of every call of the step joined into one text. The
27B teacher issues up to a dozen calls in one step, and the shell tool ends its output
with its own `[exit code N]` line, which is in neither list of refusal and absence
phrases. A `cd <project-dir>` into a path that is not there, which printed "No such file or directory" was
"reached" as soon as a sibling `read_file` returned text, or the command exited
non-zero. Every class also failed the audit, and a hidden path named in a call whose
read failed ("no such file") failed it as well.

Measured on the 370 CE1 pass-1 attempts (a Python port of the rule reproduced the
recorded hits on all 370): 152 attempts pass the hidden checks and 100 survived the
audit. 24 of the 52 removed passes were removed only by the pooled output, 6 by a
failed read of a hidden test file name, 1 by the solver re-reading its own fix; no
removed attempt saw hidden material (confined by Landlock, hidden tests laid over only
at verification).

Fix: each call is classed by its own result, the exit-code trailer is not a result,
denied and absent probes and failed reads of hidden paths are recorded under
`outside_probes` and no longer fail the audit; reached outside paths, hidden paths in
outputs and messages, shingles and the state root stay exclusions. Spec:
`crates/eval/tests/exposure_audit.rs` (`the_shells_exit_code_line_is_not_something_that_came_back`,
`a_probe_is_judged_by_its_own_result_not_by_its_parallel_siblings`,
`a_failed_read_of_a_hidden_path_saw_nothing_and_is_only_a_probe`). Registered as
Amendment 13; the published yield was computed under the old rule.
