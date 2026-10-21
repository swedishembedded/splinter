<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 092. The replayed agent's shell could read the hidden diff, and only the gate noticed afterwards

The repo-replay agent ran with sven's own `shell`, which starts in the attempt tree but is not
confined: a command could print any file the process can read. The case directory beside the tree
holds `case.json`, which has the whole unmasked diff and the plan; the source repository and the
mirror of the history were readable too. The gate (`forbidden_shell`, `outside_tree`, the
`forbidden_locations` check) only judged the calls after the fact, so a record could still carry
the content in a tool result when the call was not recognised as a reach (a glob, a `cat` of a
relative path).

Fix: the attempt runs through `splinter_agent::harness_work::run_harness_in_tree` on an engine built
with `confined`, which replaces `shell` with the sandbox's (`SandboxShell`, Landlock on the process
sandbox). A command sees the attempt tree and its own scratch directory (`$HOME` and `$TMPDIR`)
and nothing else of the disk; the harness's state is kept outside the tree and `.sven` in the tree
is a link to it, which `read_tree` ignores. The gate commands (`repo-replay check*`) are run by
the host, not by the agent's shell, so the case directory stays readable by them only. The gate
rules stay as a second line of defence. `solve_harness_in_workspace` runs through the same
function.

What changes for a run: `git` works inside the tree (its own repository); `/tmp` and the cargo
registry are not readable from the agent's shell (the project checks run by the host are not
affected); a command that reads a path outside the tree now gets an error instead of the content.
Specs: `samples/repo-replay/tests/confinement_spec.rs` (source repository, mirror, case file) and
`crates/agent/tests/harness_work.rs` (`a_harness_run_in_a_tree_gets_no_answer_from_a_file_beside_the_tree`,
red with an unconfined shell).
