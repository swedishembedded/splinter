<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 104. The harness-edit tests author documents the v2 compiler refuses

`make test` is red on `main` in `crates/agent/src/harness_edit.rs`: two of its tests build their
fixture from a step-map document (`api_version: "1"`, `entry:`, `steps:`), and the compiler
refuses that language outright - "`api_version: \"1\"` documents (the step-map language) are no
longer accepted; write a workflow with `api_version: \"2\"`". The tests therefore assert on a
failure message that is now the parse error of their own fixture, not the edit mechanism.

Bumping the fixture's version is not enough, and that is the interesting part: a v2 document's
top level is `variables, imports, on_failure, api_version, name, description, inputs, outputs,
agent, execution, capabilities, gates, workflow, success` - the steps live in a `workflow` LIST,
whereas `harness_edit::apply` edits a value at a PATH OF KEYS through mappings. So the feature's
input format and its edit model no longer meet: either `apply` learns list paths (an index in
the path), or the harness-edit flow is redefined for workflows. That is a decision for the
harness workstream, not a test fix, and it is why the two tests were left red rather than
rewritten on 2026-10-25.

What is unaffected: `make check` (gates, fmt, clippy, paper claims) is green on the same commit,
and the sample crates' own tests pass. The observation was made while landing an unrelated
change; the failure reproduces without it.

Evidence: `runs/g8/tests-splinter-workspace.log` (the two panics and the compiler message),
`crates/agent/src/harness_edit.rs` (the `DOC` fixture, `apply`'s path walk), and sven's
`crates/harness-spec/src/lib.rs` (the v2 field list).

**Resolved (2026-10-26).** `apply` learned list paths: a path segment that is a number indexes a
list, so `["workflow", "0", "task"]` edits a v2 task's prompt, and the proposer's brief now
describes the v2 workflow instead of the removed step language (it asked models for edits the
compiler could only refuse). The unit tests edit a v2 document.
