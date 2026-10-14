# 014. A sandbox assumed its scratch directory existed

The first `practice` run on a fresh state root failed in its first stage that
ran a command in a tree: `qualifying the baseline: .../state/work/sandbox/
workspace/calls: No such file or directory`. Four of the five scripted specs
failed the same way, each in a state root of its own, and the one that passed
was the one that never ran a command.

The process sandbox made a directory per call with `tempfile` under its scratch
root and never made the root. A workspace's commands run under a root of their
own (`work/sandbox/workspace/calls`) which nothing made, so the first command of
a fresh state failed before it started. The sandbox's own specs passed because
each created its root by hand before building the sandbox, which is exactly the
assumption that was wrong.

The call directory now creates its root. `crates/sandbox/tests/workspace.rs`
runs a command under a root three levels deep that does not exist, and fails
with the original error when the creation is removed.

How to avoid it: a component that owns a scratch location makes it, and its
spec builds it over a path that is not there; a spec that prepares the
directory a component needs proves the component works only where someone did.
