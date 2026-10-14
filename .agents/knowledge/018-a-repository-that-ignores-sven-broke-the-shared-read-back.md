# 018. A repository that ignores `.sven` broke the shared read-back of a change

`WorkspaceEnvironment::diff_since` reads a worker's change back with
`git add -A -- . ':(exclude,glob).sven/**' ...`. Git refuses a pathspec that
names a path the tree's own ignore rules hide: `git add` exits 1 with
`The following paths are ignored by one of your .gitignore files: .sven`. Most
repositories ignore `.sven/` (sven writes its audit log there) and Python
bytecode, so the read-back of any worker run in such a repository failed, and
`solve_in_workspace` reported it as "cannot record the environment", which it
was not. The sandbox's own specs never met it: their repository ignores only a
log file.

The coding-loop sample had met it, privately. Its old checkout code unstaged
`.sven` after staging everything and said why in a comment, and its spec
repository ignores `.sven/` on purpose. The workaround lived in a duplicate of
the shared function, so the shared one stayed broken until the sample was moved
onto it: 13 of the loop's 15 specs failed with `staging the tree failed`.

The read-back now stages the tree without the exclusions, takes the excluded
paths out of the index with `git rm --cached --ignore-unmatch`, and keeps the
exclusions on the diff, which has no such check. The sandbox spec
`a_tree_whose_own_ignore_rules_hide_the_excluded_paths_still_has_its_change_read_back`
fails with the git message above before the change and passes after. The error
text of a failed read-back no longer says the environment could not be
recorded.

How to avoid it: a workaround kept in a copy of a function is a defect report
nobody filed. When deleting a duplicate, read what it did differently and find
which of those differences were repairs.
