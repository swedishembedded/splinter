<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 086. The solver's shell gave it no scratch space and showed it the harness's own log

In the 27B teacher sets 28 attempts hit `Permission denied` writing to `/tmp` (30 times), 13 changed directory into a
project path they had invented, 91 opened `.sven/audit.jsonl`, and 14 left scratch scripts (`t.lua`, `repro.lua`) in
the patch. The sandbox was doing what it should (Landlock lets a command write its tree and a per-call directory
only); the model was not told, so it guessed.

* The shell's description now says it: every command starts in the project directory (no other directory to change
  to), `/tmp` is not available, `$TMPDIR` (also `$HOME`) is. The text is part of the tool schemas a trajectory
  records, so a record shows the model what it was told. It does not name the random path of the tree.
* `$TMPDIR` is a directory of the episode outside the tree, kept between the episode's commands
  (`Sandbox::run_command_with_scratch`; the per-call directory it replaces was removed after every command), so a
  script written there can be run by the next command and is never in the patch.
* sven writes its audit log beneath the project root's `.sven`. The tree's `.sven` is a link to a directory of the
  episode that the sandbox does not allow a command to open, and the diff leaves `.sven` out. A change in sven
  (a place for its state other than the root) would remove the link; none is needed to remove the exposure.
* `PYTHONDONTWRITEBYTECODE=1` is a variable of the sandbox's environment, so it is in the environment record and
  changes the address of every task generated with it. It is therefore `--workspace-env`, off by default: with it
  as a default no task already in a store could be solved by the new build (the solve refuses a task whose record
  it does not reproduce). A campaign that wants it generates its tasks with the flag.

The container sandbox already has a writable in-memory `/tmp` per command and ignores the scratch directory.
