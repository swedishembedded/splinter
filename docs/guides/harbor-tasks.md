<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Harbor tasks and trials

A Harbor task is a directory (`instruction.md`, `task.toml`, `environment/` with a
Dockerfile, `tests/test.sh` that writes a reward, optionally `solution/solve.sh`);
a splinter workspace task is the same thing as a request, a tree, hidden files and
the commands that must fail before and pass after.

```sh
splinter tasks harbor discover ~/harbor-tasks        # which tasks can be imported, and why not
splinter tasks harbor import ~/harbor-tasks/calc --oracle require
splinter tasks harbor export TASKSET-ID --out out/   # splinter tasks as Harbor tasks
splinter experiences ingest-harbor TASKSET-ID trials/*   # Harbor trials as experiences
```

*Import* admits a task whose environment is a repository checkout: a Dockerfile with
`FROM`, `WORKDIR` and `COPY` (no `RUN`, `ADD`, stages or wildcards), no prebuilt image,
a shell `tests/test.sh`, one step, Linux. Anything else is refused with the reason.
The task is held to the bar of every workspace task: its check fails on the tree the
solver is given on every run, and with `--oracle require` the change `solve.sh` makes
(run on the host with `/solution` and the working directory mapped to the tree) must
pass the check. The task is graded by its own `test.sh`, run with `/tests`,
`/logs/verifier` and the working directory mapped to host paths; it passes only when
`reward` is exactly 1, read the way Harbor's verifier reads `reward.json` or
`reward.txt`, so partial credit is not a pass and a missing reward is a failure. A task
splinter exported carries its check in `[metadata.splinter]` and is rebuilt exactly:
export then import gives the same task, with the same content address. Harbor's
verifier does not enforce the check's protected and pinned paths; splinter's does.
Each imported task is recorded with where it came from (the checkout's commit, whether
it was dirty, and a digest of the task directory) outside the task, so the address does
not depend on where it was read; `tasks harbor discover` reads a task repository at its
revision and writes nothing.

*Trials* are read from `result.json` (or `results.json`) and `agent/trajectory.json`
(ATIF; a trial without one is refused, never reconstructed). The experience's verdict
is Harbor's: pass at reward 1, fail at any other reward, none for a trial with an
exception or no reward. `verify` leaves such experiences alone: they carry no patch.
