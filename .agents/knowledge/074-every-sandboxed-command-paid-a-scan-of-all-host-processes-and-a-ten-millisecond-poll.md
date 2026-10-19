<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 074. Every sandboxed command paid a scan of all host processes and a ten-millisecond poll

`pipelines/tests/generations.rs` ran for 5001 s in one gate log (`replay_follows_the_policy...`
runs the two-generation fixture four times), and a single test of it was still running
after 580 s alone. Stack samples of the test thread showed the time in
`splinter_sandbox::process::run`: a campaign fixture runs thousands of short git, tar and
test commands through it, and each paid two fixed costs.

1. `kill_group` listed the run's descendants by reading `/proc/<pid>/stat` of every process
   of the host. On the shared 144-core node (3500 processes) the scan took 0.12 s, and many
   concurrent runs made it worse. The kernel lists a task's children in
   `/proc/<pid>/task/<tid>/children`; the walk now reads only the tree, and a kernel without
   that file gets the old scan.
2. The wait loop slept a fixed 10 ms between looks, so a command that ended at once cost 10.5 ms
   (100 runs of `/bin/true`). The wait now starts at 100 us and doubles up to 10 ms: 0.93 ms.

One test of `generations.rs` (`the_full_matrix...`, alone, one thread, same node) took over 580 s
before and 320 s after both changes. The fixture is still command-bound (the test process is
at about 9% CPU while git and the checks run); making it cheaper means fewer commands, which
is a change of the fixture, not of the runner.

How to avoid it: a primitive that sits under thousands of calls gets its fixed cost measured
(100 runs of a no-op) before anything built on it is called slow, and a scan of a shared
namespace (`/proc`, a directory of every tenant) never sits on a hot path.
