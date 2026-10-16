<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 066. The family pre-flight killed the test runner and orphaned its Lua processes

Finding 044 left open which layer failed to stop the looping Lua processes. The
Rust process runner is not the one: it starts a command in its own process
group and kills the whole group at the timeout, and a test that starts a busy
loop two levels down (a shell that starts a shell that loops) finds the loop
gone half a second after a 500 ms timeout.

The layer that failed is `scripts/tasks/families.py`. `run_suite` used
`subprocess.run(..., timeout=...)`, which on timeout kills only the direct
child, the `python3 -m unittest` runner. Reproduced: a suite whose one test
runs `lua -e "while true do end"` under a 2 s timeout left that `lua` alive
with parent 1, in the process group of the dead runner, running at full speed.
A host that gives each `lua` a wall timeout of its own does not help, because
the timer died with the runner. The Lua hosts of caldate and tokenbucket also
allowed 30 s per test, so a mutant that looped in every test of a 250 test
suite could occupy a runner for two hours.

Fixes, one per layer: `run_suite` starts the suite in a new session and kills
the group on timeout; every Lua host sets a kernel CPU limit (10 s) on its
interpreter, which ends a loop even after its parent is gone, and uses a wall
timeout of 5 s. Specs: `scripts/tasks/test_runaway.py` (a looping grandchild
is gone after `run_suite` times out; every host's `lua` has a CPU limit) and
`crates/sandbox/tests/process_runner.rs` (the runner leaves no looping
grandchild).

Not covered: a process that leaves its group on purpose, and the death of the
supervising process itself (a killed `splinter` leaves the leader of its
command group running until the command ends; only the hosts' CPU limit bounds
what it started).
