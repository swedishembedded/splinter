<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 044. Mutated Lua programs outlived their test runs and loaded the node for hours

While the repository families were being measured with `splinter practice`, the host
load average stood between 143 and 225 on a machine with 72 cores. A process listing at
06:47 showed at least 14 `lua -` processes, each using about half a core, with elapsed
times between 2 h 04 min and 2 h 25 min, and one `python3 -m unittest` run of 4 h 32 min.
Their command lines named the temporary trees of the Lua families (`graphkit` among them).
After they were killed by name the load average began to fall (157 -> 144 within seconds).

What this shows: a mutated copy of a Lua family can loop forever, and the process that
runs it was not stopped when the test run ended. The Lua families are driven from Python
`unittest` files that start one `lua` process per test, so the program under test is a
grandchild of the runner.

What is not established: which layer failed to stop it. Candidates, unchecked: the
bounded process runner kills the process group of its child but the `lua` process is in
another group; the wall-time limit of the test command is longer than the budget of the
whole run; or the kill reached `unittest` and not its children. The measurements of the
funnel for the Lua families were slow for the same reason (the host was loaded), so no
funnel numbers exist for them.

Consequence for the next change: a family test harness must start its interpreter with a
hard timeout of its own (for example `timeout` or a deadline in the Python wrapper), and
the runner's process-group kill must be exercised by a test that spawns a looping
grandchild.
