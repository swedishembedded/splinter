<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 157. A run directory released a moment ago looked held, because a forked child still had the lock

**Lesson:** an `flock` taken with `LOCK_NB` can be refused by a process that never meant to hold it: every
child forked by any thread of the holder carries a copy of the descriptor until it execs, so "held" right
after a release is not proof of a second owner. Ask again for a bounded moment before refusing.

Found 2026-11-16 running the minimind acceptance specs (`samples/minimind/tests/acceptance.rs`) at 16
test threads: 4 of 7 specs failed with `run directory ... is held by another process` on temporary
directories no other process used, while the same specs passed at 6 threads. Each spec opens the
stage ledger of its run directory again right after the previous `Run` dropped it, while other specs'
threads were forking sven's shell tool children. With `StageLedger::open` asking up to 40 times, 50 ms
apart, all 7 passed at 16 threads; a run directory another process really holds is still refused, after
two seconds.
