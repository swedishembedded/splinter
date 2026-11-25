<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 224. A program this process just wrote is busy while a sibling thread forks

**Lesson:** a script written by one thread (a fleet's step script, a test's stand-in `brain` or `srun`) cannot be
executed while any process forked by another thread in the meantime still holds the descriptor it was written
through: the kernel refuses `execve` with `ETXTBSY` ("Text file busy") until that child execs and its inherited
descriptor closes. Tests that write stand-ins and start servers in parallel threads hit it at random, and so can a
process that writes a launch script while it starts other children. Every server start
(`splinter_model::server::spawn_when_written`) now retries `ETXTBSY` up to 50 times 10 ms apart; any other error is
returned at once. This was the cause of the one-off failure of
`the_server_a_run_owns_serves_the_store_..._as_wide_as_it_decodes` the distributed-jobs ledger kept open.

## Evidence (2026-11-25, lane tts-lane-i)

With one more fleet test in `crates/model/tests/service.rs`, `cargo test -p splinter-model --test service` failed 3
of 6 runs on n80, each time `... could not start: Text file busy (os error 26)` in a different test; with the
retry, 8 of 8 runs passed.
