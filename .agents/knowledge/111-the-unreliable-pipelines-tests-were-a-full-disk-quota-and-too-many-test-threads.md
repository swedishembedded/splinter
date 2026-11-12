<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 111. The "unreliable" pipelines tests were a full disk quota and too many test threads

`splinter-pipelines --test it` (354 tests) failed 4 to 67 tests when `cargo test` ran it one test per
core (288 on the development node), different ones each time, and each reported test passed alone in
seconds. The failures were not flakiness in the code. Measured: they were `Disk quota exceeded`
(EDQUOT) on the scratch directories the tests create (the project drive has an inode quota, see
#083), plus timing assertions ("three requests are in flight together") starved by the
oversubscription. With the scratch directory emptied and 8 tests at a time, all 352 tests that
finished passed (two slow ones had not when the run was cut at 28 minutes).

`make test` now bounds the threads (`--test-threads=$(TEST_THREADS)`, 16; splinter 0385f207a).

Rule: before treating a test as flaky, read the error (EDQUOT is an environment fact, not a race) and
re-run at low parallelism on a clean scratch directory. A timing assertion that holds at 8 threads
and fails at 288 is a test-design limit, not a product bug. Evidence: ledger `learning-methods.md`,
"The unreliable pipelines tests" (68337a208).
