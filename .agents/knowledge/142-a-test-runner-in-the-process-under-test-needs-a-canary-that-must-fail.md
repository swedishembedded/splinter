<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 142. A test runner in the process under test needs a canary that must fail

A bug-fixing verifier runs the project's tests, and the code under test is imported into the runner's own process, so a "fixed" library can patch the test framework (`unittest.TestCase.run` replaced by a function that reports success). With the plain verifier that patch earned a full reward on a generated instance whose two faults were not fixed; with one always-failing canary test added to the run, and the verifier requiring that it failed, the same patch earned zero (`the_canary_is_what_stops_a_library_that_patches_the_test_framework`). The other attempts the repository plugin is attacked with (exit at import, forged result lines, everything skipped) are stopped by three other rules: the result line carries a nonce read before anything of the workspace is imported, the reward is written by the parent process that imports none of it, and the number of tests that ran must equal the pinned size of the suite.

A fourth item found on the way: grading changed the workspace it graded (Python wrote `__pycache__` into the library, so running the reference twice gave different workspace digests); the runner now sets `sys.dont_write_bytecode`.

Rule: a verifier whose oracle runs in the same process as the solver's code must include a check that cannot honestly pass, and must not write its reward from that process. Evidence: `crates/knowledge/src/factory/plugins/repository_bugs.rs`, `crates/knowledge/tests/it/factory_store.rs`.
