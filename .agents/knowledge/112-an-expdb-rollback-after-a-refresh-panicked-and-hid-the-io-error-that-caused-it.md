<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 112. An expdb rollback after a refresh panicked and hid the I/O error that caused it

A guarded write refreshed the expdb session and then failed on the disk. The rollback that handled
the failure still held the mark taken before the refresh, which was now past the end of the
refreshed state, and panicked (`split index (is 3) should be <= len (is 0)`). The panic replaced the
real error, so the report said nothing about the full disk behind it. It was found while diagnosing
the pipelines tests of #111, where the underlying cause was EDQUOT.

`Session::rollback` now treats everything pending after a refresh as written after the mark and
forgets it, so the original error propagates (splinter 50bb57920, regression test in
`crates/expdb/tests/it/session.rs`).

Rule: error-handling code runs when invariants are already broken. A cleanup path must not assume
state it was handed before the failure still holds, and must never replace the error that caused it.
