<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 183. A killed campaign leaves the examiner's processes running

**Lesson:** a `splinter campaign run` stopped by SIGKILL takes none of the processes it started with it. Bench's
`gateway serve`, `harbor run` and every trial sandbox below them (`unshare` with Sven inside) are reparented to init and
keep driving the brain endpoint, writing into the stage directory the resumed run is about to replace. The ledger
itself resumes correctly (the next run takes over the open attempt, removes the partial job directory and starts a fresh
one), but only after those processes are gone. Kill them by PID (the run's children and their descendants, plus any
sandbox already reparented to init) before the next `campaign run`, until splinter starts its examiner children in a
process group it can take down, or with a parent-death signal.

Found 2026-11-19 (lane CR), in the first real campaign run (Qwen3-4B + A2s1 champion, brain 1bdd5400c, bench 225bbe5f,
splinter 67b1f60d9 and d1cac5987). Each of three SIGKILLs (mid-rollout twice, mid-evaluate once) left the harbor and
gateway processes and 80 to 120 descendants alive; one trial sandbox had already been reparented to init by the time
the run's children were listed, so a descendant walk from the dead run missed it. With them killed by PID, every
resume began at the first stage without a commit: the rollout as attempt 2 under the same key, the train stage
answered by brain's job under the same `resume_key` without training again (one brain run directory, 30 steps
logged once), the evaluate stage from fresh job directories.

Two smaller observations of the same run:

- `campaign status` cannot be read while a run holds the campaign (`ledger.lock is locked`), so there is no
  "status before the kill" from the status command; the ledger file itself is readable.
- A resumed train commit equals an uninterrupted one in key, outputs, candidate id, adapter digest and steps, but not
  byte for byte: its `gpu_hours` is the wall time of the attempt that committed, and its model reference names the
  adapter under the state root. A ledger copy cut after `g0001/ingest` and run uninterrupted under another state root
  recorded the same rollout, ingest and curate commits (dataset `blake3:0a4eeee1...`), and a train commit differing only
  in those two fields.
