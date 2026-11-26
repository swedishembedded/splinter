<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 225. Sandbox image copies in memory exhaust a node; an interrupted batch could not resume

**Lesson:** every userns sandbox unpacks a private, writable copy of its image. Held in a tmpfs, the copies count
against the job's memory limit as shared memory, and nothing bounds how many sandboxes run at once on a node. Two runs
with about 900 attempts in flight on one node took 290 GB of shared memory on top of 145 GB of model servers; the
job's 411 GB limit was reached and the kernel killed the brain servers, both runs and the session that drove them,
while the node's local disk had more than a terabyte free. The copy is now a directory of the sandbox on the local
disk the trial directory is on, removed when the sandbox stops; it stays in memory only on a network filesystem.

The same crash showed a second defect: after it, `splinter-harbor-worker reconcile` exited 1 because the batch had
attempts with no finished trial, and the factory takes any non-zero exit as the backend failing, so the stage failed
on every resume instead of running those attempts again. Unfinished attempts are what reconcile exists for: it now
exits 0 and lists them as `missing`, and the factory records them lost and runs them again.

## Evidence (2026-11-26, n457, job 5790228)

The job cgroup's `memory.events` counted 10 OOM kills; the kernel's report at the first: `anon 145048600576`,
`shmem 289875623936`, limit 430276608 kB. After the fix the material run resumed: its student pass went from a
failed stage to 7982 running and 226 pending slots of 8208.
