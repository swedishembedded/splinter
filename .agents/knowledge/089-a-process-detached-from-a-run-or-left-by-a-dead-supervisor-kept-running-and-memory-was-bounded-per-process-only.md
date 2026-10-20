<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 089. A process detached from a run, or left by a dead supervisor, kept running, and memory was bounded per process only

During the CE7 work an orphaned `python3 -m unittest` (parent 1) of a sqlglot mutant was found at
145 GB resident, half an hour after its run, with the job near its memory cap. Reading the process
runner against that showed three gaps, all real whatever started that process:

- The kill at the end of a run reached the process group and the descendants still linked by
  parent. A process that left the group (`setsid`) and whose parent exited before the end was an
  orphan of init, linked to nothing, and survived a run that ended normally (the run then waited
  on its open pipe and failed with `OutputNotClosed`).
- When the supervisor itself died (killed, crashed, the machine's job restarted) nothing killed the
  run: the leader and everything below it kept running.
- `RLIMIT_AS` binds each process alone. A test command starts one interpreter per module or test,
  so a leak in each is bounded by none of them, and an address-space figure is not memory in use.

Fix (`crates/sandbox/src/process.rs`, `namespace.rs`): every run is a child that dies with its
supervisor (`PR_SET_PDEATHSIG`, with a check that the parent did not already change), and a run that
is not confined is put in the same user and pid namespace with its own `/proc` as a confined one
(the reaper is process 1: when it ends, or the supervisor dies, the kernel kills everything left in
the namespace; a host that forbids user namespaces runs it as before). The supervisor also sums the
anonymous resident memory of the whole tree every 100 ms against `ResourceLimits::tree_memory_bytes`
and kills a run that passes it; `ProcessOutput`/`CodeResult` carry `memory_exceeded`, and the call
reads as stopped by a limit (`timed_out`), which every consumer already treats as a resource
failure and not a verdict. The orchestrator gives the workspace's process sandbox a 24 GiB whole-
command default (`--workspace-memory-gib` sets it and the per-process bound); it is in the
environment record only when set, so environments made before keep their identity.

Consequence for tests: a program in the namespace reports a pid the host does not know, so a test
finds its processes by a marker argument in the host's command lines, not by a pid file.
Spec: `crates/sandbox/tests/process_runner.rs`. Not covered: a host without user namespaces gets
the group and parent-link kill only; container sandboxes use the container's own memory limit.
