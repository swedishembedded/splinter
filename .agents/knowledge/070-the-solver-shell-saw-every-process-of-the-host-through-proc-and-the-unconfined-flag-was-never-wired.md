<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 070. The solver's shell saw every process of the host through /proc, and `--allow-unconfined-workspace` was never wired

Found 2026-10-18 writing the adversarial containment fixtures (`crates/sandbox/tests/containment.rs`).

Landlock confinement of a workspace command allowed `/proc` read only, and the suspicion was that a
command could read `/proc/<pid>/environ` of any process of its own user, the keys in the
supervisor's environment among them. It cannot: the files that need ptrace access (`environ`,
`mem`, `maps`) are refused by Landlock's own ptrace check for a process outside the command's
domain (`crates/sandbox/tests/containment.rs`, `another_process_of_the_user_is_invisible_through_proc`
shows the refusal and, unconfined, the secret). An earlier version of this entry said the opposite
and added a non-dumpable supervisor to compensate; that was based on a test that could not tell the
two causes apart, and it is removed. What Landlock leaves readable is the rest of `/proc`: that
other processes exist, whose they are and what they are called (`status`, `stat`). Denying the
numeric `/proc` directories with Landlock is not possible, because coreutils on this host reads
`/proc/self/auxv` and `self` is a different directory in every process. The command therefore runs
in a private user and pid namespace with a `/proc` of its own (`crates/sandbox/src/namespace.rs`),
where no other process exists. Its traps: the supervisor's `spawn` waits for the descriptor that
reports a failed `exec`, so the processes that hold the namespace open must close every descriptor
but the standard three (otherwise `spawn` returns only when the command ends, and a command that
waits for input written after `spawn` deadlocks); process ids a command prints are not host ids; and
a code or command that the sandbox kills is relayed as an exit status of 128 plus the signal.

Network: Landlock ABI 4 (TCP bind and connect) is present on this el9 5.14 kernel (the vendor
backports it; the ABI probe reports 6), so the loopback is denied by an empty rule set. It does
not cover UDP, so a seccomp filter also refuses `socket()` of the internet and packet families.
`unshare -rn` fails here with ENOSPC, so a network namespace is not an option unprivileged.

The command-line flag `--allow-unconfined-workspace` was declared and documented but never
copied into the configuration, so it did nothing. It is wired now, with
`--allow-workspace-network` and `--workspace-max-processes`.

How to avoid it: a confinement fixture must attack through a subprocess and read the file back
from the outside; a flag needs a test that sets it and sees the behaviour change.
