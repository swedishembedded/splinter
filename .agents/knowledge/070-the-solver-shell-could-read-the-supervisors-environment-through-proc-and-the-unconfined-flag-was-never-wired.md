<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 070. The solver's shell could read the supervisor's environment through /proc, and `--allow-unconfined-workspace` was never wired

Found 2026-10-18 writing the adversarial containment fixtures (`crates/sandbox/tests/containment.rs`).

Landlock confinement of a workspace command allowed `/proc` read only, and a process can read
`/proc/<pid>/environ` of any process of its own user: a confined command could read the keys in
the supervisor's environment (and in any other process of the user). Denying the numeric
`/proc` directories is not possible: coreutils on this host (a Rust multicall build) reads
`/proc/self/auxv`, and `self` is a different directory in every process, so a rule naming one
pid covers the command but not what it starts. The fix is on the other side: the supervisor sets
`PR_SET_DUMPABLE` 0 (`protect_supervisor_environment`), which makes its `/proc/<pid>` files
root-owned, and the command's environment is built from an allowlist that drops names that look
like secrets. Other processes of the user that hold secrets stay readable to a solver: keep keys
out of the environment of any process that shares a user with a solve, or run the solve under
another user or in a container.

Network: Landlock ABI 4 (TCP bind and connect) is present on this el9 5.14 kernel (the vendor
backports it; the ABI probe reports 6), so the loopback is denied by an empty rule set. It does
not cover UDP, so a seccomp filter also refuses `socket()` of the internet and packet families.
`unshare -rn` fails here with ENOSPC, so a network namespace is not an option unprivileged.

The command-line flag `--allow-unconfined-workspace` was declared and documented but never
copied into the configuration, so it did nothing. It is wired now, with
`--allow-workspace-network` and `--workspace-max-processes`.

How to avoid it: a confinement fixture must attack through a subprocess and read the file back
from the outside; a flag needs a test that sets it and sees the behaviour change.
