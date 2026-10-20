<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 083. A killed repo-replay run left its attempt trees behind and filled the inode quota

Every attempt exports the parent tree of its commit (3.5 to 5.5 thousand files for a real
repository) into `state/scratch/attempt-*`. The tree was removed when the attempt function
returned, but a run killed by a lease end or a restart never returned: fifteen stale trees
(about 45 thousand files) piled up on a drive with a 500 000-file quota.

A directory name cannot tell a dead attempt from a live one of another process, so each attempt
now holds an advisory file lock on `.live` inside its directory (`scratch::Claim`); the kernel
releases it when the holder dies. A run sweeps at its start every `attempt-*` and `target-*`
directory whose lock it can take. Directories without a marker (left by a binary from before this
change) are removed when older than ten minutes. The target tree of the gate comparison lives in
a subdirectory of its claim because the claim's directory also holds the marker, and the marker
would otherwise be compared as an extra file (the audit failed every attempt on it).
Specs: `samples/repo-replay/tests/scratch_spec.rs`.
