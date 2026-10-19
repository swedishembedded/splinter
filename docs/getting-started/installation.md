<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Installation

Splinter builds from source with Cargo. [sven](https://swedishembedded.com/docs/sven) and
[brain](https://swedishembedded.com/docs/brain) are git dependencies at the revisions pinned
in `Cargo.lock`; a plain build fetches them, and nothing else needs installing in advance. See
[requirements](requirements.md) for the machine you need.

## Build

```bash
make build                                  # release build of every package
export PATH="$PWD/target/release:$PATH"     # once per shell; the commands in this manual are bare
splinter doctor                             # can a run start here? state, disk, GPU, model weights
```

`make build` is already the release profile (brain's kernels are unusably slow unoptimised), so
there is no separate profile target. `splinter doctor` prints one `ok`, `warn` or `FAIL` line per check
with the fix for each failure, and exits 1 on a failure.

## Fetch the policy model

The policy model comes from brain's model store, which `brain pull Qwen/Qwen3-0.6B` fills. The
default store is `~/.local/share/brain/models`; `BRAIN_MODELS_DIR` names another and
`BRAIN_QWEN_WEIGHTS` names another checkpoint as the policy's base. See
[configuration](configuration.md).

## Build against local sven and brain checkouts

To change sven or brain and Splinter together, write the gitignored local override once:

```bash
make deps/local SVEN_DIR=../sven BRAIN_DIR=../brain
```

Builds then compile the checkouts' working trees, offline, without touching `Cargo.toml` or
`Cargo.lock`. `make deps/lock` re-pins the lock to those checkouts' HEADs; push those commits before
sharing the lock. Delete `.cargo/config.toml` to go back to the remotes.

## Next

Run [your first learning run](first-learning-run.md).
