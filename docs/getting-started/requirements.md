<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Requirements

| Need | For | Notes |
|---|---|---|
| Linux | everything | Repository tasks need a kernel with Landlock ABI 4 to confine what an agent runs; without it the command is refused unless you pass `--allow-unconfined-workspace`. |
| Rust toolchain (stable) | building | `make build`. |
| A GPU with room for the policy | real runs | One 24 GB GPU holds Qwen3-0.6B for serving and trains LoRA adapters on it at agent-trajectory lengths. Qwen3-1.7B also trains on one such GPU at shorter records. Larger bases need more memory than one card has (`--bf16-base` helps a 7B base). Without a GPU brain falls back to the CPU backend, which is correct and too slow for a real run. |
| The policy checkpoint | everything that loads a model | `Qwen/Qwen3-0.6B` in brain's model store by default (`brain pull Qwen/Qwen3-0.6B`). |
| `brain` on `PATH` | the release gate's serve check | Or name the binary with `SPLINTER_BRAIN_BIN`. |
| `python3` | task kinds that run code, and the repository families | |
| Disk | the state root | `splinter doctor` fails below 1 GiB free and warns below 10 GiB; it is an estimate for a small base, not a bound. |
| A network | nothing | Models are local. A `remote:` model needs `--allow-remote` (or `SPLINTER_ALLOW_REMOTE=1`) and a key; that is the only use of the network. |
| Docker, with a digest-pinned image | optional | Runs an agent's commands in a container with no network instead of confining them on the host. See [limitations](../concepts/limitations.md#security-of-agent-generated-code). |
| Node 20 or later | building this documentation only | `make docs`; see [contributing](../contributing/documentation.md). |

`splinter doctor` and `make test` load no model.
