<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Set up a run workspace anywhere

A training and evaluation run needs frozen binaries (brain, sven, splinter), the Harbor worker's Python
environment, the task repository with its collections, the records to train on, a sandbox image cache and room for
run directories. `scripts/workspace-init.sh` creates all of it in one directory, on any machine that has the
repositories, a Rust toolchain, `uv` and the network:

```bash
scripts/workspace-init.sh --dir /scratch/local/work/ws --build-env ~/project/shared/containers/build-env.sh
/scratch/local/work/ws/run.sh recipes/vim-sft-qwen3-4b.yaml gpu1
```

`--dir` should be a fast local disk. Everything under it is work in progress and may be lost with the node:
source snapshots, build targets, the worker environment, the image cache (`XDG_CACHE_HOME` in its `env.sh`), run
directories (`runs/NAME`), state roots (`state/NAME`) and each run's brain files (`serve/NAME`). Copy only finished
products (an adapter, a results file) to durable storage.

`--cache` (default `$WORKSPACE_CACHE`, else `~/.cache/splinter-workspace`) is the durable part. It holds one frozen
binary set per repository and commit, built the first time and copied from then on, and the downloaded records.
Point it at shared storage and a second node starts in seconds instead of rebuilding.

- A repository is frozen at its checkout's HEAD, or at `--brain-rev`, `--sven-rev`, `--splinter-rev` or
  `--bench-rev`. A commit that is on no remote branch is refused: anyone must be able to rebuild a frozen binary.
- The records default to `swedishembedded/vim-headless-mastery-sft` (`--records`); its `train.jsonl` becomes the
  recipe's `train.jsonl` and its `eval.jsonl` its `validation.jsonl`.
- `run.sh RECIPE DEVICE [args]` starts a brain server for the run on DEVICE (training on, adapters served hot), runs
  the recipe against it, and stops the server when splinter ends. Several recipes run side by side on other cards;
  running the same command again continues the run. `run.sh --fork PARENT RECIPE DEVICE` forks a run of the workspace and serves on the parent's port: a stage that reaches a model carries the connection's address in its identity, so a fork on another port would recompute the stages it means to reuse.
- `workspace.json` records the commits and the records the workspace was built from.

`scripts/run-metrics.py RUN_DIR` reads what a run's evaluation arms did: counted and never-counted slots, passes
(over all slots, a slot that never counted as not passed, and over counted slots), retried attempts and infrastructure
timeouts, and from the trajectories how often a call that followed a Vim error repeated it, how many trials a loop
watchdog ended, refused model requests, and replies that carry reasoning.
