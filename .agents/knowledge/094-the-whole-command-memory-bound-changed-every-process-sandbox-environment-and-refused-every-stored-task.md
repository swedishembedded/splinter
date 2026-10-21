<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 094. The whole-command memory bound changed every process-sandbox environment and refused every stored task

Commit 859ee64bc gave the process sandbox a bound on the memory of a whole command
(`Limits::tree_memory_bytes`, 24 GiB by default). The field is left out of an environment's snapshot only
when unset, but the process sandbox now always sets it, so its environments got another digest: a task
set stored by an earlier build (the qualification sets of the CE evaluation) was refused by `solve` with
"the task records environment workspace X, not the offered workspace Y" for every task. The change was
deliberate for tasks made from then on (the comment says so); that it stranded stored tasks was not
noticed, and the only way back was an old binary.

Fix: `--workspace-tree-memory-gib N` names the bound (24 when not given); `0` names none, which offers the
environment of the tasks stored before the bound existed. A run that solves such a set passes `0`. The
default is unchanged, so tasks made now still name the bounded environment.
