<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# brain-session - does a served agent session train as it was sampled?

`make test/contract/brain/session` runs it. The program fails and names the step that broke. It never skips.
It refuses to start without a brain binary (`BRAIN_BIN`), without the checkpoint (`BRAIN_SESSION_MODEL`) or on a
taken port.

## The question

A policy-gradient step trains an agent's answers with the ids the policy sampled and their log-probabilities. An
agent session makes several model calls, and each later call's prompt repeats the earlier answers. If the
runtime sends those answers back as they were sampled, the session is one append-only sequence. It then trains
as one record and forwards the shared context once. This sample checks that property on the real stack: sven as
the agent, brain as the server and trainer, Splinter as the producer of the records.

## What it does

1. Starts `brain serve` on the CPU with a small Qwen3 checkpoint, f32 KV, training on, on a port in 8910-8919.
2. Runs one online group of rollouts (three by default) of a one-file task through sven, exactly as
   `train.online` does, with sessions merged. The task is to write a greeting with the shell tool and say so.
3. Checks that brain reported every call's history append-only, meaning the answers sent back were spliced as
   sampled.
4. Checks that Splinter merged at least two rollouts into one record each. Each record must hold every answer of
   its session. A rollout the harness cut off mid-call is left out and the program says so.
5. Checks that brain's `check_dataset` accepts the merged group as packed rollout records.
6. Checks that brain's `score` of each answer's sampled ids, teacher-forced at the sampling temperature, gives the
   answer's behaviour log-probabilities within 1e-3.

## Inputs

| Flag | Make variable | What |
|---|---|---|
| `--brain` | `BRAIN_BIN` | the `brain` binary |
| `--model` | `BRAIN_SESSION_MODEL` | a small Qwen3 checkpoint in brain's store layout |
| `--port` | `BRAIN_SESSION_PORT` | 8910-8919 |
| `--work` | `BRAIN_SESSION_WORK` | the work directory, emptied first |
