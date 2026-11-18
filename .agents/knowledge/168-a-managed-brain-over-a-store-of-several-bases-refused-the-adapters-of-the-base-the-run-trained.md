<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 168. A managed brain over a store of several bases refused the adapters of the base the run trained

**Found 2026-11-18**, verifying the API-only training path against a real `brain serve --training` (brain 9d28ba56d,
Qwen/Qwen3-4B, one GPU).

`ManagedBrain` started `brain serve --models-dir <store> --serve-adapters DIR` with no `--checkpoint`. brain binds
every adapter of the directory to one served base; over a store holding Qwen/Qwen3.8-27B-FP8 beside Qwen/Qwen3-4B it
chose the Qwen3.8 and refused the Qwen3-4B adapter a job had just trained ("the served base is a Qwen3.8 that takes
LoRA adapters of family qwen35"). Every `locate` of a candidate then timed out after 120 s. Over a store of only the
Qwen3-4B, brain refused to start at all without `--checkpoint`.

The scripted doubles never bind an adapter to a base, so no spec saw it.

**Fix:** the server a run owns is started with `--checkpoint <policy base>` (`ServiceSettings::adapter_base`) when the
base exists, and keeps `--adapter-targets all` (an adapter that also corrects the MLP is refused under brain's default,
attention only). Verified: the trained adapter was then listed as `Qwen/Qwen3-4B:splinter:candidate:<attempt>` and
scored teacher-forced through `/v1/run` (held-out loss 6.91 for the base, 1.29 with the adapter, 69 tokens).
A run whose adapters sit on several bases at once still needs brain to bind an adapter to the base its card names.
