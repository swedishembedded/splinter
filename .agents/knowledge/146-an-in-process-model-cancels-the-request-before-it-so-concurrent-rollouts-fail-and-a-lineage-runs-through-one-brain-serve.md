<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 146. An in-process model cancels the request before it, so concurrent rollouts fail and a lineage runs through one brain serve

**Lesson:** The in-process local model (`LocalQwen::complete`) registers each generation as the one in flight and cancels
the previous one, so two requests at once end the first with "generation ended without a complete reply: cancelled".
Rollouts, proposals and the dev exam of a campaign are many independent requests; they go through one `brain serve`
whose continuous batching decodes them together, and training goes through the same service as a job.

## Measured (Qwen3-4B, one GH200, 2026-11-13)

- In process, `campaign-lua --probe` with `--concurrency` 4, 16 and 32 failed with the cancellation above at 24 and at 256
  tokens; concurrency 1 worked. A proposer pilot (12 requests of up to 1024 tokens, one stream at a time) had not finished
  one round of requests in 30 minutes.
- One `brain serve` per card (`--max-batch 32`, `--kv-cache f32`, `--serve-adapters DIR`, `--training --artifact-store
  DIR`): a calibration of 280 tasks x 8 attempts (2,240 completions of at most 256 tokens) took about 260 s with four
  cards running; a full arm (calibration, 576 rollouts, 24 training steps as a brain job, a dev exam of 400 and a
  retention exam of 300 tasks, greedy, for the candidate and the champion) took 445 to 475 s, 0.12 GPU-hours.
- Proposals: the policy's typed requests over the three manual pages ran one wave of eight `generate` calls
  (12 requests each) in about seven minutes and kept 156 to 165 programs after admission and deduplication; a second wave
  kept 74 more of 226 admitted (the manual is small: the proposals saturate near 240 distinct programs).

## Defects found on the way, each fixed with a regression test

- `brain_server::start` took a ready marker and a key left in its work directory by an earlier server for the new one's,
  so a restarted lineage found a dead server (`a_stale_ready_marker_does_not_make_a_new_server_ready`).
- `Rsi2Family::prompt` locked the proposals twice (the first guard lived through the match arm) and hung the second
  collection request of every run (`family::tests`, a generated task's prompt).
- The disk preflight threshold for free files is now a flag of the recipe; the project disk's inode quota was at 98-99%
  during the pilots because of other work, and a run that needs 20,000 free files refuses at 8,000.
