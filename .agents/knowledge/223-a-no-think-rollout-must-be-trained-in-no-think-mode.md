<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 223. A no-think rollout must be trained in no-think mode

**Lesson:** the rollouts ask a local brain server for no reasoning block (`RemoteModel::thinking` off), so Qwen3's
template ends every prompt the sampler saw with a closed, empty think block (`<think>\n\n</think>\n\n`, four
tokens). brain's trainer renders the prompt it scores an answer in according to the run's `thinking` setting, which
defaults to on and leaves the block out. Every record's prompt ids then disagree with the trainer's context, and
brain refused the first real-model run's records by name ("the sampler was given 222 ids before supervised answer 0
and the context the chat template renders encodes to 218"). The toy of `make test/online` never showed it.
`train.online` now gives the learner `thinking: false`, the mode its rollouts are drawn in, and refuses a recipe's
`params.thinking`. Spec: `crates/factory/src/run/online.rs` `the_learner_trains_the_mode_the_rollouts_are_drawn_in`.

## Evidence (2026-11-25, lane agrpo-real)

Run smoke0 on a GH200 (Qwen3-0.6B, brain 760c04d3b) failed with `HTTP 400: ... messages[2].prompt_token_ids ...
first differ at position 218, where the sampler was given 151667` (`<think>`). With `thinking: false` the same run
trained, published two versions and committed its groups.
