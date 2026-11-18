<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 176. A campaign's token budget never bound, because its rollouts counted no tokens

**Lesson:** a budget is only as real as the counter behind it. `splinter campaign`'s `[budget].tokens` was checked
before every rollout, but the one production rollout source, `PracticeRollouts`, returned `tokens: None` for every
stage, and the account added `None` as 0: a campaign could spend any number of tokens and the budget would never stop
it, while the budget test passed on a scripted rollout source that did count. Test a budget through the production
counter, and make "not counted" visible rather than a zero.

Found 2026-11-18 (lane C2). Bench's export already carries each trial's usage in its score record
(`bench.score/1` `usage`: `prompt_tokens`, `completion_tokens`, summed by bench from every gateway request of the
trial). The practice run now reads it per trial (`PracticeTrial::tokens`, prompt plus completion) and the campaign's
rollouts sum it; a rollout none of whose trials states usage records no tokens instead of 0.
`bench_practice::as_a_campaigns_rollouts_the_tokens_are_summed_from_each_trials_usage_so_a_token_budget_binds` runs
`PracticeRollouts` over the scripted examiner and checks 30 tokens for two graded trials of 15 and one trial that never
reached the model.
