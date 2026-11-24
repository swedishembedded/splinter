// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for accounting the compute an
// agent's attempts cost, for its clients. If your team needs expertise in
// the economics of reinforcement learning on agent rollouts, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What an attempt's model requests cost, in tokens.
//!
//! An examiner's score record states a trial's usage summed over its
//! requests: the prompt tokens (cached ones included), the cached share of
//! them and the completion tokens. A cached prompt token costs a prefix-cache
//! hit, not a forward pass, so the cost of an attempt is its completion
//! tokens plus its uncached prompt tokens.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The tokens an attempt's requests processed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens the policy generated; `None` when the record states none (an
    /// unmeasured count is never zero).
    pub completion: Option<u64>,
    /// Prompt tokens sent, cached ones included; `None` when the record
    /// states none.
    pub prompt: Option<u64>,
    /// Of the prompt tokens, those the server read from its prefix cache;
    /// zero when the record states none.
    pub cached: u64,
}

impl TokenUsage {
    /// The usage a score record's `usage` object states (`completion_tokens`,
    /// `prompt_tokens`, `cached_tokens`); `None` when it states neither
    /// completion nor prompt tokens, or more cached prompt tokens than prompt
    /// tokens (a record that contradicts itself is not a measurement). A
    /// count it does not state stays unmeasured.
    #[must_use]
    pub fn from_score_usage(usage: &Value) -> Option<Self> {
        let count = |field: &str| usage.get(field).and_then(Value::as_u64);
        let (completion, prompt) = (count("completion_tokens"), count("prompt_tokens"));
        if completion.is_none() && prompt.is_none() {
            return None;
        }
        let cached = count("cached_tokens").unwrap_or(0);
        prompt
            .is_none_or(|prompt| cached <= prompt)
            .then_some(Self {
                completion,
                prompt,
                cached,
            })
    }

    /// The usage a trial manifest's `tokens` object states (`prompt`,
    /// `completion`), which counts no cache: every prompt token is taken as
    /// uncached. `None` when it states neither.
    #[must_use]
    pub fn from_trial_tokens(tokens: &Value) -> Option<Self> {
        let count = |field: &str| tokens.get(field).and_then(Value::as_u64);
        match (count("completion"), count("prompt")) {
            (None, None) => None,
            (completion, prompt) => Some(Self {
                completion,
                prompt,
                cached: 0,
            }),
        }
    }

    /// The usage a trial's verdict evidence states: its score record's
    /// usage, else its manifest's token counts.
    #[must_use]
    pub fn of_trial(evidence: &Value) -> Option<Self> {
        evidence
            .pointer("/score/usage")
            .and_then(Self::from_score_usage)
            .or_else(|| evidence.get("tokens").and_then(Self::from_trial_tokens))
    }

    /// Prompt and completion tokens together, cached or not; `None` unless
    /// both were measured.
    #[must_use]
    pub fn processed(&self) -> Option<u64> {
        self.prompt?.checked_add(self.completion?)
    }

    /// What the attempt cost: its completion tokens plus its uncached prompt
    /// tokens; `None` unless both were measured (and the cached share is no
    /// more than the prompt, which a deserialised value need not honour).
    #[must_use]
    pub fn cost(&self) -> Option<u64> {
        self.completion?
            .checked_add(self.prompt?.checked_sub(self.cached)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_cost_is_the_completion_and_the_uncached_prompt() {
        let usage = TokenUsage::from_score_usage(&json!({
            "completion_tokens": 22803, "prompt_tokens": 634722, "cached_tokens": 602128,
        }))
        .expect("a usage");
        assert_eq!(usage.cost(), Some(22803 + 634722 - 602128));
        assert_eq!(usage.processed(), Some(22803 + 634722));
    }

    #[test]
    fn a_record_contradicting_itself_or_without_counts_is_no_usage() {
        assert_eq!(TokenUsage::from_score_usage(&json!({"turns": 3})), None);
        assert_eq!(
            TokenUsage::from_score_usage(&json!({"prompt_tokens": 5, "cached_tokens": 6})),
            None
        );
        // A manifest's counts state no cache, so all of the prompt is a cost.
        let manifest = json!({"tokens": {"prompt": 10, "completion": 5}});
        assert_eq!(
            TokenUsage::of_trial(&manifest).and_then(|u| u.cost()),
            Some(15)
        );
    }

    #[test]
    fn a_count_the_record_does_not_state_is_unmeasured_never_zero() {
        let completion_only = TokenUsage::from_score_usage(&json!({"completion_tokens": 7}))
            .expect("a count is a usage");
        assert_eq!(completion_only.completion, Some(7));
        assert_eq!(completion_only.cost(), None, "no prompt count, no cost");
        let prompt_only =
            TokenUsage::from_score_usage(&json!({"prompt_tokens": 9})).expect("a count is a usage");
        assert_eq!(prompt_only.completion, None, "not zero completion tokens");
        assert_eq!(prompt_only.cost(), None);
        let manifest = TokenUsage::from_trial_tokens(&json!({"prompt": 4})).expect("a usage");
        assert_eq!(manifest.completion, None);
    }

    #[test]
    fn a_usage_deserialised_with_more_cached_than_prompt_has_no_cost() {
        let usage: TokenUsage =
            serde_json::from_value(json!({"completion": 1, "prompt": 2, "cached": 5}))
                .expect("fields are plain numbers");
        assert_eq!(usage.cost(), None);
    }
}
