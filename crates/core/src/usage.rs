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
    /// Tokens the policy generated.
    pub completion: u64,
    /// Prompt tokens sent, cached ones included.
    pub prompt: u64,
    /// Of the prompt tokens, those the server read from its prefix cache;
    /// zero when the record states none.
    pub cached: u64,
}

impl TokenUsage {
    /// The usage a score record's `usage` object states (`completion_tokens`,
    /// `prompt_tokens`, `cached_tokens`); `None` when it states neither
    /// completion nor prompt tokens, or more cached prompt tokens than prompt
    /// tokens (a record that contradicts itself is not a measurement).
    #[must_use]
    pub fn from_score_usage(usage: &Value) -> Option<Self> {
        let count = |field: &str| usage.get(field).and_then(Value::as_u64);
        let (completion, prompt) = match (count("completion_tokens"), count("prompt_tokens")) {
            (None, None) => return None,
            (completion, prompt) => (completion.unwrap_or(0), prompt.unwrap_or(0)),
        };
        let cached = count("cached_tokens").unwrap_or(0);
        (cached <= prompt).then_some(Self {
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
                completion: completion.unwrap_or(0),
                prompt: prompt.unwrap_or(0),
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

    /// Prompt and completion tokens together, cached or not.
    #[must_use]
    pub fn processed(&self) -> u64 {
        self.prompt + self.completion
    }

    /// What the attempt cost: its completion tokens plus its uncached prompt
    /// tokens.
    #[must_use]
    pub fn cost(&self) -> u64 {
        self.completion + (self.prompt - self.cached)
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
        assert_eq!(usage.cost(), 22803 + 634722 - 602128);
        assert_eq!(usage.processed(), 22803 + 634722);
    }

    #[test]
    fn a_record_without_counts_or_contradicting_itself_is_no_usage() {
        assert_eq!(TokenUsage::from_score_usage(&json!({"turns": 3})), None);
        assert_eq!(
            TokenUsage::from_score_usage(&json!({"prompt_tokens": 5, "cached_tokens": 6})),
            None
        );
        let uncached = TokenUsage::from_score_usage(&json!({"completion_tokens": 7}))
            .expect("completion alone is a usage");
        assert_eq!(uncached.cost(), 7);
        // A manifest's counts state no cache, so all of the prompt is a cost.
        let manifest = json!({"tokens": {"prompt": 10, "completion": 5}});
        assert_eq!(TokenUsage::of_trial(&manifest).map(|u| u.cost()), Some(15));
    }
}
