// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checks that an agent did what it was taught,
// for its clients. If your team needs expertise in testing whether an agent
// has learned from its users, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Whether a reply shows that something was learned.
//!
//! A [`Check`] is written with the scenario, before any reply exists, and says
//! what a reply must and must not contain. It is a plain text check on purpose:
//! the answer to "when is the deadline" either names the day it was taught or
//! it does not, and no model's opinion is needed to say so.

use serde::{Deserialize, Serialize};

/// What a reply must satisfy. Every condition that is set must hold.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    /// The reply contains at least one of these (ignoring case); empty means no requirement.
    #[serde(default)]
    pub any_of: Vec<String>,
    /// The reply contains none of these (ignoring case).
    #[serde(default)]
    pub none_of: Vec<String>,
    /// The reply has at most this many words.
    #[serde(default)]
    pub max_words: Option<usize>,
    /// The reply ends with this, ignoring case and trailing punctuation and space.
    #[serde(default)]
    pub ends_with: Option<String>,
}

impl Check {
    /// Whether `reply` satisfies every condition that is set.
    #[must_use]
    pub fn passes(&self, reply: &str) -> bool {
        let lower = reply.to_lowercase();
        let has = |needle: &String| lower.contains(&needle.to_lowercase());
        let ends = |tail: &String| {
            lower
                .trim_end_matches(|c: char| !c.is_alphanumeric())
                .ends_with(&tail.to_lowercase())
        };
        (self.any_of.is_empty() || self.any_of.iter().any(has))
            && !self.none_of.iter().any(has)
            && self
                .max_words
                .is_none_or(|max| reply.split_whitespace().count() <= max)
            && self.ends_with.as_ref().is_none_or(ends)
    }
}

#[cfg(test)]
mod specs {
    use super::*;

    fn check(any_of: &[&str], none_of: &[&str]) -> Check {
        Check {
            any_of: any_of.iter().map(ToString::to_string).collect(),
            none_of: none_of.iter().map(ToString::to_string).collect(),
            ..Check::default()
        }
    }

    #[test]
    fn a_reply_must_name_what_was_taught_and_not_what_it_replaced() {
        let monday = check(&["Monday"], &["Friday"]);
        assert!(monday.passes("The deadline is monday."));
        assert!(!monday.passes("The deadline is Friday."));
        assert!(
            !monday.passes("The deadline was moved from Friday to Monday."),
            "naming the old one fails"
        );
        assert!(!monday.passes("I do not know."));
    }

    #[test]
    fn a_check_with_nothing_set_passes_and_each_set_condition_must_hold() {
        assert!(Check::default().passes("anything"));
        let rule = Check {
            max_words: Some(5),
            ends_with: Some("over".into()),
            ..Check::default()
        };
        assert!(rule.passes("Roger that, over."));
        assert!(!rule.passes("Roger that, over and out."));
        assert!(!rule.passes("This reply is far too long, over."));
    }
}
