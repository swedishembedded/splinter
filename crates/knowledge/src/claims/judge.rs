// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The decisions a gate leaves to a judge when one is configured; see
//! [`super::pairing`] for what is decided without one.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_core::claim::Claim;

/// What a later claim does to an earlier claim about the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairVerdict {
    /// The later contradicts the earlier: the earlier is replaced.
    Supersede,
    /// The later says the same fact again: both are recorded, the earlier
    /// stays the live claim.
    Reinforce,
    /// They are two facts: both live.
    Separate,
}

/// Why a judge gave no decision.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct JudgeError(pub String);

/// A model asked what a gate cannot decide by code.
pub trait ClaimJudge: Send + Sync {
    /// What `later` does to `earlier`, two live claims that name the same
    /// thing.
    fn pair(&self, earlier: &Claim, later: &Claim) -> Result<PairVerdict, JudgeError>;

    /// Whether the person's `quotes`, taken as they are, assert
    /// `statement`. The judge is shown nothing else: not the question, the
    /// session or what the agent said.
    fn entails(&self, quotes: &[&str], statement: &str) -> Result<bool, JudgeError>;
}

/// The name of the typed call that decides a pair.
pub const PAIR_METHOD: &str = "judge_claim_pair";

/// The role the pair judge is given.
pub const PAIR_ROLE: &str = "You compare two statements a person made on different days about \
what may be the same thing, and say what the later one does to the earlier one. You are exact: \
you go by what the statements say, never by what you believe is true.";

/// The pair judge's brief.
pub const PAIR_BRIEF: &str = "You are given an `earlier` and a `later` claim, each a statement \
and the question it answers. Decide with `verdict`: `supersede` when the later statement \
contradicts the earlier one about the same thing, so only the later is now true; `reinforce` \
when the later says the same fact as the earlier, in whatever words; `separate` when they are \
about different things or different aspects of one thing, so both can be true. Give a short \
`reason`.";

/// One claim as the pair judge is shown it.
#[derive(Debug, Serialize)]
pub struct ClaimView<'a> {
    statement: &'a str,
    question: &'a str,
}

/// The two claims a pair judge is shown.
#[derive(Debug, Serialize)]
pub struct PairInput<'a> {
    earlier: ClaimView<'a>,
    later: ClaimView<'a>,
}

impl<'a> PairInput<'a> {
    /// `earlier` and `later` as the judge is shown them: their words, never
    /// the session they came from.
    #[must_use]
    pub fn of(earlier: &'a Claim, later: &'a Claim) -> Self {
        let view = |c: &'a Claim| ClaimView {
            statement: &c.statement,
            question: &c.question,
        };
        Self {
            earlier: view(earlier),
            later: view(later),
        }
    }
}

/// What the pair judge decided.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum WireVerdict {
    Supersede,
    Reinforce,
    Separate,
}

/// The pair judge's reply.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PairReply {
    verdict: WireVerdict,
    /// Why, in a sentence.
    reason: String,
}

impl PairReply {
    /// The decision the reply holds.
    #[must_use]
    pub fn verdict(&self) -> PairVerdict {
        match self.verdict {
            WireVerdict::Supersede => PairVerdict::Supersede,
            WireVerdict::Reinforce => PairVerdict::Reinforce,
            WireVerdict::Separate => PairVerdict::Separate,
        }
    }

    /// Why, in the judge's words.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// The name of the typed call that decides entailment.
pub const ENTAIL_METHOD: &str = "judge_claim_entailment";

/// The role the entailment judge is given.
pub const ENTAIL_ROLE: &str = "You read words a person wrote and a statement, and say whether \
the person's words assert that statement. You are strict: you go by what the words say, not by \
what you believe is true or by what the person might have meant.";

/// The entailment judge's brief.
pub const ENTAIL_BRIEF: &str = "You are given `quotes`, words a person wrote, and a `statement`. \
Set `entailed` to true only when the person, in those words, asserts the statement as true: not \
when they deny it, ask about it, suppose it, doubt it, joke about it, or attribute it to \
someone else, and not when the statement adds something they did not say. Give a short `reason`.";

/// What the entailment judge is shown.
#[derive(Debug, Serialize)]
pub struct EntailInput<'a> {
    quotes: &'a [&'a str],
    statement: &'a str,
}

impl<'a> EntailInput<'a> {
    /// The person's `quotes` and the `statement` made from them: nothing else.
    #[must_use]
    pub fn of(quotes: &'a [&'a str], statement: &'a str) -> Self {
        Self { quotes, statement }
    }
}

/// The entailment judge's reply.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EntailReply {
    entailed: bool,
    /// Why, in a sentence.
    reason: String,
}

impl EntailReply {
    /// Whether the words assert the statement.
    #[must_use]
    pub fn entailed(&self) -> bool {
        self.entailed
    }

    /// Why, in the judge's words.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}
