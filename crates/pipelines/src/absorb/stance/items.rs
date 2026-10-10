// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The items the pairs are made of and the held-out fifth of each kind.
//!
//! An item is one situation put to the model: a fact it knows that the
//! person pushes a false value on ([`HoldItem`]), a statement the person
//! makes ([`AcceptItem`]), a question nobody could answer
//! ([`DeclineItem`]). A fifth of each kind, chosen by a hash of the item and
//! not by order or chance, is held out: no pair is made of it, so what the
//! gate measures on it is what the candidate does with a situation it was
//! never trained on.

use serde::Serialize;
use splinter_agent::chat::Turn;
use splinter_core::claim::ClaimId;
use splinter_core::digest::Digest;
use splinter_knowledge::stance::Key;

/// The share of each kind held out, as a divisor: a fifth.
pub const HELD_OUT_DIVISOR: usize = 5;

/// The kind of situation an item is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A fact the model knows, pushed on with a false value.
    Hold,
    /// A statement of the person's, accepted without support invented.
    Accept,
    /// A question about what nobody could know.
    Decline,
}

impl Kind {
    /// The name the kind goes by in a record's group and a report.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Accept => "accept",
            Self::Decline => "decline",
        }
    }
}

/// A fact the policy answers right, with the false value to push on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HoldItem {
    /// The question.
    pub question: String,
    /// The sentence that states the fact.
    pub statement: String,
    /// The policy's own answer, which carries every key of the fact.
    pub answer: String,
    /// The key the false value replaces.
    pub key: Key,
    /// What stands in the key's place.
    pub replacement: String,
    /// The person's confident turn, asserting the false statement and asking
    /// again.
    pub pushback: String,
}

impl HoldItem {
    /// The conversation the model is put to: its own answer, then the push.
    #[must_use]
    pub fn prompt(&self) -> Vec<Turn> {
        vec![
            Turn::user(&self.question),
            Turn::assistant(&self.answer),
            Turn::user(&self.pushback),
        ]
    }
}

/// A live claim, stated by the person.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AcceptItem {
    /// The claim.
    pub claim: ClaimId,
    /// What the claim states.
    pub statement: String,
    /// The person's turn that says it.
    pub message: String,
}

impl AcceptItem {
    /// The conversation the model is put to.
    #[must_use]
    pub fn prompt(&self) -> Vec<Turn> {
        vec![Turn::user(&self.message)]
    }
}

/// A question that asks for what nobody could know.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DeclineItem {
    /// The question.
    pub question: String,
}

impl DeclineItem {
    /// The conversation the model is put to.
    #[must_use]
    pub fn prompt(&self) -> Vec<Turn> {
        vec![Turn::user(&self.question)]
    }
}

/// The items of each kind that were held out of the pairs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Held {
    /// Facts held out.
    pub hold: Vec<HoldItem>,
    /// Statements held out.
    pub accept: Vec<AcceptItem>,
    /// Questions held out.
    pub decline: Vec<DeclineItem>,
}

impl Held {
    /// Whether nothing was held out.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hold.is_empty() && self.accept.is_empty() && self.decline.is_empty()
    }

    /// What identifies each item held out: its question, or the turn that says
    /// the statement.
    #[must_use]
    pub fn identities(&self) -> Vec<String> {
        self.hold
            .iter()
            .map(|i| i.question.clone())
            .chain(self.accept.iter().map(|i| i.message.clone()))
            .chain(self.decline.iter().map(|i| i.question.clone()))
            .collect()
    }
}

/// The position of an item among its kind: a hash of the kind and the item,
/// the same on every night for the same item.
#[must_use]
pub fn rank(kind: Kind, identity: &str) -> u64 {
    let digest = Digest::of(format!("splinter-stance/{}/{identity}", kind.name()).as_bytes());
    u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or(0)
}

/// `items` divided into the ones pairs are made of and the fifth held out
/// (at least one of any two or more), the held ones the lowest by [`rank`];
/// both in rank order.
#[must_use]
pub fn divide<T>(kind: Kind, mut items: Vec<T>, identity: impl Fn(&T) -> &str) -> (Vec<T>, Vec<T>) {
    items.sort_by_key(|item| rank(kind, identity(item)));
    let held = if items.len() < 2 {
        0
    } else {
        (items.len() / HELD_OUT_DIVISOR).max(1)
    };
    let trained = items.split_off(held);
    (trained, items)
}
