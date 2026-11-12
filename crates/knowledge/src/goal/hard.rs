// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! What makes a synthetic corpus hard for lexical selection.
//!
//! * **Paraphrase** - some current documents say what the goal's probes ask
//!   in other words (`maintenance period` for `service interval`, `reluctant`
//!   for `hesitant`, `payment is overdue` for `the rent is late`), so they
//!   share almost none of the probes' words. [`LEXICON`] pairs each variant
//!   with the word it stands for; only a reader of meaning, or a model
//!   proposer that normalises keys, knows they are the same.
//! * **Mirrors** - near-copies of a current document with no header and one
//!   wrong value in every section: as relevant as the original to any words.
//! * **Partial staleness** - the current manual covers only part of a topic;
//!   the older manual is the only source for the rest, and right about it.
//! * **Decoys** - prose that names the goal's topics and uses its
//!   vocabulary but states nothing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Which difficulties a corpus has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardness {
    /// Half of the current documents are paraphrased.
    pub paraphrase: bool,
    /// Header-less near-copies of current documents with a wrong value per section.
    pub mirrors: bool,
    /// The current manual leaves out half the units; the old one is right about them.
    pub partial_stale: bool,
    /// Factless prose that uses the goal's names and words.
    pub decoys: bool,
}

impl Hardness {
    /// Every difficulty on.
    pub const ALL: Self = Self {
        paraphrase: true,
        mirrors: true,
        partial_stale: true,
        decoys: true,
    };
}

/// (variant, the word it stands for), lower case. Longer phrases first.
pub const LEXICON: &[(&str, &str)] = &[
    ("maintenance period", "service interval"),
    ("nominal turning force", "rated torque"),
    ("peak pressure rating", "maximum pressure"),
    ("guarantee duration", "warranty period"),
    ("payment is overdue", "the rent is late"),
    ("the tenancy is extended", "the lease is renewed"),
    ("a fix is postponed", "a repair is delayed"),
    ("a neighbour grumbles", "a neighbour complains"),
    ("the security money is raised", "the deposit is discussed"),
    ("a site visit is announced", "an inspection is announced"),
    ("the radiators stop working", "the heating fails"),
    ("a higher rent is suggested", "a rent increase is proposed"),
    ("a key goes missing", "a key is lost"),
    (
        "a term of the contract is vague",
        "a contract clause is unclear",
    ),
    ("an overdue notice arrives", "a payment reminder arrives"),
    ("the departure date is fixed", "the move-out date is set"),
    ("the lift stops working", "the elevator breaks down"),
    ("a pet is mentioned", "a pet is reported"),
    ("the washing room is shut", "the laundry room is closed"),
    ("a pipe leaks", "a water leak is found"),
    ("a visitor overstays", "a guest stays too long"),
    ("the waste rules are altered", "the garbage rules change"),
    ("a parking spot is moved", "a parking space is reassigned"),
    ("a complaint about noise is sent", "a noise warning is sent"),
    ("proof of cover is requested", "the insurance is requested"),
    ("a showing is booked", "a viewing is scheduled"),
    ("a sublet is denied", "a subletting request is refused"),
    ("the bills are shared", "the utilities are split"),
    ("reluctant", "hesitant"),
    ("confrontational", "combative"),
    ("absent-minded", "forgetful"),
    ("thorough", "meticulous"),
    ("elusive", "evasive"),
    ("chatty", "talkative"),
    ("nervous", "anxious"),
    ("disdainful", "dismissive"),
];

/// `text` with every canonical phrase replaced by its variant.
#[must_use]
pub fn paraphrase(text: &str) -> String {
    let mut out = text.to_string();
    for (variant, canonical) in LEXICON {
        out = out.replace(canonical, variant);
    }
    out
}

/// `text` (lower case words) with every variant replaced by what it stands for.
#[must_use]
pub fn canonical(text: &str, lexicon: &BTreeMap<String, String>) -> String {
    let mut out = text.to_lowercase();
    let mut pairs: Vec<(&String, &String)> = lexicon.iter().collect();
    pairs.sort_by_key(|(variant, _)| std::cmp::Reverse(variant.len()));
    for (variant, canonical) in pairs {
        out = out.replace(variant.as_str(), canonical);
    }
    out
}

/// The lexicon as a map.
#[must_use]
pub fn lexicon() -> BTreeMap<String, String> {
    LEXICON
        .iter()
        .map(|(v, c)| ((*v).to_string(), (*c).to_string()))
        .collect()
}

/// A section of prose that names `topic` and `attributes` and states nothing.
#[must_use]
pub fn decoy_section(topic: &str, attributes: &[&str], n: usize) -> String {
    let attr = attributes[n % attributes.len().max(1)];
    format!(
        "## {topic} newsletter {n}\n\nThe {topic} range won a design award this year, and visitors to the {topic} stand \
         asked about the {attr}, which the team promised to look into. Several {topic} owners wrote in; opinions \
         about the {attr} differ and nobody could say more at the summit.\n"
    )
}
