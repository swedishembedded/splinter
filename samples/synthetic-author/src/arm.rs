// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in telling which part of a method
// accounts for a model's result, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The arms of the examination: what a model is given besides the item, apart
//! from the item itself.

use serde::{Deserialize, Serialize};

use crate::persona::Persona;
use crate::prompts::SYSTEM;

/// What retrieval draws its passages from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Over {
    /// The paragraphs of the whole articles of the corpus.
    Articles,
    /// The training records themselves: each decision kept, with the situation
    /// as the reader briefed it and what was decided and why. The fair
    /// comparator for an adapter trained on them.
    Records,
}

/// How many passages retrieval hands over. These three are the settings the
/// ablation compares; others are not asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Passages {
    /// Two.
    Two,
    /// Six.
    Six,
    /// Twelve.
    Twelve,
}

impl Passages {
    /// The number of passages.
    #[must_use]
    pub const fn count(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Six => 6,
            Self::Twelve => 12,
        }
    }
}

/// One configuration of the retrieval arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retrieval {
    /// Where the passages come from.
    pub over: Over,
    /// How many are handed over.
    pub passages: Passages,
}

impl Retrieval {
    /// The configuration the arm was first asked with, and still is under the
    /// name `retrieval`: paragraphs of whole articles, six of them.
    pub const ORIGINAL: Self = Self {
        over: Over::Articles,
        passages: Passages::Six,
    };
}

/// The system turn each arm of the examination is asked under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// The plain system turn, no adapter.
    Base,
    /// The plain system turn, with the adapter.
    Tuned,
    /// The system turn and the author's four convictions, in order of priority.
    Principles,
    /// The principles arm, and what each conviction makes of each way to act.
    PrinciplesWithVerdicts,
    /// The system turn and the author's name.
    Named,
    /// The plain system turn and the passages of the corpus that share the
    /// most with the situation.
    Retrieval(Retrieval),
}

impl Arm {
    /// Every arm there is.
    pub const ALL: [Self; 11] = [
        Self::Base,
        Self::Tuned,
        Self::Principles,
        Self::PrinciplesWithVerdicts,
        Self::Named,
        Self::Retrieval(Retrieval::ORIGINAL),
        Self::Retrieval(Retrieval {
            over: Over::Articles,
            passages: Passages::Two,
        }),
        Self::Retrieval(Retrieval {
            over: Over::Articles,
            passages: Passages::Twelve,
        }),
        Self::Retrieval(Retrieval {
            over: Over::Records,
            passages: Passages::Two,
        }),
        Self::Retrieval(Retrieval {
            over: Over::Records,
            passages: Passages::Six,
        }),
        Self::Retrieval(Retrieval {
            over: Over::Records,
            passages: Passages::Twelve,
        }),
    ];

    /// The arm's name, as used in file names and the exam manifest.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Tuned => "tuned",
            Self::Principles => "principles",
            Self::PrinciplesWithVerdicts => "principles-verdicts",
            Self::Named => "named",
            Self::Retrieval(Retrieval { over, passages }) => match (over, passages) {
                (Over::Articles, Passages::Six) => "retrieval",
                (Over::Articles, Passages::Two) => "retrieval-2",
                (Over::Articles, Passages::Twelve) => "retrieval-12",
                (Over::Records, Passages::Two) => "retrieval-records-2",
                (Over::Records, Passages::Six) => "retrieval-records-6",
                (Over::Records, Passages::Twelve) => "retrieval-records-12",
            },
        }
    }

    /// The arm called `name`, if there is one.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|arm| arm.name() == name)
    }

    /// The system turn the arm is asked under, before any retrieved passages.
    #[must_use]
    pub fn system(self, persona: &Persona) -> String {
        match self {
            Self::Base | Self::Tuned | Self::Retrieval(_) => SYSTEM.to_string(),
            Self::Named => format!("{SYSTEM} You are {}.", persona.name),
            Self::Principles => principles(persona),
            Self::PrinciplesWithVerdicts => {
                let mut text = principles(persona);
                text.push_str(
                    "\n\nWhat each conviction makes of the ways to act, in the order you hold them:",
                );
                for belief in &persona.beliefs {
                    text.push_str(&format!("\n- {}", belief.statement()));
                    for rule in belief.rules() {
                        text.push_str(&format!("\n  * {}", rule.describe()));
                    }
                }
                text.push_str(
                    "\n\nYou apply them in that order. A conviction first rules out the ways it forbids (unless that would leave none), then keeps only the ways it favours most; the next conviction chooses among the ways that remain, and the first to leave a single way decides.",
                );
                text
            }
        }
    }

    /// What the arm retrieves, if it retrieves.
    #[must_use]
    pub fn retrieval(self) -> Option<Retrieval> {
        match self {
            Self::Retrieval(r) => Some(r),
            _ => None,
        }
    }
}

fn principles(persona: &Persona) -> String {
    let mut text = format!(
        "{SYSTEM}\n\nYou hold these convictions, listed from the most important to the least, and they decide how you settle a matter:"
    );
    for belief in &persona.beliefs {
        text.push_str(&format!("\n- {}", belief.statement()));
    }
    text
}
