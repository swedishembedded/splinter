// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gates an article passes before it joins a corpus. They are code: a
//! writer that tells the right story badly, or quotes what it was told to keep
//! private, is turned away without asking a model.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::episode::Episode;
use crate::persona::{Belief, Persona, Polarity, Tenet};
use crate::scenario::Archetype;

/// How many consecutive words of a belief an article may not repeat.
pub const QUOTE_RUN_WORDS: usize = 6;

/// A written article.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Article {
    /// A title.
    pub title: String,
    /// The text.
    pub body: String,
}

/// Why an article was turned away.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Rejection {
    /// Fewer words than the floor.
    TooShort {
        /// Words it has.
        words: usize,
        /// Words it needs.
        min: usize,
    },
    /// It repeats a run of a belief's words.
    QuotesBelief {
        /// The start of the belief it repeats.
        statement: String,
    },
    /// An episode's case is not mentioned.
    MissingCase {
        /// The case it leaves out.
        case: String,
    },
    /// An episode is told with a decision other than the one the beliefs determine.
    WrongDecision {
        /// The case.
        case: String,
        /// What the beliefs determine.
        expected: Archetype,
    },
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { words, min } => write!(f, "{words} words, at least {min} needed"),
            Self::QuotesBelief { statement } => write!(f, "repeats a belief: {statement}"),
            Self::MissingCase { case } => write!(f, "does not tell {case}"),
            Self::WrongDecision { case, expected } => {
                write!(f, "does not show the decision {expected:?} for {case}")
            }
        }
    }
}

/// Words that show a way of acting in the paragraphs that tell a case.
fn markers(option: Archetype) -> &'static [&'static str] {
    match option {
        Archetype::ActAlone => &[
            "myself",
            "on my own",
            "my own authority",
            "at once",
            "without waiting",
            "alone",
        ],
        Archetype::ConsultThenAct => &[
            "assembly",
            "council",
            "put it to",
            "put the matter",
            "consult",
            "their consent",
        ],
        Archetype::DeferToOutsider => &[
            "neighbour",
            "their offer",
            "accepted their",
            "outside",
            "their help",
            "the offer",
        ],
        Archetype::HoldAndWait => &[
            "wait",
            "left it",
            "did nothing",
            "held back",
            "postponed",
            "let it stand",
        ],
    }
}

const ALL: [Archetype; 4] = [
    Archetype::ActAlone,
    Archetype::ConsultThenAct,
    Archetype::DeferToOutsider,
    Archetype::HoldAndWait,
];

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Every statement a belief can be stated with, held or not.
fn every_statement() -> Vec<&'static str> {
    const TENETS: [Tenet; 6] = [
        Tenet::WithinGrant,
        Tenet::Consent,
        Tenet::SelfReliance,
        Tenet::Caution,
        Tenet::Burden,
        Tenet::Haste,
    ];
    TENETS
        .iter()
        .flat_map(|&tenet| {
            [Polarity::Strict, Polarity::Lax].map(|polarity| Belief { tenet, polarity }.statement())
        })
        .collect()
}

/// Whether `article` may join the corpus of `persona`, told from `episodes`.
///
/// # Errors
/// The first reason it may not.
pub fn check(
    article: &Article,
    persona: &Persona,
    episodes: &[Episode],
    min_words: usize,
) -> Result<(), Rejection> {
    let count = article.body.split_whitespace().count();
    if count < min_words {
        return Err(Rejection::TooShort {
            words: count,
            min: min_words,
        });
    }
    let body_words = words(&article.body);
    let runs: BTreeSet<&[String]> = body_words.windows(QUOTE_RUN_WORDS).collect();
    for statement in every_statement() {
        let s = words(statement);
        if s.windows(QUOTE_RUN_WORDS).any(|w| runs.contains(w)) {
            return Err(Rejection::QuotesBelief {
                statement: statement.chars().take(48).collect(),
            });
        }
    }
    let lower = article.body.to_lowercase();
    let paragraphs: Vec<&str> = lower.split("\n\n").collect();
    for episode in episodes {
        let name = episode.case.name.to_lowercase();
        let told: String = paragraphs
            .iter()
            .filter(|p| p.contains(&name))
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        if told.is_empty() {
            return Err(Rejection::MissingCase {
                case: episode.case.name.clone(),
            });
        }
        let score = |o: Archetype| {
            markers(o)
                .iter()
                .map(|m| told.matches(m).count())
                .sum::<usize>()
        };
        let expected = persona.decide(&episode.scenario).choice;
        let best = score(expected);
        if best == 0 || ALL.iter().any(|&o| o != expected && score(o) >= best) {
            return Err(Rejection::WrongDecision {
                case: episode.case.name.clone(),
                expected,
            });
        }
    }
    Ok(())
}
