// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gates an article passes before it joins a corpus. They are code: a
//! writer that tells the right story badly, or quotes what it was told to keep
//! private, is turned away without asking a model.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::episode::{brief_phrases, did_phrase, did_phrase_first_person, scene_text, Episode};
use crate::persona::{Belief, Persona, Polarity, Tenet};
use crate::scenario::{Archetype, Features, TOPICS};

/// How many consecutive words an article may repeat of what it is shown, before
/// it counts as quoting.
pub const QUOTE_RUN_WORDS: usize = 6;
/// How many consecutive words of a belief an article may not repeat: long enough
/// that only a maxim recited, and not a phrase of ordinary narrative, matches.
pub const BELIEF_RUN_WORDS: usize = 9;
/// How many consecutive words of the brief's scene-setting an article may not
/// repeat.
pub const SCENE_RUN_WORDS: usize = 8;

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
    /// It repeats a run of the words the writer was briefed with.
    CopiesBrief {
        /// The start of the run it repeats.
        run: String,
    },
    /// The essay leaves a fact of the case out or states it wrongly.
    MisstatedFact {
        /// The case.
        case: String,
        /// The fact.
        fact: &'static str,
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
    /// The reader's answer is in no shape that can be read, so the article has
    /// not been judged and is turned away for it.
    Unreadable {
        /// What was wrong with the answer.
        detail: String,
    },
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { words, min } => write!(f, "{words} words, at least {min} needed"),
            Self::QuotesBelief { statement } => write!(f, "repeats a belief: {statement}"),
            Self::CopiesBrief { run } => write!(f, "repeats the brief: {run}"),
            Self::MisstatedFact { case, fact } => {
                write!(
                    f,
                    "does not state, or states wrongly, whether {fact} for {case}"
                )
            }
            Self::MissingCase { case } => write!(f, "does not tell {case}"),
            Self::WrongDecision { case, expected } => {
                write!(f, "does not show the decision {expected:?} for {case}")
            }
            Self::Unreadable { detail } => {
                write!(f, "the reader's answer could not be read: {detail}")
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

/// The sentences of `text` that do not deny anything. An author names the ways
/// he did not act as readily as the way he did, and "I could not do it alone"
/// shows the opposite of acting alone.
fn affirmed_sentences(text: &str) -> String {
    const DENIALS: [&str; 8] = [
        " not ",
        "n't",
        " never ",
        " no ",
        " nor ",
        " neither ",
        "cannot",
        " nothing ",
    ];
    text.split(['.', '!', '?', ';'])
        .filter(|sentence| {
            let padded = format!(" {} ", sentence.replace([',', ':'], " "));
            !DENIALS.iter().any(|d| padded.contains(d))
        })
        .collect::<Vec<_>>()
        .join(". ")
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The first run of [`QUOTE_RUN_WORDS`] words of `text` that is also a run of
/// any phrase of the brief for what was done, in either voice.
#[must_use]
pub fn brief_phrase_copied(text: &str) -> Option<String> {
    let said = words(text);
    let runs: BTreeSet<&[String]> = said.windows(QUOTE_RUN_WORDS).collect();
    brief_phrases().iter().find_map(|phrase| {
        words(phrase)
            .windows(QUOTE_RUN_WORDS)
            .find(|w| runs.contains(*w))
            .map(|w| w.join(" "))
    })
}

/// Every statement and private note a belief can be told with, held or not.
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
            [Polarity::Strict, Polarity::Lax]
                .into_iter()
                .flat_map(move |polarity| {
                    let belief = Belief { tenet, polarity };
                    [belief.statement(), belief.note()]
                })
        })
        .collect()
}

/// The paragraphs of `article` that name `case`, as written.
#[must_use]
pub fn told_paragraphs(article: &Article, case: &str) -> String {
    let name = case.to_lowercase();
    article
        .body
        .split("\n\n")
        .filter(|p| p.to_lowercase().contains(&name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether `article` is long enough, keeps the author's convictions to itself
/// and tells every episode.
///
/// # Errors
/// The first reason it is not.
pub fn check_form(
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
    let belief_runs: BTreeSet<&[String]> = body_words.windows(BELIEF_RUN_WORDS).collect();
    for statement in every_statement() {
        let s = words(statement);
        if s.windows(BELIEF_RUN_WORDS).any(|w| belief_runs.contains(w)) {
            return Err(Rejection::QuotesBelief {
                statement: statement.chars().take(48).collect(),
            });
        }
    }
    for episode in episodes {
        if told_paragraphs(article, &episode.case.name).is_empty() {
            return Err(Rejection::MissingCase {
                case: episode.case.name.clone(),
            });
        }
        // The topic's name is the subject of the essay, not a copied sentence, and
        // the facts have to be stated, so only the scene and the decision's words
        // count. The house that offers help is named wherever it is mentioned.
        let scene = scene_text(episode.scenario.topic, &episode.case)
            .replace(TOPICS[episode.scenario.topic], "");
        let chose = persona.decide(&episode.scenario).choice;
        let did = did_phrase(chose, &episode.case).replace(&episode.case.house, "");
        let did_myself =
            did_phrase_first_person(chose, &episode.case).replace(&episode.case.house, "");
        for (brief, length) in [
            (&scene, SCENE_RUN_WORDS),
            (&did, QUOTE_RUN_WORDS),
            (&did_myself, QUOTE_RUN_WORDS),
        ] {
            let runs: BTreeSet<&[String]> = body_words.windows(length).collect();
            if let Some(run) = words(brief).windows(length).find(|w| runs.contains(*w)) {
                return Err(Rejection::CopiesBrief { run: run.join(" ") });
            }
        }
    }
    Ok(())
}

/// Whether the words that show a way of acting, in the paragraphs that tell
/// each episode, show the decision the beliefs determine. Cruder than a
/// reader, and free.
///
/// # Errors
/// The first episode told another way.
pub fn check_markers(
    article: &Article,
    persona: &Persona,
    episodes: &[Episode],
) -> Result<(), Rejection> {
    for episode in episodes {
        let told = told_paragraphs(article, &episode.case.name).to_lowercase();
        let affirmed = affirmed_sentences(&told);
        let score = |o: Archetype| {
            markers(o)
                .iter()
                .map(|m| affirmed.matches(m).count())
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

/// Whether `article` may join the corpus of `persona`, told from `episodes`,
/// judging the decisions by their marker words.
///
/// # Errors
/// The first reason it may not.
pub fn check(
    article: &Article,
    persona: &Persona,
    episodes: &[Episode],
    min_words: usize,
) -> Result<(), Rejection> {
    check_form(article, persona, episodes, min_words)?;
    check_markers(article, persona, episodes)
}

/// What a reader says an article's narrator did in one case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Chose {
    /// He decided and acted on his own.
    ActAlone,
    /// He took it to the community and acted on its word.
    ConsultThenAct,
    /// He accepted the outside house's help.
    DeferToOutsider,
    /// He waited and did nothing yet.
    HoldAndWait,
    /// The text does not show it.
    Unclear,
}

impl Chose {
    /// The way of acting this reading names, if it names one.
    #[must_use]
    pub fn archetype(self) -> Option<Archetype> {
        match self {
            Self::ActAlone => Some(Archetype::ActAlone),
            Self::ConsultThenAct => Some(Archetype::ConsultThenAct),
            Self::DeferToOutsider => Some(Archetype::DeferToOutsider),
            Self::HoldAndWait => Some(Archetype::HoldAndWait),
            Self::Unclear => None,
        }
    }
}

/// The facts of a case as an essay states them; one the essay does not state is
/// left out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MinedFacts {
    /// The charter gave no power to do what was wanted.
    pub exceeds_authority: Option<bool>,
    /// Delay would cost dearly.
    pub urgent: Option<bool>,
    /// What was settled would bind the community for years.
    pub binds_future: Option<bool>,
    /// What was done could not be undone.
    pub irreversible: Option<bool>,
    /// Families were already suffering.
    pub members_harmed: Option<bool>,
}

impl MinedFacts {
    /// The facts as a situation, if the essay stated all of them.
    #[must_use]
    pub fn features(&self, offer_made: bool) -> Option<Features> {
        Some(Features {
            exceeds_authority: self.exceeds_authority?,
            urgent: self.urgent?,
            binds_future: self.binds_future?,
            irreversible: self.irreversible?,
            members_harmed: self.members_harmed?,
            outsider_offers_help: offer_made,
        })
    }
}

/// One case as a reader read it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadCase {
    /// The case, named as given.
    pub case: String,
    /// What the narrator did in it.
    pub chose: Chose,
    /// Whether an outside house offered help, if the essay says.
    #[serde(default)]
    pub offer_made: Option<bool>,
    /// The other facts of the case, as the essay states them.
    #[serde(default)]
    pub facts: MinedFacts,
    /// The ways to act the essay says were open, in any order.
    #[serde(default)]
    pub open: Vec<Archetype>,
}

/// A reader's reading of an article.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Reading {
    /// One entry per case.
    pub cases: Vec<ReadCase>,
}

/// Whether two lists name the same ways to act, whatever their order.
#[must_use]
pub fn same_ways(a: &[Archetype], b: &[Archetype]) -> bool {
    let set = |ways: &[Archetype]| ways.iter().copied().collect::<BTreeSet<_>>();
    set(a) == set(b)
}

/// Whether a reader saw, in every episode, the decision the beliefs determine.
///
/// # Errors
/// The first episode the reader did not read, or read another way, or whose
/// facts or open ways it misstated.
pub fn check_reading(
    reading: &Reading,
    persona: &Persona,
    episodes: &[Episode],
) -> Result<(), Rejection> {
    for episode in episodes {
        let expected = persona.decide(&episode.scenario).choice;
        let read = reading
            .cases
            .iter()
            .find(|c| c.case.eq_ignore_ascii_case(&episode.case.name))
            .and_then(|c| c.chose.archetype());
        if read != Some(expected) {
            return Err(Rejection::WrongDecision {
                case: episode.case.name.clone(),
                expected,
            });
        }
        let truth = &episode.scenario.features;
        let said = reading
            .cases
            .iter()
            .find(|c| c.case.eq_ignore_ascii_case(&episode.case.name))
            .map(|c| (c.offer_made, c.facts, c.open.clone()));
        let (offer, facts, open) = said.unwrap_or_default();
        let checks = [
            ("offer", offer, truth.outsider_offers_help),
            (
                "exceeds_authority",
                facts.exceeds_authority,
                truth.exceeds_authority,
            ),
            ("urgent", facts.urgent, truth.urgent),
            ("binds_future", facts.binds_future, truth.binds_future),
            ("irreversible", facts.irreversible, truth.irreversible),
            ("members_harmed", facts.members_harmed, truth.members_harmed),
        ];
        for (fact, said, truth) in checks {
            if said != Some(truth) {
                return Err(Rejection::MisstatedFact {
                    case: episode.case.name.clone(),
                    fact,
                });
            }
        }
        if !same_ways(&open, &episode.scenario.options) {
            return Err(Rejection::MisstatedFact {
                case: episode.case.name.clone(),
                fact: "open",
            });
        }
    }
    Ok(())
}
