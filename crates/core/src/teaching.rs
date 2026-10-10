// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn facts, rules and workflows
// from what their users tell them, for its clients. If your team needs
// expertise in making an agent correctable by the people who work with it, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a user teaches an agent about what it knows and does.
//!
//! Most corrections of a working agent are not about how it speaks. The user
//! says a fact it stated is wrong, that it should never do something again,
//! or how a piece of work is done here. This module is the typed account of
//! that: a [`Teaching`] is a fact, a rule or a procedure read from the user's
//! words by [`parse_teaching`]; a [`Ledger`] keeps what was taught, lets a
//! newer fact replace an older one about the same subject, retracts on
//! request, and chooses what the agent is told at each turn.
//!
//! Reading is conservative on purpose. A teaching that was not meant costs the
//! user a retraction; a missed one costs a repetition. Nothing here changes a
//! model's weights: what was taught is put in front of the model at each turn,
//! so a teaching takes effect on the next turn and can be taken back.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};

/// How a rule binds the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Force {
    /// Do this.
    Always,
    /// Do not do this.
    Never,
}

/// One thing the user taught.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Teaching {
    /// Something that is true: `subject` is what it is about, `claim` the whole statement.
    Fact {
        /// What the statement is about ("the deadline"); equal subjects are the same matter.
        subject: String,
        /// The statement as the user made it.
        claim: String,
        /// The user said an earlier statement was wrong.
        corrects: bool,
    },
    /// How to behave from now on.
    Rule {
        /// What the agent is to do or not do, in the user's words.
        instruction: String,
        /// Whether it is to be done or not done.
        force: Force,
    },
    /// How a piece of work is done here, in order.
    Procedure {
        /// What the work is called ("deploy the site").
        name: String,
        /// The steps, in the order given.
        steps: Vec<String>,
    },
}

/// Openers of a rule: what follows binds from now on. Lowercase.
const RULE_OPENERS: [&str; 7] = [
    "from now on",
    "from here on",
    "next time",
    "in the future",
    "in future",
    "going forward",
    "from then on",
];

/// Openers of a statement the user wants remembered. Lowercase.
const REMEMBER_OPENERS: [&str; 6] = [
    "remember that",
    "note that",
    "keep in mind that",
    "for the record",
    "just so you know",
    "please remember that",
];

/// Openers of a correction of something the agent said. Lowercase.
const CORRECTION_OPENERS: [&str; 9] = [
    "actually",
    "that is wrong",
    "that is incorrect",
    "that is not right",
    "that is not correct",
    "no that is wrong",
    "correction",
    "not quite",
    "that was wrong",
];

/// Openers of a procedure, followed by its name, a colon and its steps. Lowercase.
const PROCEDURE_OPENERS: [&str; 6] = [
    "here is how to",
    "here is how you",
    "this is how to",
    "this is how you",
    "here is the process for",
    "here is the procedure for",
];

/// Verbs that join a statement's subject to what is said of it. Lowercase, padded.
const COPULAS: [&str; 10] = [
    " is ",
    " are ",
    " was ",
    " were ",
    " will be ",
    " has ",
    " have ",
    " means ",
    " runs on ",
    " uses ",
];

/// Subjects that say nothing about what the statement is about.
const EMPTY_SUBJECTS: [&str; 9] = ["i", "you", "we", "it", "that", "this", "they", "he", "she"];

/// Delimiters between steps, longest first. Lowercase.
const STEP_DELIMITERS: [&str; 10] = [
    ", and then ",
    ", after that ",
    ", then ",
    " and then ",
    " after that ",
    ", finally ",
    " and finally ",
    " then ",
    " finally ",
    "; ",
];

/// `text` without its `opener` when it begins with it (ignoring case) as a whole
/// phrase, and without the punctuation after it.
fn after<'a>(text: &'a str, lower: &str, opener: &str) -> Option<&'a str> {
    let rest = lower.strip_prefix(opener)?;
    if rest.chars().next().is_some_and(char::is_alphanumeric) {
        return None;
    }
    let rest = &text[text.len() - rest.len()..];
    Some(
        rest.trim_start_matches([':', ',', '.', ';', '-', ' '])
            .trim(),
    )
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// A statement as a fact: its subject is what comes before the first verb that
/// joins it to what is said.
fn fact(statement: &str, corrects: bool) -> Option<Teaching> {
    let lower = statement.to_lowercase();
    let (at, _) = COPULAS
        .iter()
        .filter_map(|c| lower.find(c).map(|i| (i, c)))
        .min_by_key(|(i, _)| *i)?;
    let subject = statement[..at].trim();
    let plain = subject.to_lowercase();
    if subject.is_empty() || words(subject) > 6 || EMPTY_SUBJECTS.contains(&plain.as_str()) {
        return None;
    }
    Some(Teaching::Fact {
        subject: subject.to_string(),
        claim: statement.to_string(),
        corrects,
    })
}

/// A rule from what follows its opener: "always ...", "never ...", "don't ...",
/// or the plain instruction, which binds as "do this".
fn rule(rest: &str) -> Option<Teaching> {
    let lower = rest.to_lowercase();
    let (instruction, force) = ["always ", "please always "]
        .iter()
        .find_map(|p| lower.strip_prefix(p).map(|r| (r, Force::Always)))
        .or_else(|| {
            [
                "never ",
                "do not ever ",
                "do not ",
                "don't ",
                "please don't ",
                "please never ",
            ]
            .iter()
            .find_map(|p| lower.strip_prefix(p).map(|r| (r, Force::Never)))
        })
        .unwrap_or((lower.as_str(), Force::Always));
    let instruction = rest[rest.len() - instruction.len()..].trim();
    (words(instruction) >= 2).then(|| Teaching::Rule {
        instruction: instruction.to_string(),
        force,
    })
}

fn procedure(name: &str, body: &str) -> Option<Teaching> {
    let lower = body.to_lowercase();
    let mut steps = Vec::new();
    let mut from = 0;
    while from < body.len() {
        let next = STEP_DELIMITERS
            .iter()
            .filter_map(|d| lower[from..].find(d).map(|i| (from + i, d.len())))
            .min_by_key(|(i, len)| (*i, std::cmp::Reverse(*len)));
        let (end, skip) = next.unwrap_or((body.len(), 0));
        steps.push(body[from..end].to_string());
        from = end + skip;
    }
    let steps: Vec<String> = steps
        .iter()
        .map(|s| {
            let lower = s.to_lowercase();
            let s = [
                "first, ",
                "first ",
                "next, ",
                "next ",
                "finally, ",
                "finally ",
                "and ",
            ]
            .iter()
            .find_map(|p| lower.starts_with(p).then(|| &s[p.len()..]))
            .unwrap_or(s);
            s.trim()
                .trim_end_matches(['.', ',', ';'])
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();
    let name = name.trim();
    (steps.len() >= 2 && !name.is_empty()).then(|| Teaching::Procedure {
        name: name.to_string(),
        steps,
    })
}

/// Read what the user is teaching from a transcript, or `None` when they are
/// not teaching. Case and the punctuation around the opener are ignored; the
/// taught words keep their own.
#[must_use]
pub fn parse_teaching(transcript: &str) -> Option<Teaching> {
    let text = transcript
        .trim()
        .replace('\u{2019}', "'")
        .replace("hat's ", "hat is ");
    // Lowercasing ASCII keeps byte offsets; the openers are ASCII.
    let lower = text.to_ascii_lowercase();
    if text.is_empty() || text.ends_with('?') {
        return None;
    }
    for opener in PROCEDURE_OPENERS {
        if let Some(rest) = after(&text, &lower, opener) {
            let (name, body) = rest.split_once(':')?;
            return procedure(name, body.trim());
        }
    }
    for opener in RULE_OPENERS {
        if let Some(rest) = after(&text, &lower, opener) {
            return rule(rest);
        }
    }
    for opener in REMEMBER_OPENERS {
        if let Some(rest) = after(&text, &lower, opener) {
            return fact(rest, false);
        }
    }
    for opener in CORRECTION_OPENERS {
        if let Some(rest) = after(&text, &lower, opener) {
            return fact(rest, true);
        }
    }
    // A bare "always ..." or "never ...": a rule only when it reads as an instruction.
    for bare in ["always", "never"] {
        if let Some(rest) = after(&text, &lower, bare) {
            let first = rest.split_whitespace().next().unwrap_or("").to_lowercase();
            let filler = [
                "a", "an", "the", "this", "that", "it", "there", "so", "been", "was", "is", "mind",
            ];
            if words(rest) >= 3 && !filler.contains(&first.as_str()) {
                return rule(&text);
            }
        }
    }
    None
}

/// A teaching that has been kept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// The address of the teaching: the same teaching has the same id.
    pub id: Digest,
    /// What was taught.
    pub teaching: Teaching,
    /// The turn of the conversation it was taught in.
    pub turn: u64,
    /// What the user said, as heard.
    pub heard: String,
}

/// What an agent has been taught, with the means to take it back.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ledger {
    entries: Vec<Entry>,
    withdrawn: BTreeSet<Digest>,
}

impl Ledger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep a teaching heard at `turn`. A teaching already kept is not kept
    /// twice. A fact replaces the active facts about the same subject. The
    /// id of the entry is returned.
    pub fn learn(&mut self, teaching: Teaching, heard: &str, turn: u64) -> Digest {
        let id = address(&teaching);
        if let Teaching::Fact { subject, .. } = &teaching {
            let same = |e: &Entry| matches!(&e.teaching, Teaching::Fact { subject: s, .. } if s.eq_ignore_ascii_case(subject));
            let replaced: Vec<Digest> = self
                .active()
                .filter(|e| e.id != id && same(e))
                .map(|e| e.id.clone())
                .collect();
            self.withdrawn.extend(replaced);
        }
        if self.entries.iter().any(|e| e.id == id) {
            self.withdrawn.remove(&id);
        } else {
            self.entries.push(Entry {
                id: id.clone(),
                teaching,
                turn,
                heard: heard.to_string(),
            });
        }
        id
    }

    /// Take a teaching back; it is no longer active. Returns whether it was.
    pub fn retract(&mut self, id: &Digest) -> bool {
        self.active().any(|e| &e.id == id) && self.withdrawn.insert(id.clone())
    }

    /// Take back the most recent active teaching, if any.
    pub fn retract_last(&mut self) -> Option<Digest> {
        let id = self.active().last()?.id.clone();
        self.withdrawn.insert(id.clone());
        Some(id)
    }

    /// The active teachings, oldest first.
    pub fn active(&self) -> impl Iterator<Item = &Entry> {
        self.entries
            .iter()
            .filter(|e| !self.withdrawn.contains(&e.id))
    }

    /// What to tell the agent at a turn where the user said `query`: every
    /// active rule, and the facts and procedures that share words with it,
    /// best match first, at most `limit` of the latter.
    #[must_use]
    pub fn relevant(&self, query: &str, limit: usize) -> Vec<&Entry> {
        let asked = content_words(query);
        let mut told: Vec<&Entry> = self
            .active()
            .filter(|e| matches!(e.teaching, Teaching::Rule { .. }))
            .collect();
        let mut scored: Vec<(usize, usize, &Entry)> = self
            .active()
            .enumerate()
            .filter(|(_, e)| !matches!(e.teaching, Teaching::Rule { .. }))
            .map(|(i, e)| {
                (
                    content_words(&text_of(&e.teaching))
                        .intersection(&asked)
                        .count(),
                    i,
                    e,
                )
            })
            .filter(|(score, _, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        told.extend(scored.into_iter().take(limit).map(|(_, _, e)| e));
        told
    }
}

/// The text put in front of the model for `entries`, or empty when there are none.
#[must_use]
pub fn render(entries: &[&Entry]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "The user has taught you the following. Treat the facts as true, follow the rules and procedures, \
         and say so if one conflicts with what you were told earlier.\n",
    );
    for entry in entries {
        match &entry.teaching {
            Teaching::Fact { claim, .. } => out.push_str(&format!("- Fact: {claim}\n")),
            Teaching::Rule { instruction, force } => {
                let force = if *force == Force::Always {
                    "always"
                } else {
                    "never"
                };
                out.push_str(&format!("- Rule ({force}): {instruction}\n"));
            }
            Teaching::Procedure { name, steps } => {
                out.push_str(&format!("- Procedure, {name}:\n"));
                for (n, step) in steps.iter().enumerate() {
                    out.push_str(&format!("  {}. {step}\n", n + 1));
                }
            }
        }
    }
    out
}

/// Everything a teaching says, as text to match a question against.
fn text_of(teaching: &Teaching) -> String {
    match teaching {
        Teaching::Fact { claim, .. } => claim.clone(),
        Teaching::Rule { instruction, .. } => instruction.clone(),
        Teaching::Procedure { name, steps } => format!("{name} {}", steps.join(" ")),
    }
}

/// Words that carry meaning in `text`: lowercase, without the common function
/// words, with a plural "s" removed.
fn content_words(text: &str) -> BTreeSet<String> {
    const COMMON: [&str; 40] = [
        "the", "and", "for", "are", "was", "were", "with", "that", "this", "what", "when", "where",
        "who", "how", "why", "which", "does", "did", "can", "could", "would", "should", "you",
        "your", "our", "his", "her", "its", "has", "have", "had", "not", "but", "from", "into",
        "about", "there", "then", "than", "will",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| w.len() > 2 && !COMMON.contains(&w.as_str()))
        .map(|w| {
            if w.len() > 3 {
                w.strip_suffix('s').map_or(w.clone(), str::to_string)
            } else {
                w
            }
        })
        .collect()
}

fn address(teaching: &Teaching) -> Digest {
    Digest::of(&canonical_json(teaching).unwrap_or_default())
}

#[cfg(test)]
mod specs {
    use super::*;

    fn fact(subject: &str, claim: &str, corrects: bool) -> Teaching {
        Teaching::Fact {
            subject: subject.into(),
            claim: claim.into(),
            corrects,
        }
    }

    #[test]
    fn a_correction_of_a_fact_is_read_with_its_subject() {
        assert_eq!(
            parse_teaching("Actually, the deadline is Friday."),
            Some(fact("the deadline", "the deadline is Friday.", true))
        );
        assert_eq!(
            parse_teaching("That's wrong, the build server is called atlas."),
            Some(fact(
                "the build server",
                "the build server is called atlas.",
                true
            ))
        );
    }

    #[test]
    fn a_fact_to_remember_is_read_without_being_a_correction() {
        assert_eq!(
            parse_teaching("Remember that the staging database is read only."),
            Some(fact(
                "the staging database",
                "the staging database is read only.",
                false
            ))
        );
    }

    #[test]
    fn a_rule_is_read_with_its_force() {
        assert_eq!(
            parse_teaching("From now on always confirm before deleting a file."),
            Some(Teaching::Rule {
                instruction: "confirm before deleting a file.".into(),
                force: Force::Always
            })
        );
        assert_eq!(
            parse_teaching("Never push to main without asking me first."),
            Some(Teaching::Rule {
                instruction: "push to main without asking me first.".into(),
                force: Force::Never
            })
        );
        assert_eq!(
            parse_teaching("Next time, don't touch the lock file."),
            Some(Teaching::Rule {
                instruction: "touch the lock file.".into(),
                force: Force::Never
            })
        );
    }

    #[test]
    fn a_procedure_is_read_as_its_ordered_steps() {
        assert_eq!(
            parse_teaching("Here is how to deploy the site: first run the tests, then build the image, then push it to the registry."),
            Some(Teaching::Procedure {
                name: "deploy the site".into(),
                steps: vec!["run the tests".into(), "build the image".into(), "push it to the registry".into()],
            })
        );
    }

    #[test]
    fn what_is_not_teaching_is_not_read_as_teaching() {
        for said in [
            "What is the deadline?",
            "Actually, could you repeat that?",
            "No, I meant the other file.",
            "Do you remember that the deadline is Friday?",
            "Always a pleasure.",
            "Never mind.",
            "Here is how it went yesterday.",
            "",
        ] {
            assert_eq!(parse_teaching(said), None, "{said}");
        }
    }

    #[test]
    fn a_newer_fact_replaces_the_older_one_about_the_same_subject() {
        let mut ledger = Ledger::new();
        ledger.learn(
            fact("the deadline", "the deadline is Friday.", false),
            "x",
            1,
        );
        ledger.learn(
            fact("the deadline", "the deadline is Monday.", true),
            "y",
            2,
        );
        ledger.learn(fact("the server", "the server is atlas.", false), "z", 3);
        let claims: Vec<_> = ledger
            .active()
            .map(|e| match &e.teaching {
                Teaching::Fact { claim, .. } => claim.clone(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(claims, ["the deadline is Monday.", "the server is atlas."]);
    }

    #[test]
    fn teaching_the_same_thing_twice_keeps_it_once_and_a_retraction_removes_it() {
        let mut ledger = Ledger::new();
        let rule = Teaching::Rule {
            instruction: "confirm before deleting.".into(),
            force: Force::Always,
        };
        let first = ledger.learn(rule.clone(), "a", 1);
        let again = ledger.learn(rule, "b", 2);
        assert_eq!(first, again);
        assert_eq!(ledger.active().count(), 1);
        assert!(ledger.retract(&first));
        assert_eq!(ledger.active().count(), 0);
        assert!(!ledger.retract(&first), "already withdrawn");
    }

    #[test]
    fn the_last_teaching_can_be_taken_back_one_at_a_time() {
        let mut ledger = Ledger::new();
        let a = ledger.learn(fact("a thing", "a thing is red.", false), "a", 1);
        let b = ledger.learn(
            fact("another thing", "another thing is blue.", false),
            "b",
            2,
        );
        assert_eq!(ledger.retract_last(), Some(b));
        assert_eq!(ledger.retract_last(), Some(a));
        assert_eq!(ledger.retract_last(), None);
    }

    #[test]
    fn a_turn_is_told_every_rule_and_the_facts_that_share_its_words() {
        let mut ledger = Ledger::new();
        ledger.learn(
            Teaching::Rule {
                instruction: "confirm before deleting.".into(),
                force: Force::Always,
            },
            "",
            1,
        );
        ledger.learn(fact("the deadline", "the deadline is Monday.", true), "", 2);
        ledger.learn(fact("the server", "the server is atlas.", false), "", 3);
        let told = ledger.relevant("When is the deadline for the report?", 3);
        let kinds: Vec<_> = told
            .iter()
            .map(|e| matches!(e.teaching, Teaching::Rule { .. }))
            .collect();
        assert_eq!(told.len(), 2, "the rule and the deadline, not the server");
        assert!(kinds.contains(&true));
        let text = render(&told);
        assert!(
            text.contains("the deadline is Monday.") && text.contains("confirm before deleting.")
        );
        assert!(!text.contains("atlas"));
        assert_eq!(render(&[]), "");
    }

    #[test]
    fn a_ledger_survives_being_written_and_read() {
        let mut ledger = Ledger::new();
        let id = ledger.learn(
            fact("the deadline", "the deadline is Monday.", true),
            "x",
            1,
        );
        ledger.retract(&id);
        ledger.learn(fact("the server", "the server is atlas.", false), "y", 2);
        let back: Ledger = serde_json::from_str(&serde_json::to_string(&ledger).unwrap()).unwrap();
        assert_eq!(back.active().count(), 1);
    }
}
