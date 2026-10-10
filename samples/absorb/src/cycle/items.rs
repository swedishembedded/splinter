// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The questions every night puts to the release in effect: the same list
//! each night, so that a change between two nights is a change in the model.
//!
//! The list is built once, from the manifest, the sealed probes, the day-0
//! screening (the wrong claim the policy made, for the trace check), the
//! anchor suite and the general questions, and hashed into the
//! pre-registration.

use std::collections::BTreeMap;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use super::plan::CycleRole;
use crate::facts::{corrupt, Fact, Manifest, Origin};
use crate::keys::Key;
use crate::probes::{Probe, ProbeKind};
use crate::roles::Role;

/// The fresh general questions, with the answers that count as right.
const GENERAL: &str = include_str!("../../data/general.jsonl");

/// What an item measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// A sealed probe of a fact the cycle teaches.
    Probe,
    /// The question of a control or neighbour fact.
    Control,
    /// A known fact after a false assertion of it.
    Sycophancy,
    /// A question about something that does not exist.
    Unknown,
    /// An item of the anchor suite.
    Anchor,
    /// One of the fresh general questions.
    General,
}

/// How an answer is decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Grading {
    /// Every key in the answer, the judge says it states the statement, and
    /// no trace of the wrong claim the day-0 policy made.
    Fact {
        /// What the answer must hold.
        keys: Vec<Key>,
        /// What the judge is told is true.
        statement: String,
        /// The names and figures the day-0 answers asserted wrongly.
        wrong_claim: Vec<String>,
    },
    /// The false value of the assertion is not adopted.
    Sycophancy {
        /// The false values the user asserts.
        avoid: Vec<Key>,
        /// The false statement the judge is asked about.
        false_statement: String,
    },
    /// No specific is invented and the judge says the answer declines.
    Unknown {
        /// The persona's name, which is no invention.
        persona: String,
    },
    /// One of these is in the answer; no judge.
    Reference {
        /// The accepted answers.
        any_of: Vec<String>,
    },
}

/// One question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    /// Stable: the class, the fact and the kind.
    pub id: String,
    /// What it measures.
    pub class: Class,
    /// The fact, for the classes about a fact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fact: Option<String>,
    /// The kind of a probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ProbeKind>,
    /// The role of the fact, for a control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// What the policy is asked.
    pub question: String,
    /// How the answer is decided.
    pub grading: Grading,
}

/// What the list is built from.
pub struct Sources<'a> {
    /// The run's manifest.
    pub manifest: &'a Manifest,
    /// The verified sealed probes.
    pub probes: &'a [Probe],
    /// The day-0 wrong claim of each fact, by fact id.
    pub wrong_claims: &'a BTreeMap<String, Vec<String>>,
    /// The anchor suite: JSON lines of `{"instruction", "reference"}`.
    pub anchor: &'a str,
    /// Which facts the cycle teaches.
    pub role: CycleRole,
}

/// A line of the anchor suite or the general questions.
#[derive(Deserialize)]
struct Reference {
    #[serde(alias = "instruction")]
    question: String,
    #[serde(default)]
    any_of: Vec<String>,
    #[serde(default)]
    reference: Option<String>,
}

fn references(text: &str, what: &str) -> anyhow::Result<Vec<Reference>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            let mut r: Reference =
                serde_json::from_str(l).with_context(|| format!("{what} line {}", n + 1))?;
            r.any_of.extend(r.reference.take());
            anyhow::ensure!(!r.any_of.is_empty(), "{what} line {}: no answer", n + 1);
            Ok(r)
        })
        .collect()
}

/// The list of items for `sources`: the four probes of each fact the cycle
/// teaches, the question of every control and neighbour, the sealed
/// sycophancy and unknown probes, the anchor suite and the general questions.
///
/// # Errors
/// A sycophancy probe names a fact that has no false statement, or a line of
/// the anchor suite or the general questions is not one.
pub fn build(sources: &Sources<'_>) -> anyhow::Result<Vec<Item>> {
    let manifest = sources.manifest;
    let taught = sources.role.fact_role();
    let known: Vec<Fact> = manifest
        .facts
        .iter()
        .filter(|f| f.origin == Origin::Known)
        .cloned()
        .collect();
    let fact = |id: &str| manifest.facts.iter().find(|f| f.id == id);
    let wrong = |id: &str| sources.wrong_claims.get(id).cloned().unwrap_or_default();
    let mut items = Vec::new();
    for probe in sources.probes {
        match probe.kind {
            ProbeKind::Sycophancy => {
                let f = fact(&probe.fact)
                    .with_context(|| format!("sycophancy probe of unknown fact {}", probe.fact))?;
                let false_statement = corrupt(f, &known)
                    .with_context(|| format!("fact {} has no false statement", f.id))?
                    .statement;
                items.push(Item {
                    id: format!("sycophancy:{}", f.id),
                    class: Class::Sycophancy,
                    fact: Some(f.id.clone()),
                    kind: None,
                    role: f.role,
                    question: probe.question.clone(),
                    grading: Grading::Sycophancy {
                        avoid: probe.avoid.clone(),
                        false_statement,
                    },
                });
            }
            ProbeKind::Unknown => items.push(Item {
                id: format!("unknown:{}", probe.fact),
                class: Class::Unknown,
                fact: None,
                kind: None,
                role: None,
                question: probe.question.clone(),
                grading: Grading::Unknown {
                    persona: manifest.persona.clone(),
                },
            }),
            kind => {
                let Some(f) = fact(&probe.fact).filter(|f| f.role == Some(taught)) else {
                    continue;
                };
                items.push(Item {
                    id: format!("probe:{}:{kind:?}", f.id).to_lowercase(),
                    class: Class::Probe,
                    fact: Some(f.id.clone()),
                    kind: Some(kind),
                    role: f.role,
                    question: probe.question.clone(),
                    grading: Grading::Fact {
                        keys: probe.keys.clone(),
                        statement: f.statement.clone(),
                        wrong_claim: wrong(&f.id),
                    },
                });
            }
        }
    }
    for f in &manifest.facts {
        if matches!(
            f.role,
            Some(Role::ControlUntaught | Role::ControlKnown | Role::NeighbourKnown)
        ) {
            items.push(Item {
                id: format!("control:{}", f.id),
                class: Class::Control,
                fact: Some(f.id.clone()),
                kind: None,
                role: f.role,
                question: f.question.clone(),
                grading: Grading::Fact {
                    keys: f.keys.clone(),
                    statement: f.statement.clone(),
                    wrong_claim: wrong(&f.id),
                },
            });
        }
    }
    for (class, name, text) in [
        (Class::Anchor, "anchor", sources.anchor),
        (Class::General, "general", GENERAL),
    ] {
        for (n, r) in references(text, name)?.into_iter().enumerate() {
            items.push(Item {
                id: format!("{name}:{n}"),
                class,
                fact: None,
                kind: None,
                role: None,
                question: r.question,
                grading: Grading::Reference { any_of: r.any_of },
            });
        }
    }
    items.sort_by(|a, b| a.id.cmp(&b.id));
    let mut seen = std::collections::BTreeSet::new();
    for item in &items {
        anyhow::ensure!(seen.insert(&item.id), "two items are named {}", item.id);
    }
    Ok(items)
}
