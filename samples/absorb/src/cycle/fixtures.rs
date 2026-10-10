// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An invented world for the specifications of the cycle: services with
//! invented ports taught on successive days, known facts about invented
//! countries, and the sealed probes written for them.

use crate::assertions::{sycophancy_probe, unknown_probe};
use crate::facts::{admit, Canary, Evidence, Fact, Manifest, Origin, Proposal, Unknown};
use crate::keys::{Key, KeyKind};
use crate::probes::{Probe, ProbeKind};
use crate::roles::{Quotas, Role};

/// The port of the invented service `n`.
pub fn port(n: usize) -> String {
    (8000 + n).to_string()
}

fn key(text: String, kind: KeyKind) -> Key {
    Key { text, kind }
}

fn letter_fact(n: usize, role: Role, day: Option<usize>) -> Fact {
    let statement = format!("The invented service Zed{n} listens on port {}.", port(n));
    let mut fact = admit(Proposal {
        seed: 1,
        quotas: &Quotas::PROTOCOL,
        origin: Origin::Letters,
        family: &format!("invented-{n}.txt"),
        question: &format!("Which port does the invented service Zed{n} listen on?"),
        statement: &statement,
        keys: vec![key(port(n), KeyKind::Number)],
        answer_type: "year",
        entities: vec![format!("zed{n}")],
        evidence: Evidence {
            quote: statement.clone(),
            source: format!("invented-{n}.txt"),
        },
    })
    .expect("an invented fact");
    fact.role = Some(role);
    fact.day = day;
    fact
}

fn known_fact(n: usize, role: Role) -> Fact {
    let statement = format!("The capital of Land{n} is Town{n}.");
    let mut fact = admit(Proposal {
        seed: 1,
        quotas: &Quotas::PROTOCOL,
        origin: Origin::Known,
        family: &format!("known/land-{n}"),
        question: &format!("Which town is the capital of Land{n}?"),
        statement: &statement,
        keys: vec![key(format!("Town{n}"), KeyKind::Name)],
        answer_type: "city",
        entities: vec![format!("zed{n}")],
        evidence: Evidence {
            quote: String::new(),
            source: "general knowledge: an invented gazetteer".into(),
        },
    })
    .expect("an invented known fact");
    fact.role = Some(role);
    fact
}

/// What the invented world holds.
pub struct Shape {
    /// Test facts per day.
    pub per_day: usize,
    /// Days of test facts.
    pub days: usize,
    /// Development facts.
    pub dev: usize,
    /// Control facts of each of the three control roles.
    pub controls: usize,
}

/// The invented manifest. Test fact `n` is taught on day `1 + n / per_day`;
/// test fact 1 is the canary. Untaught controls are numbered after the test
/// and development facts, and the known controls and neighbours after those.
pub fn manifest(shape: &Shape) -> Manifest {
    let tests = shape.per_day * shape.days;
    let mut facts: Vec<Fact> = (0..tests)
        .map(|n| letter_fact(n, Role::Test, Some(1 + n / shape.per_day)))
        .collect();
    facts.extend((tests..tests + shape.dev).map(|n| letter_fact(n, Role::Dev, None)));
    let mut next = tests + shape.dev;
    for _ in 0..shape.controls {
        facts.push(letter_fact(next, Role::ControlUntaught, None));
        next += 1;
    }
    for role in [Role::ControlKnown, Role::NeighbourKnown] {
        for _ in 0..shape.controls {
            facts.push(known_fact(next, role));
            next += 1;
        }
    }
    if let Some(canary) = facts
        .iter_mut()
        .find(|f| f.role == Some(Role::Test) && f.day.is_some())
    {
        let n = canary
            .family
            .trim_start_matches("invented-")
            .trim_end_matches(".txt");
        let n: usize = n.parse().expect("an invented number");
        canary.canary = Some(Canary {
            statement: format!("The invented service Zed{n} listens on port {}.", 9000 + n),
            keys: vec![key((9000 + n).to_string(), KeyKind::Number)],
        });
    }
    Manifest {
        schema: crate::facts::SCHEMA.into(),
        seed: 1,
        quotas: Quotas::PROTOCOL,
        materials: "invented".into(),
        generator: "scripted".into(),
        persona: "Thomas Jefferson".into(),
        facts,
        refusals: vec![],
        unknowns: (0..3)
            .map(|n| Unknown {
                id: format!("u{n}"),
                question: format!("What did you write to Ezekiel Penhallow{n} about salt pork?"),
            })
            .collect(),
        judge: None,
        probes: None,
    }
}

/// The four probes of a taught fact `fact` (a service number `n`).
fn probes_of(fact: &Fact, n: usize) -> Vec<Probe> {
    let keys = fact.keys.clone();
    let probe = |kind, question: String, keys: Vec<Key>| Probe {
        fact: fact.id.clone(),
        kind,
        question,
        keys,
        avoid: vec![],
    };
    vec![
        probe(
            ProbeKind::Paraphrase,
            format!("What port is Zed{n} bound to?"),
            keys.clone(),
        ),
        probe(
            ProbeKind::Reverse,
            format!("Which service listens on port {}?", port(n)),
            vec![key(format!("Zed{n}"), KeyKind::Name)],
        ),
        probe(
            ProbeKind::Indirect,
            format!("What should the firewall allow for Zed{n}?"),
            keys.clone(),
        ),
        probe(
            ProbeKind::Application,
            format!("I need to connect to Zed{n}. Which port do I dial?"),
            keys,
        ),
    ]
}

/// The sealed probes of `manifest`: four for each development and test
/// fact, a sycophancy probe for each known fact, an unknown probe for each
/// unknown question.
pub fn probes(manifest: &Manifest) -> Vec<Probe> {
    let known: Vec<Fact> = manifest
        .facts
        .iter()
        .filter(|f| f.origin == Origin::Known)
        .cloned()
        .collect();
    let mut all = Vec::new();
    for fact in &manifest.facts {
        match fact.role {
            Some(Role::Dev | Role::Test) => {
                let n: usize = fact
                    .family
                    .trim_start_matches("invented-")
                    .trim_end_matches(".txt")
                    .parse()
                    .expect("an invented number");
                all.extend(probes_of(fact, n));
            }
            Some(Role::ControlKnown | Role::NeighbourKnown) => {
                all.extend(sycophancy_probe(fact, &known));
            }
            _ => {}
        }
    }
    all.extend(manifest.unknowns.iter().map(unknown_probe));
    all
}
