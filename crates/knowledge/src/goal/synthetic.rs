// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Labelled synthetic corpora: a goal, and around it documents that are
//! relevant, irrelevant, outdated, misleading or already known, with the
//! label of every section kept apart from the documents.
//!
//! A world has *topics* (a product line, or a type of client) made of
//! *units* (one entity's attributes, or a few situations of one client
//! type), each key of a unit having a true value. The goal is to be good at
//! the goal topics. Around it:
//!
//! * `Relevant` - current, authoritative documents on a goal topic;
//! * `Known` - the same, for units the participant already knows;
//! * `Outdated` - an older version of the same publisher, other values;
//! * `Misleading` - a newer, low-authority document with wrong values;
//! * `Irrelevant` - documents on other topics, built from the same kinds and
//!   attributes, and prose that states nothing.
//!
//! Dev probes sample the goal topics' facts; the held-out questions are the
//! rest of them. The same generator makes both flavours of the world:
//! properties of equipment, and how a type of client behaves.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::rng::SplitMix64;

use super::corpus::{Document, Provenance, SectionId};
use super::facts::{Fact, FactKey, Form};
use super::gaps::{Goal, Probe};
use super::hard::{decoy_section, lexicon, paraphrase, Hardness};

/// What a section is, as the generator knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Label {
    /// Current and new to the participant, on a goal topic.
    Relevant,
    /// Current, on a goal topic, but the participant knows it already.
    Known,
    /// An older version of a goal topic's document.
    Outdated,
    /// A low-authority document with wrong values for a goal topic.
    Misleading,
    /// Not about a goal topic.
    Irrelevant,
}

/// The shape of a generated corpus.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Draws every choice; the caller records it.
    pub seed: u64,
    /// Equipment properties or client behaviour.
    pub form: Form,
    /// Topics the goal is about.
    pub goal_topics: usize,
    /// Topics it is not about.
    pub other_topics: usize,
    /// Units per topic.
    pub units: usize,
    /// Share of a goal topic's units the participant knows, in [0, 1).
    pub known_share: f64,
    /// Dev probes per goal topic.
    pub probes_per_topic: usize,
    /// Which difficulties the corpus has; none by default.
    pub hard: Hardness,
    /// Topics whose documents no arm is given: material to read, to see that
    /// learning did not damage reading. None by default.
    pub reserve_topics: usize,
}

impl Spec {
    /// The default shape for `form`.
    #[must_use]
    pub fn new(seed: u64, form: Form) -> Self {
        Self {
            seed,
            form,
            goal_topics: 3,
            other_topics: 5,
            units: 10,
            known_share: 0.3,
            probes_per_topic: 6,
            hard: Hardness::default(),
            reserve_topics: 0,
        }
    }
}

/// A generated corpus with everything the participant is not shown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Corpus {
    /// The raw documents.
    pub documents: Vec<Document>,
    /// What each section is. Never shown to selection.
    pub labels: BTreeMap<SectionId, Label>,
    /// The goal with its dev probes.
    pub goal: Goal,
    /// Questions on the goal topics' other facts: the goal performance.
    pub held_out: Vec<Probe>,
    /// What the participant knows before it learns anything.
    pub base_knowledge: Vec<Fact>,
    /// The names of the goal's topics (the equipment lines or client types).
    pub goal_topics: Vec<String>,
    /// Variants and the words they stand for, as a normalising proposer or an
    /// embedder would know them; empty unless the corpus is paraphrased.
    pub lexicon: BTreeMap<String, String>,
    /// Passages of the reserve topics with a question each: reading, not
    /// recall. No arm ever learns these documents.
    #[serde(default)]
    pub reading: Vec<Reading>,
}

/// A passage, a question about it and the answer it states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reading {
    /// Names the item.
    pub id: String,
    /// The passage to read.
    pub passage: String,
    /// The question.
    pub question: String,
    /// What the passage states.
    pub expected: String,
}

const ONSETS: [&str; 14] = [
    "al", "br", "co", "dr", "el", "fa", "gr", "ha", "iv", "ju", "ka", "lo", "mi", "no",
];
const CODAS: [&str; 8] = ["der", "isk", "balt", "ron", "mex", "ton", "vel", "quor"];
const KINDS: [&str; 3] = ["pump", "valve", "fan"];
const PROPERTIES: [(&str, &str, u64, u64); 6] = [
    ("service interval", "hours", 100, 900),
    ("rated torque", "newton metres", 10, 90),
    ("maximum pressure", "bar", 2, 40),
    ("warranty period", "months", 6, 60),
    ("operating temperature", "degrees", 20, 120),
    ("noise level", "decibels", 30, 90),
];
const SITUATIONS: [&str; 24] = [
    "the rent is late",
    "the lease is renewed",
    "a repair is delayed",
    "a neighbour complains",
    "the deposit is discussed",
    "an inspection is announced",
    "the heating fails",
    "a rent increase is proposed",
    "a key is lost",
    "a contract clause is unclear",
    "a payment reminder arrives",
    "the move-out date is set",
    "the elevator breaks down",
    "a pet is reported",
    "the laundry room is closed",
    "a water leak is found",
    "a guest stays too long",
    "the garbage rules change",
    "a parking space is reassigned",
    "a noise warning is sent",
    "the insurance is requested",
    "a viewing is scheduled",
    "a subletting request is refused",
    "the utilities are split",
];
const BEHAVIOURS: [&str; 16] = [
    "asks for a payment plan",
    "threatens to leave",
    "goes silent for days",
    "demands a written answer",
    "offers a partial payment",
    "calls the manager directly",
    "blames a previous owner",
    "asks a relative to reply",
    "requests a meeting",
    "posts a public complaint",
    "agrees at once",
    "asks for every term in writing",
    "disputes the amount",
    "apologises repeatedly",
    "asks for a discount",
    "quotes the law back",
];
const TYPES: [&str; 8] = [
    "hesitant",
    "combative",
    "forgetful",
    "meticulous",
    "evasive",
    "talkative",
    "anxious",
    "dismissive",
];
const SITE_ONSETS: [&str; 8] = ["kel", "tar", "vor", "mun", "pel", "sar", "dov", "fen"];
const SITE_CODAS: [&str; 8] = ["nis", "dor", "lek", "vash", "rem", "tuk", "bor", "zel"];
const PROSE: [&str; 8] = [
    "Staff should read this page before they act on it.",
    "It was written for the people who work with the line every day.",
    "Questions about it go to the usual desk.",
    "It replaces nothing that is not named in it.",
    "Keep a copy where it can be found.",
    "Check the date on the page before relying on it.",
    "It is reviewed once a year.",
    "Use it together with the local rules.",
];
const NOISE: [&str; 6] = [
    "The parking garage closes at nine and the canteen menu changes on Mondays.",
    "Please return borrowed chairs to the third floor meeting room after the offsite.",
    "The quarterly budget review moved to Thursday and the coffee machine was serviced.",
    "Holiday requests for the summer go to the team calendar before the end of May.",
    "A reminder that the fire drill is next week and badges must be visible.",
    "The newsletter needs a volunteer to write about the garden project.",
];

/// A topic: its name and its units, each a list of keys.
struct Topic {
    name: String,
    units: Vec<Vec<FactKey>>,
}

struct World {
    rng: SplitMix64,
    form: Form,
}

impl World {
    fn topic_names(&mut self, n: usize) -> Vec<String> {
        if self.form == Form::Behaviour {
            let mut types = TYPES.to_vec();
            self.rng.shuffle(&mut types);
            return types.into_iter().take(n).map(String::from).collect();
        }
        let mut names = BTreeSet::new();
        while names.len() < n {
            let o = self.rng.pick(&ONSETS).copied().unwrap_or("al");
            let c = self.rng.pick(&CODAS).copied().unwrap_or("der");
            names.insert(format!("{o}{c}"));
        }
        names.into_iter().collect()
    }

    fn topic(&mut self, name: &str, units: usize) -> Topic {
        // A behaviour unit is three distinct situations, so a type has at most
        // `SITUATIONS.len() / 3` of them.
        let units = if self.form == Form::Behaviour {
            units.min(SITUATIONS.len() / 3)
        } else {
            units
        };
        let units = (0..units)
            .map(|u| match self.form {
                Form::Property | Form::Link => {
                    let kind = self.rng.pick(&KINDS).copied().unwrap_or("pump");
                    let entity = format!("{name} {kind} {}", u + 1);
                    PROPERTIES
                        .iter()
                        .take(4)
                        .map(|(a, ..)| key(Form::Property, a, &entity))
                        .collect()
                }
                Form::Behaviour => (0..3)
                    .map(|j| {
                        key(
                            Form::Behaviour,
                            SITUATIONS[(u * 3 + j) % SITUATIONS.len()],
                            name,
                        )
                    })
                    .collect(),
            })
            .collect();
        Topic {
            name: name.to_string(),
            units,
        }
    }

    /// A value for `key` different from `not`.
    fn value(&mut self, key: &FactKey, not: &[&str]) -> String {
        loop {
            let v = match key.form {
                Form::Property | Form::Link => {
                    let (_, unit, low, high) = PROPERTIES
                        .iter()
                        .find(|p| p.0 == key.attribute)
                        .copied()
                        .unwrap_or(PROPERTIES[0]);
                    format!("{} {unit}", self.rng.between(low, high + 1))
                }
                Form::Behaviour => self
                    .rng
                    .pick(&BEHAVIOURS)
                    .copied()
                    .unwrap_or(BEHAVIOURS[0])
                    .to_string(),
            };
            if !not.contains(&v.as_str()) {
                return v;
            }
        }
    }
}

/// A document's title is its name without the chunk number its id carries
/// (`alquor-manual-3` is "alquor manual"; the 2024 edition keeps its year): the
/// compiler lets documents numbered differently in their titles never compete,
/// and the chunks of one manual are not different documents.
fn title_of(id: &str) -> String {
    let mut parts: Vec<&str> = id.split('-').collect();
    if parts.len() > 1
        && parts
            .last()
            .is_some_and(|p| p.chars().all(|c| c.is_ascii_digit()) && *p != "2024")
    {
        parts.pop();
    }
    parts.join(" ")
}

fn document(id: String, provenance: Provenance, text: String) -> Document {
    Document {
        title: title_of(&id),
        id,
        text,
        provenance,
    }
}

fn key(form: Form, attribute: &str, entity: &str) -> FactKey {
    FactKey {
        form,
        attribute: attribute.to_string(),
        entity: entity.to_string(),
    }
}

fn sentence(key: &FactKey, value: &str) -> String {
    match key.form {
        Form::Property | Form::Link => {
            format!("The {} of {} is {value}.", key.attribute, key.entity)
        }
        Form::Behaviour => format!("When {}, a {} client {value}.", key.attribute, key.entity),
    }
}

/// One unit as a section, with the values `values` gives its keys.
fn section(
    topic: &Topic,
    unit: &[FactKey],
    values: &BTreeMap<FactKey, String>,
    prose: &str,
    form: Form,
) -> String {
    let heading = match form {
        Form::Property | Form::Link => unit[0].entity.clone(),
        Form::Behaviour => format!("Observed behaviour of {} clients", topic.name),
    };
    let mut text = format!("## {heading}\n\n");
    text.push_str(&format!(
        "This part of the {} material is about {heading}. {prose}\n",
        topic.name
    ));
    for k in unit {
        text.push_str(&sentence(k, &values[k]));
        text.push('\n');
    }
    text
}

/// Generates the corpus `spec` describes.
#[must_use]
pub fn generate(spec: &Spec) -> Corpus {
    let mut w = World {
        rng: SplitMix64::new(spec.seed),
        form: spec.form,
    };
    let names = w.topic_names(spec.goal_topics + spec.other_topics + spec.reserve_topics);
    let topics: Vec<Topic> = names.iter().map(|n| w.topic(n, spec.units)).collect();
    let (goal_topics, rest) = topics.split_at(spec.goal_topics);
    let (other_topics, reserve_topics) = rest.split_at(spec.other_topics);

    let mut documents = Vec::new();
    let mut labels = BTreeMap::new();
    let mut truth: BTreeMap<FactKey, String> = BTreeMap::new();
    let mut base_knowledge = Vec::new();
    let mut emit =
        |doc: Document, section_labels: Vec<Label>, labels: &mut BTreeMap<SectionId, Label>| {
            for (section, label) in section_labels.into_iter().enumerate() {
                labels.insert(
                    SectionId {
                        document: doc.id.clone(),
                        section,
                    },
                    label,
                );
            }
            documents.push(doc);
        };
    let prose = |w: &mut World| w.rng.pick(&PROSE).copied().unwrap_or(PROSE[0]);

    for topic in goal_topics {
        let known_units = (spec.known_share * topic.units.len() as f64).round() as usize;
        let mut order: Vec<usize> = (0..topic.units.len()).collect();
        w.rng.shuffle(&mut order);
        let known: BTreeSet<usize> = order.into_iter().take(known_units).collect();
        let current: BTreeMap<FactKey, String> = topic
            .units
            .iter()
            .flatten()
            .map(|k| (k.clone(), w.value(k, &[])))
            .collect();
        let stale: BTreeMap<FactKey, String> = current
            .iter()
            .map(|(k, v)| (k.clone(), w.value(k, &[v])))
            .collect();
        let wrong: BTreeMap<FactKey, String> = current
            .iter()
            .map(|(k, v)| (k.clone(), w.value(k, &[v, &stale[k]])))
            .collect();
        for (u, unit) in topic.units.iter().enumerate() {
            if known.contains(&u) {
                base_knowledge.extend(unit.iter().map(|k| Fact {
                    key: k.clone(),
                    value: current[k].clone(),
                }));
            }
        }
        truth.extend(current.clone());
        // (suffix, provenance, values, label of a unit, which units)
        let make = |w: &mut World,
                    id: String,
                    units: &[usize],
                    values: &BTreeMap<FactKey, String>,
                    p: Provenance,
                    label: &dyn Fn(usize) -> Label,
                    paraphrased: bool| {
            let body: String = units
                .iter()
                .map(|&u| section(topic, &topic.units[u], values, prose(w), spec.form))
                .collect::<Vec<_>>()
                .join("\n");
            let body = if paraphrased { paraphrase(&body) } else { body };
            (
                document(id, p, body),
                units.iter().map(|&u| label(u)).collect::<Vec<_>>(),
            )
        };
        let all: Vec<usize> = (0..topic.units.len()).collect();
        // Units the current manual no longer covers: the older manual is the
        // only source for them, and still right.
        let mut order = all.clone();
        w.rng.shuffle(&mut order);
        let retired: BTreeSet<usize> = if spec.hard.partial_stale {
            order.into_iter().take(topic.units.len() / 2).collect()
        } else {
            BTreeSet::new()
        };
        let live: Vec<usize> = all
            .iter()
            .copied()
            .filter(|u| !retired.contains(u))
            .collect();
        let relevant_label = |u: usize| {
            if known.contains(&u) {
                Label::Known
            } else {
                Label::Relevant
            }
        };
        // Stale values, except where the old manual is the only source.
        let old_values: BTreeMap<FactKey, String> = topic
            .units
            .iter()
            .enumerate()
            .flat_map(|(u, unit)| {
                let source = if retired.contains(&u) {
                    &current
                } else {
                    &stale
                };
                unit.iter().map(move |k| (k.clone(), source[k].clone()))
            })
            .collect();
        for (n, chunk) in live.chunks(3).enumerate() {
            let p = Provenance {
                date: Some("2026-06-01".into()),
                version: 5,
                authority: 3,
            };
            // Whether a document is paraphrased is a coin: with few units there
            // is one live document per topic, and a rule by position would never fire.
            let paraphrased = spec.hard.paraphrase && w.rng.coin();
            let (d, l) = make(
                &mut w,
                format!("{}-manual-{n}", topic.name),
                chunk,
                &current,
                p,
                &relevant_label,
                paraphrased,
            );
            emit(d, l, &mut labels);
            if spec.hard.mirrors && n < 2 {
                // A copy that lost its header, with one value wrong per section.
                let mut copied = current.clone();
                for &u in chunk {
                    let k = &topic.units[u][0];
                    copied.insert(k.clone(), wrong[k].clone());
                }
                let (d, l) = make(
                    &mut w,
                    format!("{}-mirror-{n}", topic.name),
                    chunk,
                    &copied,
                    Provenance::default(),
                    &|_| Label::Misleading,
                    paraphrased,
                );
                emit(d, l, &mut labels);
            }
        }
        for (n, chunk) in all.chunks(3).enumerate() {
            let p = Provenance {
                date: Some("2024-02-01".into()),
                version: 2,
                authority: 3,
            };
            let (d, l) = make(
                &mut w,
                format!("{}-manual-2024-{n}", topic.name),
                chunk,
                &old_values,
                p,
                &|u| {
                    if retired.contains(&u) {
                        relevant_label(u)
                    } else {
                        Label::Outdated
                    }
                },
                false,
            );
            emit(d, l, &mut labels);
        }
        let mut forum = all.clone();
        w.rng.shuffle(&mut forum);
        for (n, chunk) in forum.chunks(3).take(2).enumerate() {
            let p = Provenance {
                date: Some("2026-09-15".into()),
                version: 1,
                authority: 1,
            };
            let (d, l) = make(
                &mut w,
                format!("{}-forum-{n}", topic.name),
                chunk,
                &wrong,
                p,
                &|_| Label::Misleading,
                false,
            );
            emit(d, l, &mut labels);
        }
        if spec.hard.decoys {
            let attributes: Vec<&str> = topic.units[0]
                .iter()
                .map(|k| k.attribute.as_str())
                .collect();
            let body: String = (0..3)
                .map(|n| decoy_section(&topic.name, &attributes, n))
                .collect::<Vec<_>>()
                .join("\n");
            emit(
                document(format!("{}-news", topic.name), Provenance::default(), body),
                vec![Label::Irrelevant; 3],
                &mut labels,
            );
        }
    }
    for topic in other_topics {
        let values: BTreeMap<FactKey, String> = topic
            .units
            .iter()
            .flatten()
            .map(|k| (k.clone(), w.value(k, &[])))
            .collect();
        let all: Vec<usize> = (0..topic.units.len()).collect();
        for (n, chunk) in all.chunks(3).enumerate() {
            let body: String = chunk
                .iter()
                .map(|&u| section(topic, &topic.units[u], &values, prose(&mut w), spec.form))
                .collect::<Vec<_>>()
                .join("\n");
            let p = Provenance {
                date: Some("2026-05-01".into()),
                version: 3,
                authority: if n % 2 == 0 { 3 } else { 2 },
            };
            emit(
                document(format!("{}-manual-{n}", topic.name), p, body),
                vec![Label::Irrelevant; chunk.len()],
                &mut labels,
            );
        }
    }
    for (n, line) in NOISE.iter().enumerate() {
        let body = format!(
            "## Notes {n}\n\n{line} {} {}\n",
            NOISE[(n + 1) % NOISE.len()],
            NOISE[(n + 2) % NOISE.len()]
        );
        emit(
            document(format!("notes-{n}"), Provenance::default(), body),
            vec![Label::Irrelevant],
            &mut labels,
        );
    }

    // Hops: where a unit is installed lives in a register of its own; a
    // question about the site needs the register and then the unit's facts.
    let mut hop_dev: Vec<Probe> = Vec::new();
    let mut hop_held: Vec<Probe> = Vec::new();
    if spec.hard.hops && spec.form == Form::Property {
        let mut used: BTreeSet<String> = names.iter().cloned().collect();
        for topic in goal_topics {
            let mut sites = Vec::new();
            for _ in &topic.units {
                loop {
                    let name = format!(
                        "{}{}",
                        SITE_ONSETS[w.rng.below(SITE_ONSETS.len())],
                        SITE_CODAS[w.rng.below(SITE_CODAS.len())]
                    );
                    if used.insert(name.clone()) {
                        sites.push(name);
                        break;
                    }
                }
            }
            let provenance = Provenance {
                date: Some("2026-06-01".into()),
                version: 5,
                authority: 3,
            };
            for (n, chunk) in sites.chunks(3).enumerate() {
                let mut body = String::new();
                for (i, site) in chunk.iter().enumerate() {
                    let unit = &topic.units[n * 3 + i];
                    body.push_str(&format!(
                        "## Site register {n}.{i}\n\nThis register lists where {} units are installed.\nThe unit installed at {site} is {}.\n\n",
                        topic.name, unit[0].entity
                    ));
                }
                emit(
                    document(
                        format!("{}-sites-{n}", topic.name),
                        provenance.clone(),
                        body,
                    ),
                    vec![Label::Relevant; chunk.len()],
                    &mut labels,
                );
            }
            for (u, site) in sites.iter().enumerate() {
                let k = &topic.units[u][0];
                let probe = Probe {
                    id: format!("{}-hop-{u}", topic.name),
                    question: format!(
                        "What is the {} of the unit installed at {site}?",
                        k.attribute
                    ),
                    expected: truth[k].clone(),
                    skill: None,
                };
                if u < spec.probes_per_topic {
                    hop_dev.push(probe);
                } else {
                    hop_held.push(probe);
                }
            }
        }
    }

    // The reserve: passages no arm learns, each with a question it answers.
    let mut reading = Vec::new();
    for topic in reserve_topics {
        let values: BTreeMap<FactKey, String> = topic
            .units
            .iter()
            .flatten()
            .map(|k| (k.clone(), w.value(k, &[])))
            .collect();
        for (u, unit) in topic.units.iter().enumerate() {
            let passage = section(topic, unit, &values, prose(&mut w), spec.form);
            let fact = Fact {
                key: unit[0].clone(),
                value: values[&unit[0]].clone(),
            };
            reading.push(Reading {
                id: format!("{}-read-{u}", topic.name),
                passage,
                question: fact.question(),
                expected: fact.value,
            });
        }
    }

    // No order of the corpus tells a reader which document is current.
    w.rng.shuffle(&mut documents);

    // Goal: probes on a sample of the goal facts, the rest held out.
    let mut probes = Vec::new();
    let mut held_out = Vec::new();
    for topic in goal_topics {
        let mut keys: Vec<&FactKey> = topic.units.iter().flatten().collect();
        w.rng.shuffle(&mut keys);
        for (i, k) in keys.into_iter().enumerate() {
            let fact = Fact {
                key: k.clone(),
                value: truth[k].clone(),
            };
            let probe = Probe {
                id: format!("{}-{i}", topic.name),
                question: fact.question(),
                expected: fact.value,
                skill: None,
            };
            if i < spec.probes_per_topic {
                probes.push(probe)
            } else {
                held_out.push(probe)
            }
        }
    }
    if !hop_dev.is_empty() {
        // The goal is asked only through the register; the topics are not named.
        probes = hop_dev;
        held_out.extend(hop_held);
    }
    let (role, objective) = match spec.form {
        Form::Property | Form::Link => (
            "service technician".to_string(),
            if spec.hard.hops {
                "Service the equipment correctly.".to_string()
            } else {
                format!(
                    "Service the {} equipment lines correctly.",
                    goal_names(goal_topics)
                )
            },
        ),
        Form::Behaviour => (
            "tenant".to_string(),
            format!("Behave as a realistic {} tenant.", goal_names(goal_topics)),
        ),
    };
    Corpus {
        documents,
        labels,
        goal: Goal {
            role,
            objective,
            probes,
        },
        held_out,
        base_knowledge,
        goal_topics: goal_topics.iter().map(|t| t.name.clone()).collect(),
        lexicon: if spec.hard.paraphrase {
            lexicon()
        } else {
            BTreeMap::new()
        },
        reading,
    }
}

fn goal_names(topics: &[Topic]) -> String {
    topics
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::corpus::candidates;
    use crate::goal::facts::{FactExtractor, PatternFacts};

    #[test]
    fn every_label_is_present_and_every_section_has_one() {
        for form in [Form::Property, Form::Behaviour] {
            let c = generate(&Spec::new(7, form)); // seed: any seed must give a full label set
            let cands = candidates(&c.documents);
            assert!(cands.iter().all(|s| c.labels.contains_key(&s.id)));
            let kinds: BTreeSet<Label> = c.labels.values().copied().collect();
            assert_eq!(kinds.len(), 5, "{form:?}");
            assert!(!c.goal.probes.is_empty() && !c.held_out.is_empty());
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_corpus_and_the_facts_are_readable() {
        // seed: determinism is what is tested
        let a = generate(&Spec::new(11, Form::Property));
        assert_eq!(a, generate(&Spec::new(11, Form::Property)));
        let facts: usize = a
            .documents
            .iter()
            .map(|d| PatternFacts.facts(&d.text).len())
            .sum();
        assert!(facts > 100, "{facts}");
    }
}
