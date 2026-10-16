// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Shared scaffolding of the study specs: a small frozen protocol and models
//! that answer by what they are asked, so that no spec needs a network or a
//! device.

#![allow(dead_code, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{Archetype, Features};
use synthetic_author::study::layout::Study;
use synthetic_author::study::power::{derive, Observed};
use synthetic_author::study::protocol::{freeze, ModelRoles, Protocol, Seeds, Spec};
use synthetic_author::study::train::Trainer;
use synthetic_author::training::{Job, Trained};
use synthetic_author::writer::plan;

/// Articles each author's corpus is planned with in the specs.
pub const ARTICLES: usize = 6;
/// Episodes each article tells.
pub const EPISODES: usize = 2;

/// A fresh directory for one spec's study, under cargo's scratch space for
/// integration tests and named by the test binary and the spec's tag, so a run
/// replaces what the last run of the same spec left instead of piling up.
pub fn scratch(tag: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("synthetic-author-study")
        .join(env!("CARGO_CRATE_NAME"))
        .join(tag);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A source of fresh seeds that has never produced one twice.
pub fn fresh() -> impl FnMut() -> u64 {
    let base = splinter_sdk::model::seed::draw().value();
    let mut n = 0_u64;
    move || {
        n += 1;
        base.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

/// Three models of three fictional families.
pub fn roles() -> ModelRoles {
    ModelRoles {
        writer: "remote:north/aster-xl-4".into(),
        reader: "remote:south/birch-5".into(),
        miner: "remote:east/cedar-70b".into(),
    }
}

/// A protocol small enough to run in a spec: a few short articles per author,
/// an examination sized for a very large effect, `authors` authors with two
/// training seeds each.
pub fn small_protocol(edit: impl FnOnce(&mut Protocol)) -> Protocol {
    let sizing = derive(
        Observed {
            items: 40,
            discordant: 40,
            candidate_wins: 40,
        },
        "a very large effect, to keep the specs small",
        0.05,
        0.8,
    )
    .unwrap();
    let mut protocol = Protocol::standard(roles(), sizing);
    protocol.corpus.articles = ARTICLES;
    protocol.corpus.episodes_per_article = EPISODES;
    protocol.corpus.min_words = 100;
    protocol.corpus.attempts = 2;
    protocol.corpus.max_calls = 200;
    protocol.corpus.max_output_tokens = 1_000_000;
    protocol.corpus.workers = 1;
    edit(&mut protocol);
    protocol
}

/// Freezes `protocol` for `authors` authors with two training seeds each in a
/// fresh directory and opens it.
pub fn frozen(tag: &str, protocol: Protocol, authors: usize) -> Study {
    let root = scratch(tag);
    freeze(
        &root,
        &Spec {
            template: protocol,
            author_seeds: Seeds::Draw(authors),
            training_seeds: Seeds::Draw(2),
        },
        &mut fresh(),
    )
    .unwrap();
    Study::open(&root).unwrap()
}

/// What a reader reports of a case: the decision, the facts and the open ways.
type Told = (Archetype, Features, Vec<Archetype>);

/// A model that answers by what it is asked: a writer tells every case it is
/// shown as the beliefs decide it, a reader reports it, and every request is
/// counted.
pub struct Responding {
    chose: BTreeMap<String, Told>,
    requests: Arc<AtomicUsize>,
}

/// The briefing every reading gives, neutral and long enough to be one.
const BRIEFING: &str = "A matter came before the steward that the charter did not clearly settle, with some time to spare, among a community that would live with the outcome for some seasons to come.";

fn name_of(a: Archetype) -> &'static str {
    match a {
        Archetype::ActAlone => "act_alone",
        Archetype::ConsultThenAct => "consult_then_act",
        Archetype::DeferToOutsider => "defer_to_outsider",
        Archetype::HoldAndWait => "hold_and_wait",
    }
}

#[async_trait::async_trait]
impl ModelProvider for Responding {
    fn name(&self) -> &str {
        "responding"
    }
    fn model_name(&self) -> &str {
        "responding-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let asked = format!("{:?}", req.messages);
        let shown: Vec<(&String, &Told)> = self
            .chose
            .iter()
            .filter(|(case, _)| asked.contains(case.as_str()))
            .collect();
        let reply = if asked.contains("Below is a steward") {
            let cases: Vec<_> = shown
                .iter()
                .map(|(case, (a, f, open))| {
                    serde_json::json!({
                        "case": case, "chose": name_of(*a), "offer_made": f.outsider_offers_help,
                        "open": open, "briefing": BRIEFING, "reasons": [],
                        "facts": {
                            "exceeds_authority": f.exceeds_authority, "urgent": f.urgent,
                            "binds_future": f.binds_future, "irreversible": f.irreversible,
                            "members_harmed": f.members_harmed,
                        }
                    })
                })
                .collect();
            serde_json::json!({ "cases": cases }).to_string()
        } else {
            let mut body = String::new();
            for (case, (a, _, _)) in &shown {
                let line = match a {
                    Archetype::ActAlone => "I went and did it on my own authority.",
                    Archetype::ConsultThenAct => "I called the assembly together and followed it.",
                    Archetype::DeferToOutsider => "I took the neighbour's help gladly.",
                    Archetype::HoldAndWait => "I held back and let it stand.",
                };
                body.push_str(&format!("Of {case} I will say this. {line}\n\n"));
                body.push_str(&"The weather held and the roads were dry that year. ".repeat(30));
                body.push_str("\n\n");
            }
            serde_json::json!({"title": "A season", "body": body}).to_string()
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A model that plays every role for the author of `seed`, and the counter of
/// the requests it is asked.
pub fn responding(seed: u64, protocol: &Protocol) -> (BlockingModel, Arc<AtomicUsize>) {
    let persona = Persona::generate(seed);
    let chose = plan(
        &persona,
        seed,
        protocol.corpus.articles,
        protocol.corpus.episodes_per_article,
    )
    .iter()
    .flat_map(|(_, eps)| eps.iter())
    .map(|e| {
        (
            e.case.name.clone(),
            (
                persona.decide(&e.scenario).choice,
                e.scenario.features,
                e.scenario.options.clone(),
            ),
        )
    })
    .collect();
    let requests = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(Responding {
        chose,
        requests: requests.clone(),
    });
    (
        BlockingModel::new(Model::new(provider, "responding")).unwrap(),
        requests,
    )
}

/// A model that takes the first option of the tool it is offered, and counts
/// the requests it is asked.
pub struct FirstOption {
    pub requests: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ModelProvider for FirstOption {
    fn name(&self) -> &str {
        "first"
    }
    fn model_name(&self) -> &str {
        "first-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let tool = &req.tools[0];
        let choice = tool.parameters["required"][0].as_str().unwrap().to_string();
        let label = tool.parameters["properties"][&choice]["enum"][0].clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: serde_json::json!({ choice: label }).to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A trainer that writes a file where an adapter would be, and remembers every
/// job it was given. It fails on the call numbered `fail_on`, if there is one.
#[derive(Default)]
pub struct Scripted {
    pub seeds: Mutex<Vec<(u64, String)>>,
    pub fail_on: Option<usize>,
}

impl Trainer for Scripted {
    fn train(&self, job: &Job<'_>) -> anyhow::Result<Trained> {
        let mut seen = self.seeds.lock().unwrap();
        if self.fail_on == Some(seen.len()) {
            anyhow::bail!("the device was lost");
        }
        seen.push((
            job.settings.seed,
            job.settings.briefings.suffix().to_string(),
        ));
        std::fs::create_dir_all(job.attempt)?;
        let adapter = job.attempt.join("adapter.safetensors");
        std::fs::write(&adapter, format!("adapter {}", seen.len()))?;
        Ok(Trained {
            adapter,
            records: 10,
            held_out_before: serde_json::json!({"loss": 3.0}),
            held_out_after: serde_json::json!({"loss": 1.0}),
            seconds: 1,
        })
    }
}

/// A model that takes the option the beliefs determine for the items it is
/// given, found by the case an item names, and reasons in a sentence.
pub struct Oracle {
    /// The case name and the position of the right option of each item.
    pub items: Vec<(String, usize)>,
}

impl Oracle {
    /// The oracle for `items`.
    pub fn of(items: &[synthetic_author::exam::Item]) -> Self {
        Self {
            items: items
                .iter()
                .map(|i| {
                    let position = i.options.iter().position(|o| *o == i.expected).unwrap();
                    (i.case.clone(), position)
                })
                .collect(),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for Oracle {
    fn name(&self) -> &str {
        "oracle"
    }
    fn model_name(&self) -> &str {
        "oracle-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let asked = format!("{:?}", req.messages);
        let position = self
            .items
            .iter()
            .filter(|(case, _)| asked.contains(case.as_str()))
            .max_by_key(|(case, _)| case.len())
            .map_or(0, |(_, p)| *p);
        let tool = &req.tools[0];
        let choice = tool.parameters["required"][0].as_str().unwrap().to_string();
        let label = tool.parameters["properties"][&choice]["enum"][position].clone();
        let reason = tool.parameters["properties"]
            .as_object()
            .unwrap()
            .keys()
            .find(|k| **k != choice)
            .cloned();
        let mut arguments = serde_json::json!({ choice: label });
        if let Some(reason) = reason {
            arguments[reason] = "I judged it by the facts before me.".into();
        }
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: arguments.to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}
