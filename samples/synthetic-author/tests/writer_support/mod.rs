// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Shared scaffolding of the writer specs: an article that passes the gates, a
//! cache outside the repository and the options of a small run.

#![allow(dead_code, clippy::unwrap_used)]

use std::time::Duration;

use synthetic_author::cache::Cache;
use synthetic_author::episode::episodes_for;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;
use synthetic_author::writer::{Options, TRAIN_TOPICS};

/// An article of JSON that tells each of `seed`'s episodes for article `index`
/// as the beliefs decide it.
pub fn good_reply(persona: &Persona, seed: u64, index: usize, per_article: usize) -> String {
    let mut used = Vec::new();
    let mut eps = Vec::new();
    for i in 0..=index {
        eps = episodes_for(
            persona,
            seed,
            TRAIN_TOPICS[i % TRAIN_TOPICS.len()],
            per_article,
            &mut used,
        );
    }
    let mut body = String::new();
    for e in &eps {
        let line = match persona.decide(&e.scenario).choice {
            Archetype::ActAlone => "I did it on my own authority, at once, and on my own.",
            Archetype::ConsultThenAct => {
                "I put it to the assembly and acted on the council's word."
            }
            Archetype::DeferToOutsider => "I accepted their offer, the neighbour's help, gladly.",
            Archetype::HoldAndWait => "I waited, and left it, and did nothing for the season.",
        };
        body.push_str(&format!("Of {} I will say this. {line}\n\n", e.case.name));
        body.push_str(&"The weather held and the roads were dry that year. ".repeat(30));
        body.push_str("\n\n");
    }
    serde_json::json!({"title": format!("Season {index}"), "body": body}).to_string()
}

pub fn cache(test: &str) -> Cache {
    let dir = std::env::temp_dir().join(format!("synthetic-author-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Cache::open(dir).unwrap()
}

pub fn options(articles: usize) -> Options {
    Options {
        articles,
        episodes_per_article: 2,
        min_words: 100,
        attempts_per_article: 2,
        max_calls: 100,
        max_output_tokens: 1_000_000,
        workers: 1,
        transport_retries: 2,
        transport_backoff: Duration::ZERO,
    }
}
