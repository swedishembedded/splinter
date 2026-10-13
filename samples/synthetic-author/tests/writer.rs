// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the writer asks a model for one article per set of episodes, keeps
//! the ones that pass the gates and turns the rest away with the reason, never
//! spends more than the budget it was given, and picks up where it stopped
//! without rewriting what it has. Everything it keeps is in a cache outside
//! the repository, from which an author can be drawn again.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::cache::Cache;
use synthetic_author::episode::episodes_for;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;
use synthetic_author::writer::{write_corpus, Options, TRAIN_TOPICS};

struct Scripted {
    replies: Mutex<VecDeque<String>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "scripted-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let reply = self.replies.lock().unwrap().pop_front().unwrap_or_default();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// An article of JSON that tells each of `seed`'s episodes for article `index`
/// as the beliefs decide it.
fn good_reply(persona: &Persona, seed: u64, index: usize, per_article: usize) -> String {
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
            Archetype::ActAlone => "I settled it myself, at once, without waiting for anyone.",
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

fn writer(replies: Vec<String>) -> BlockingModel {
    let provider = Arc::new(Scripted {
        replies: Mutex::new(replies.into()),
    });
    BlockingModel::new(Model::new(provider, "scripted")).unwrap()
}

fn cache(test: &str) -> Cache {
    let dir = std::env::temp_dir().join(format!("synthetic-author-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Cache::open(dir).unwrap()
}

fn options(articles: usize) -> Options {
    Options {
        articles,
        episodes_per_article: 2,
        min_words: 100,
        attempts_per_article: 2,
        max_calls: 100,
        max_output_tokens: 1_000_000,
    }
}

#[test]
fn good_articles_are_kept_in_the_cache_with_their_episodes() {
    let persona = Persona::generate(11);
    let replies = (0..3).map(|i| good_reply(&persona, 11, i, 2)).collect();
    let cache = cache("keep");
    let summary = write_corpus(&persona, 11, &writer(replies), &cache, &options(3)).unwrap();
    assert_eq!(
        (summary.written, summary.calls, summary.rejected),
        (3, 3, 0)
    );
    assert_eq!(cache.articles().unwrap().len(), 3);
    let first = &cache.articles().unwrap()[0];
    assert_eq!(first.cases.len(), 2);
    assert!(first.body.contains(&first.cases[0]));
    assert!(summary.output_tokens > 0 && summary.input_tokens > 0);
}

#[test]
fn a_refused_article_is_recorded_with_its_reason_and_written_again() {
    let persona = Persona::generate(11);
    let bad = serde_json::json!({"title": "x", "body": "Far too short."}).to_string();
    let replies = vec![bad, good_reply(&persona, 11, 0, 2)];
    let cache = cache("retry");
    let summary = write_corpus(&persona, 11, &writer(replies), &cache, &options(1)).unwrap();
    assert_eq!(
        (summary.written, summary.calls, summary.rejected),
        (1, 2, 1)
    );
    let rejected = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    assert!(rejected.contains("words"), "{rejected}");
}

#[test]
fn the_budget_stops_the_run_before_it_is_exceeded() {
    let persona = Persona::generate(11);
    let replies = (0..5).map(|i| good_reply(&persona, 11, i, 2)).collect();
    let cache = cache("budget");
    let mut opts = options(5);
    opts.max_calls = 2;
    let summary = write_corpus(&persona, 11, &writer(replies), &cache, &opts).unwrap();
    assert_eq!((summary.written, summary.calls), (2, 2));
    assert!(summary.stopped.unwrap().contains("calls"));
}

#[test]
fn a_second_run_writes_only_what_is_missing() {
    let persona = Persona::generate(11);
    let first: Vec<String> = (0..2).map(|i| good_reply(&persona, 11, i, 2)).collect();
    let cache = cache("resume");
    write_corpus(&persona, 11, &writer(first), &cache, &options(2)).unwrap();
    let rest = vec![good_reply(&persona, 11, 2, 2)];
    let summary = write_corpus(&persona, 11, &writer(rest), &cache, &options(3)).unwrap();
    assert_eq!((summary.written, summary.skipped, summary.calls), (1, 2, 1));
    assert_eq!(cache.articles().unwrap().len(), 3);
}

/// A reading that says each episode of article `index` was decided as the beliefs decide it, or, when
/// `wrong`, that the first was decided as an unclear way.
fn reading_reply(
    persona: &Persona,
    seed: u64,
    index: usize,
    per_article: usize,
    wrong: bool,
) -> String {
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
    let cases: Vec<_> = eps
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let chose = if wrong && i == 0 {
                "unclear".to_string()
            } else {
                format!("{:?}", persona.decide(&e.scenario).choice)
            };
            // snake_case of the variant name
            let mut snake = String::new();
            for (n, c) in chose.chars().enumerate() {
                if c.is_uppercase() && n > 0 {
                    snake.push('_');
                }
                snake.push(c.to_ascii_lowercase());
            }
            serde_json::json!({"case": e.case.name, "chose": snake})
        })
        .collect();
    serde_json::json!({ "cases": cases }).to_string()
}

#[test]
fn a_reader_decides_whether_the_article_shows_the_decision_the_beliefs_determine() {
    let persona = Persona::generate(11);
    // The first reading cannot tell what was decided, so the article is written again.
    let replies = vec![
        good_reply(&persona, 11, 0, 2),
        reading_reply(&persona, 11, 0, 2, true),
        good_reply(&persona, 11, 0, 2),
        reading_reply(&persona, 11, 0, 2, false),
    ];
    let both = writer(replies);
    let cache = cache("reader");
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &both,
        Some(&both),
        &cache,
        &options(1),
    )
    .unwrap();
    assert_eq!((summary.written, summary.rejected), (1, 1));
    assert_eq!(summary.calls, 4, "two writings and two readings");
    let rejected = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    assert!(rejected.contains("decision"), "{rejected}");
}
