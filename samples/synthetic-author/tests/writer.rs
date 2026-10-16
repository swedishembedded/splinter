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
//! the repository, from which an author can be drawn again. A request that
//! fails in transit is not a verdict on the article: it is retried without
//! using up an attempt, logged under its own kind, kept out of what the writer
//! is told was wrong, and reported apart when it ends the run.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::cache::{Cache, Kind};
use synthetic_author::episode::episodes_for;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;
use synthetic_author::writer::{write_corpus, Models, Options, TRAIN_TOPICS};

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
        workers: 1,
        transport_retries: 2,
        transport_backoff: Duration::ZERO,
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
            let f = &e.scenario.features;
            serde_json::json!({
                "case": e.case.name, "chose": snake, "offer_made": f.outsider_offers_help,
                "open": e.scenario.options,
                "briefing": "x", "reasons": [],
                "facts": {
                    "exceeds_authority": f.exceeds_authority, "urgent": f.urgent,
                    "binds_future": f.binds_future, "irreversible": f.irreversible,
                    "members_harmed": f.members_harmed,
                }
            })
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
        Some(&both as &dyn Models),
        &cache,
        &options(1),
    )
    .unwrap();
    assert_eq!((summary.written, summary.rejected), (1, 1));
    assert_eq!(summary.calls, 4, "two writings and two readings");
    let rejected = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    assert!(rejected.contains("decision"), "{rejected}");
}

/// What a reader reports of a case: the decision, the facts and the open ways.
type Told = (
    Archetype,
    synthetic_author::scenario::Features,
    Vec<Archetype>,
);

/// A model that answers by what it is asked: a writer tells every case it is
/// shown as the beliefs decide it, and a reader reports it.
struct Responding {
    chose: std::collections::BTreeMap<String, Told>,
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
                    let name = match a {
                        Archetype::ActAlone => "act_alone",
                        Archetype::ConsultThenAct => "consult_then_act",
                        Archetype::DeferToOutsider => "defer_to_outsider",
                        Archetype::HoldAndWait => "hold_and_wait",
                    };
                    serde_json::json!({
                        "case": case, "chose": name, "offer_made": f.outsider_offers_help,
                        "open": open,
                        "briefing": "x", "reasons": [],
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

#[test]
fn workers_write_every_article_once_and_keep_to_the_call_budget() {
    let persona = Persona::generate(11);
    let planned = synthetic_author::writer::plan(&persona, 11, 12, 2);
    let chose = planned
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
    let model =
        BlockingModel::new(Model::new(Arc::new(Responding { chose }), "responding")).unwrap();
    let cache = cache("workers");
    let mut opts = options(12);
    opts.workers = 4;
    opts.max_calls = 30;
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &model,
        Some(&model as &dyn Models),
        &cache,
        &opts,
    )
    .unwrap();
    assert_eq!(summary.written, 12, "{summary:?}");
    assert_eq!(summary.calls, 24, "a writing and a reading for each");
    let kept = cache.articles().unwrap();
    let indexes: Vec<_> = kept.iter().map(|a| a.index).collect();
    assert_eq!(indexes, (0..12).collect::<Vec<_>>());

    // A budget below what the work needs stops the run at the budget exactly.
    let cache = self::cache("workers-budget");
    let mut opts = options(12);
    opts.workers = 4;
    opts.max_calls = 7;
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &model,
        Some(&model as &dyn Models),
        &cache,
        &opts,
    )
    .unwrap();
    assert!(summary.calls <= 7, "{summary:?}");
    assert!(summary.stopped.unwrap().contains("calls"));
}

#[test]
fn many_threads_noting_refusals_leave_a_log_every_line_of_which_reads() {
    let cache = std::sync::Arc::new(cache("log"));
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let draft = synthetic_author::gates::Article {
                    title: "x".into(),
                    body: "word ".repeat(4000),
                };
                for n in 0..25 {
                    cache
                        .note_rejected(t, n, Kind::Content, "a reason", Some(&draft))
                        .unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let text = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 200);
    for line in text.lines() {
        let _: serde_json::Value = serde_json::from_str(line).unwrap();
    }
}

/// A model whose requests follow a script: `Some(reply)` answers, `None` fails
/// in transit. Every request it is shown is kept.
struct Flaky {
    script: Mutex<VecDeque<Option<String>>>,
    seen: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl ModelProvider for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }
    fn model_name(&self) -> &str {
        "flaky-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen
            .lock()
            .unwrap()
            .push(format!("{:?}", req.messages));
        let Some(reply) = self.script.lock().unwrap().pop_front().flatten() else {
            anyhow::bail!("connection refused");
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn flaky(script: Vec<Option<String>>) -> (BlockingModel, Arc<Flaky>) {
    let provider = Arc::new(Flaky {
        script: Mutex::new(script.into()),
        seen: Mutex::new(Vec::new()),
    });
    let model = BlockingModel::new(Model::new(provider.clone(), "flaky")).unwrap();
    (model, provider)
}

fn kinds(cache: &Cache) -> Vec<String> {
    let text = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    text.lines()
        .map(|l| {
            let row: serde_json::Value = serde_json::from_str(l).unwrap();
            row["kind"].as_str().unwrap().to_string()
        })
        .collect()
}

#[test]
fn a_request_lost_in_transit_is_retried_without_using_an_attempt_or_blaming_the_article() {
    let persona = Persona::generate(11);
    let short = serde_json::json!({"title": "x", "body": "Far too short."}).to_string();
    // Two attempts: a refused article, a lost request, then the article. Were
    // the loss to use an attempt, the second attempt would be the lost one.
    let (model, provider) = flaky(vec![
        Some(short),
        None,
        Some(good_reply(&persona, 11, 0, 2)),
    ]);
    let cache = cache("transit");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(1)).unwrap();
    assert_eq!(
        (summary.written, summary.abandoned, summary.unavailable),
        (1, 0, 0)
    );
    assert_eq!(
        summary.rejected, 1,
        "only the refusal counts as turned away"
    );
    assert_eq!(kinds(&cache), ["content", "writer-unavailable"]);
    let seen = provider.seen.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert!(seen[2].contains("words"), "the refusal is still told");
    assert!(!seen[2].contains("connection refused"), "{}", seen[2]);
}

#[test]
fn a_writer_that_stays_unreachable_stops_the_run_and_its_article_is_reported_apart() {
    let persona = Persona::generate(11);
    let (model, _) = flaky(vec![]);
    let cache = cache("unreachable");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(3)).unwrap();
    assert_eq!(summary.calls, 3, "one request and two retries");
    assert_eq!(
        (summary.written, summary.abandoned, summary.unavailable),
        (0, 0, 1)
    );
    let stopped = summary.stopped.unwrap();
    assert!(
        stopped.contains("writer") && stopped.contains("connection refused"),
        "{stopped}"
    );
    assert_eq!(kinds(&cache), ["writer-unavailable"; 3]);
}

#[test]
fn a_reader_that_stays_unreachable_leaves_the_draft_unjudged_and_a_spent_budget_is_logged_as_such()
{
    let persona = Persona::generate(11);
    let reply = good_reply(&persona, 11, 0, 2);
    let (model, _) = flaky(vec![Some(reply.clone())]);
    let cache = cache("reader-unreachable");
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &model,
        Some(&model as &dyn Models),
        &cache,
        &options(1),
    )
    .unwrap();
    assert_eq!(
        (summary.abandoned, summary.unavailable, summary.rejected),
        (0, 1, 0)
    );
    assert_eq!(kinds(&cache), ["reader-unavailable"; 3]);
    let first: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(cache.dir().join("rejected.jsonl"))
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert!(
        first["draft"]["body"].is_string(),
        "the draft is kept for judging later"
    );

    let (model, _) = flaky(vec![Some(reply)]);
    let cache = self::cache("reader-budget");
    let mut opts = options(1);
    opts.max_calls = 1;
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &model,
        Some(&model as &dyn Models),
        &cache,
        &opts,
    )
    .unwrap();
    assert_eq!(
        (summary.abandoned, summary.unavailable, summary.written),
        (0, 0, 0)
    );
    assert_eq!(kinds(&cache), ["budget"]);
}

#[test]
fn an_unreadable_usage_record_fails_the_run_before_anything_is_spent() {
    let persona = Persona::generate(11);
    let (model, provider) = flaky(vec![Some(good_reply(&persona, 11, 0, 2))]);
    let cache = cache("usage");
    std::fs::write(cache.dir().join("usage.json"), "{\"calls\": ").unwrap();
    let error = write_corpus(&persona, 11, &model, &cache, &options(1)).unwrap_err();
    assert!(format!("{error:#}").contains("usage.json"), "{error:#}");
    assert!(provider.seen.lock().unwrap().is_empty());
}

#[test]
fn a_reader_answer_in_no_readable_shape_turns_the_draft_away_as_content() {
    let persona = Persona::generate(11);
    // The reader is corrected once, so it answers twice, both times unreadably.
    let (model, _) = flaky(vec![
        Some(good_reply(&persona, 11, 0, 2)),
        Some("not json".into()),
        Some("still not json".into()),
    ]);
    let cache = cache("reader-unreadable");
    let mut opts = options(1);
    opts.attempts_per_article = 1;
    let summary = synthetic_author::writer::write_corpus_read(
        &persona,
        11,
        &model,
        Some(&model as &dyn Models),
        &cache,
        &opts,
    )
    .unwrap();
    assert_eq!(
        (summary.rejected, summary.abandoned, summary.unavailable),
        (1, 1, 0)
    );
    assert_eq!(kinds(&cache), ["content"]);
    let text = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();
    assert!(text.contains("could not be read"), "{text}");
    assert!(
        !text.contains("does not show the decision"),
        "no decision is invented for it: {text}"
    );
}
