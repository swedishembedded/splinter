// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: reading the decisions out of a corpus resumes where it stopped, case
//! by case, without reading twice what it has; and judging the saved drafts
//! again fails on a log it cannot place rather than guessing an article.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::cache::{Cache, StoredArticle};
use synthetic_author::lab::{mine_all, read_mined, MinedResult};
use synthetic_author::persona::Persona;
use synthetic_author::recheck::recheck;
use synthetic_author::writer::plan;

/// A reader that gives the same reply to every request and counts them.
struct Reader {
    reply: String,
    requests: Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Reader {
    fn name(&self) -> &str {
        "reader"
    }
    fn model_name(&self) -> &str {
        "reader-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        *self.requests.lock().unwrap() += 1;
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.reply.clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn case_reading(case: &str) -> serde_json::Value {
    serde_json::json!({
        "case": case,
        "briefing": "The steward faced a matter that the charter did not clearly settle, \
                     with a little time to spare, and a community that would live with the \
                     outcome for some seasons to come.",
        "offer_made": false,
        "chose": "hold_and_wait",
        "reasons": [],
        "open": ["act_alone", "hold_and_wait"],
    })
}

fn cache(test: &str) -> Cache {
    let dir = std::env::temp_dir().join(format!("synthetic-author-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Cache::open(dir).unwrap()
}

#[test]
fn an_article_with_cases_unread_is_read_again_and_only_those_cases_are_appended() {
    let cache = cache("mine-resume");
    cache
        .write_article(&StoredArticle {
            index: 0,
            topic: 0,
            title: "t".into(),
            body: "Alpha was settled by waiting.\n\nBeta was settled by waiting.".into(),
            cases: vec!["Alpha".into(), "Beta".into()],
            attempts: 1,
        })
        .unwrap();
    let provider = Arc::new(Reader {
        reply: serde_json::json!({"cases": [case_reading("Alpha"), case_reading("Beta")]})
            .to_string(),
        requests: Mutex::new(0),
    });
    let model = BlockingModel::new(Model::new(provider.clone(), "reader")).unwrap();

    // A run that stopped after the first case of the article.
    let first = serde_json::json!({
        "article": 0, "case": "Alpha", "refused": "stopped here",
    });
    std::fs::write(cache.dir().join("decisions.jsonl"), format!("{first}\n")).unwrap();

    assert_eq!(mine_all(&cache, &model, 1).unwrap(), 1);
    let read = read_mined(&cache).unwrap();
    let cases: Vec<&str> = read.iter().map(|m| m.case.as_str()).collect();
    assert_eq!(cases, ["Alpha", "Beta"], "Alpha is not appended twice");
    assert!(matches!(read[1].result, MinedResult::Kept(_)));

    assert_eq!(mine_all(&cache, &model, 1).unwrap(), 0);
    assert_eq!(
        *provider.requests.lock().unwrap(),
        1,
        "nothing is read twice"
    );
}

#[test]
fn a_saved_draft_whose_article_cannot_be_placed_fails_the_recheck_naming_the_line() {
    let persona = Persona::generate(3);
    let planned = plan(&persona, 3, 2, 2);
    let draft = serde_json::json!({"title": "t", "body": "Too short."});
    let placed = serde_json::json!({"article": 1, "attempt": 1, "reason": "r", "draft": draft});
    let unplaced = serde_json::json!({"attempt": 2, "reason": "r", "draft": draft});

    let rechecked = recheck(&format!("{placed}\n"), &persona, &planned, 100).unwrap();
    assert_eq!(rechecked.len(), 1);
    assert_eq!(rechecked[0].article, 1);
    assert!(rechecked[0].now.is_err());

    let error = recheck(&format!("{placed}\n{unplaced}\n"), &persona, &planned, 100).unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");

    let far = serde_json::json!({"article": 9, "attempt": 1, "reason": "r", "draft": draft});
    let error = recheck(&format!("{far}\n"), &persona, &planned, 100).unwrap_err();
    assert!(format!("{error:#}").contains("line 1"), "{error:#}");
}

#[test]
fn each_article_is_read_by_a_model_made_for_it_with_a_seed_of_its_own() {
    use synthetic_author::writer::{Lent, Models, RequestKey, Role};

    /// Hands out the same reader for every request and keeps the requests.
    struct Recording {
        model: BlockingModel,
        keys: Mutex<Vec<RequestKey>>,
    }
    impl Models for Recording {
        fn model(&self, request: RequestKey) -> anyhow::Result<Lent<'_>> {
            self.keys.lock().unwrap().push(request);
            Ok(Lent::Borrowed(&self.model))
        }
    }

    let cache = cache("mine-keys");
    for index in 0..2 {
        cache
            .write_article(&StoredArticle {
                index,
                topic: 0,
                title: "t".into(),
                body: "Alpha was settled by waiting.".into(),
                cases: vec!["Alpha".into()],
                attempts: 1,
            })
            .unwrap();
    }
    let provider = Arc::new(Reader {
        reply: serde_json::json!({"cases": [case_reading("Alpha")]}).to_string(),
        requests: Mutex::new(0),
    });
    let models = Recording {
        model: BlockingModel::new(Model::new(provider, "reader")).unwrap(),
        keys: Mutex::new(Vec::new()),
    };
    assert_eq!(mine_all(&cache, &models, 1).unwrap(), 2);
    let mut keys = models.keys.into_inner().unwrap();
    keys.sort_by_key(|k| k.article);
    assert_eq!(
        keys.iter().map(|k| (k.article, k.role)).collect::<Vec<_>>(),
        [(0, Role::Miner), (1, Role::Miner)]
    );
    let run_seed = 5; // seed: only that the three roles differ for one request matters
    let reader = RequestKey {
        role: Role::Reader,
        ..keys[0]
    };
    let writer = RequestKey {
        role: Role::Writer,
        ..keys[0]
    };
    let seeds = [
        keys[0].seed(run_seed),
        keys[1].seed(run_seed),
        reader.seed(run_seed),
        writer.seed(run_seed),
    ];
    assert_eq!(
        seeds
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4,
        "no two requests share a seed: {seeds:?}"
    );
}

#[test]
fn how_the_decisions_were_read_is_recorded_and_a_later_mine_continues_with_it() {
    use synthetic_author::manifest::begin_or_continue_mining;
    let dir = cache("mine-record").dir().to_path_buf();
    let first = begin_or_continue_mining(&dir, "local:reader", None, true).unwrap();
    let seed = first.run_seed.expect("a reader that runs here is seeded");
    assert!(dir.join("mine.json").exists());
    assert_eq!(
        begin_or_continue_mining(&dir, "local:reader", None, true).unwrap(),
        first,
        "a later mine reads the seed back"
    );
    assert_eq!(
        begin_or_continue_mining(&dir, "local:reader", Some(seed), true).unwrap(),
        first
    );
    let other_seed = format!(
        "{:#}",
        begin_or_continue_mining(&dir, "local:reader", Some(seed ^ 1), true).unwrap_err()
    );
    assert!(other_seed.contains("mine.json"), "{other_seed}");
    let other_reader = format!(
        "{:#}",
        begin_or_continue_mining(&dir, "local:another", None, true).unwrap_err()
    );
    assert!(other_reader.contains("local:reader"), "{other_reader}");

    let remote = cache("mine-record-remote").dir().to_path_buf();
    let record = begin_or_continue_mining(&remote, "remote:model", None, false).unwrap();
    assert_eq!(record.run_seed, None, "a remote reader is not seeded");
}
