// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in running studies against hosted
// language models without losing a run to a rate limit, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a request a hosted model never answered because of the provider (a
//! rate limit, a server error, a timeout, a dropped connection) is lost in
//! transit: it is waited out and repeated without using an attempt at the
//! article, whoever asks, and when the bound is spent the run stops and says
//! so in one line. A request the model refused as malformed is the request's
//! fault and is not repeated. The tokens a run records are the ones the
//! provider counted, the reasoning a model does before it answers included.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod writer_support;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use splinter_sdk::agent::CallError;
use synthetic_author::cache::{Cache, StoredArticle};
use synthetic_author::lab::{mine_all, read_mined, MinedResult};
use synthetic_author::persona::Persona;
use synthetic_author::transit::{classify, one_line, Failure, Transit};
use synthetic_author::writer::write_corpus;
use writer_support::{cache, good_reply, options};

/// What the provider does with one request.
enum Step {
    /// Answers with this text, after reasoning aloud, and reports this many
    /// tokens written.
    Answer(String, u32),
    /// Refuses with this error.
    Fail(String),
}

struct Provider {
    script: Mutex<VecDeque<Step>>,
    requests: Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Provider {
    fn name(&self) -> &str {
        "hosted"
    }
    fn model_name(&self) -> &str {
        "hosted-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        *self.requests.lock().unwrap() += 1;
        match self.script.lock().unwrap().pop_front() {
            Some(Step::Fail(text)) => anyhow::bail!("{text}"),
            Some(Step::Answer(text, written)) => Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::ThinkingDelta(
                    r#"{"title": "thought", "body": "a draft, not the answer"}"#.into(),
                )),
                Ok(ResponseEvent::TextDelta(text)),
                Ok(ResponseEvent::Usage {
                    input_tokens: 1200,
                    output_tokens: written,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    cost_usd: None,
                }),
                Ok(ResponseEvent::Done),
            ]))),
            None => anyhow::bail!("the script has no step left"),
        }
    }
}

fn hosted(script: Vec<Step>) -> (BlockingModel, Arc<Provider>) {
    let provider = Arc::new(Provider {
        script: Mutex::new(script.into()),
        requests: Mutex::new(0),
    });
    let model = BlockingModel::new(Model::new(provider.clone(), "hosted")).unwrap();
    (model, provider)
}

/// What a hosted provider's shared-pool limiter answers, as the driver
/// reports it.
const RATE_LIMIT: &str = r#"model completion failed: openrouter error 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"some-model is temporarily rate-limited upstream. Please retry shortly","provider_name":"CoreWeave","remedy_hint":"add your own provider key"}},"user_id":"user_1"}"#;

/// A refusal by the fallback a router chose after its first provider limited
/// the request.
const REROUTED: &str = r#"model completion failed: openrouter error 400 Bad Request: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"model features structured outputs not support","provider_name":"Fallback","previous_errors":[{"code":429,"message":"Provider returned error","provider_name":"First","raw":"temporarily rate-limited upstream"}]}}}"#;

/// The same, with the upstream's answer a string of JSON inside the body.
const REROUTED_NESTED: &str = r#"openrouter error 400 Bad Request: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"code\":400, \"message\":\"model features structured outputs not support\"}","provider_name":"Fallback"}}}"#;

const BAD_REQUEST: &str = r#"model completion failed: openrouter error 400 Bad Request: {"error":{"message":"messages: text content blocks must be non-empty","code":400}}"#;

fn infrastructure(text: &str) -> CallError {
    CallError::Infrastructure(anyhow::anyhow!("{text}"))
}

fn rejected(cache: &Cache) -> String {
    std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap()
}

fn kinds(cache: &Cache) -> Vec<String> {
    rejected(cache)
        .lines()
        .map(|l| {
            let row: serde_json::Value = serde_json::from_str(l).unwrap();
            row["kind"].as_str().unwrap().to_string()
        })
        .collect()
}

#[test]
fn what_the_provider_failed_at_is_lost_in_transit_and_what_the_request_got_wrong_is_not() {
    for text in [
        RATE_LIMIT,
        REROUTED,
        "openrouter error 503 Service Unavailable: overloaded",
        "openrouter error 500 Internal Server Error: boom",
        "openrouter error 408 Request Timeout: slow",
        "openrouter request failed: error sending request: connection reset by peer",
        "stream idle for 120s",
    ] {
        assert!(
            matches!(classify(&infrastructure(text)), Failure::Lost { .. }),
            "{text}"
        );
    }
    for text in [
        BAD_REQUEST,
        "openrouter error 401 Unauthorized: bad key",
        "openrouter error 413 Payload Too Large: long",
        "openrouter error 422 Unprocessable Entity: no",
    ] {
        assert!(
            matches!(classify(&infrastructure(text)), Failure::Fault),
            "{text}"
        );
    }
    let stopped = CallError::Stopped {
        conclusion: splinter_sdk::agent::RunConclusion::Timeout,
    };
    assert!(matches!(classify(&stopped), Failure::Lost { .. }));
    let spent = CallError::Stopped {
        conclusion: splinter_sdk::agent::RunConclusion::BudgetExhausted,
    };
    assert!(matches!(classify(&spent), Failure::Fault));
}

#[test]
fn a_hint_in_the_error_is_honoured_and_without_one_the_wait_doubles_to_a_minute() {
    let hinted = |text: &str| match classify(&infrastructure(text)) {
        Failure::Lost { retry_after } => retry_after,
        Failure::Fault => panic!("{text}"),
    };
    assert_eq!(
        hinted("openrouter error 429 Too Many Requests: slow down, retry-after: 7"),
        Some(Duration::from_secs(7))
    );
    assert_eq!(
        hinted(r#"error 429: {"error":{"retry_after":2.5}}"#),
        Some(Duration::from_millis(2500))
    );
    assert_eq!(hinted(RATE_LIMIT), None, "'retry shortly' is no number");

    let transit = Transit::new(8, Duration::from_secs(5));
    let waits: Vec<u64> = (1..=9)
        .map(|n| transit.pause_before(n, None).as_secs())
        .collect();
    assert_eq!(waits, [5, 10, 20, 40, 60, 60, 60, 60, 60]);
    assert_eq!(
        transit.pause_before(1, Some(Duration::from_secs(7))),
        Duration::from_secs(7)
    );
    assert_eq!(
        transit.pause_before(1, Some(Duration::from_secs(900))),
        Duration::from_secs(60),
        "a hint is honoured up to the cap"
    );
    assert!(
        Transit::default().total_wait() >= Duration::from_secs(240),
        "the default bound waits a few minutes"
    );
}

#[test]
fn a_provider_error_is_told_in_one_line() {
    let line = one_line(&anyhow::anyhow!("{RATE_LIMIT}"));
    assert!(!line.contains('\n') && !line.contains('{'), "{line}");
    assert!(
        line.contains("429") && line.contains("CoreWeave") && line.contains("rate-limited"),
        "{line}"
    );
    assert!(line.len() < 260, "{line}");
    assert!(!line.contains("user_1"), "{line}");
    let routed = one_line(&anyhow::anyhow!("{REROUTED_NESTED}"));
    assert!(
        routed.contains("Fallback")
            && routed.contains("structured outputs")
            && !routed.contains('\\'),
        "{routed}"
    );
    let plain = one_line(&anyhow::anyhow!("connection refused"));
    assert_eq!(plain, "connection refused");
}

#[test]
fn a_rate_limited_request_is_waited_out_without_using_an_attempt_or_filling_the_log() {
    let persona = Persona::generate(11);
    let (model, provider) = hosted(vec![
        Step::Fail(RATE_LIMIT.into()),
        Step::Fail(RATE_LIMIT.into()),
        Step::Answer(good_reply(&persona, 11, 0, 2), 900),
    ]);
    let cache = cache("limited");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(1)).unwrap();
    assert_eq!(
        (summary.written, summary.abandoned, summary.unavailable),
        (1, 0, 0)
    );
    assert_eq!(summary.rejected, 0);
    assert_eq!(*provider.requests.lock().unwrap(), 3);
    assert_eq!(kinds(&cache), ["writer-unavailable"; 2]);
    assert!(
        !rejected(&cache).contains("remedy_hint"),
        "a loss that was recovered is logged in one line"
    );
}

#[test]
fn a_provider_that_keeps_limiting_stops_the_run_with_one_line_and_the_body_once() {
    let persona = Persona::generate(11);
    let (model, _) = hosted((0..3).map(|_| Step::Fail(RATE_LIMIT.into())).collect());
    let cache = cache("limited-out");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(3)).unwrap();
    assert_eq!(
        (summary.written, summary.abandoned, summary.unavailable),
        (0, 0, 1)
    );
    let stopped = summary.stopped.unwrap();
    assert!(
        stopped.contains("429") && stopped.contains("CoreWeave") && !stopped.contains('{'),
        "{stopped}"
    );
    assert_eq!(rejected(&cache).matches("remedy_hint").count(), 1);
}

#[test]
fn a_request_the_provider_calls_malformed_is_not_repeated_and_not_called_unreachable() {
    let persona = Persona::generate(11);
    let (model, provider) = hosted((0..4).map(|_| Step::Fail(BAD_REQUEST.into())).collect());
    let cache = cache("malformed");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(1)).unwrap();
    assert_eq!(
        (summary.abandoned, summary.unavailable, summary.rejected),
        (1, 0, 2)
    );
    assert_eq!(
        *provider.requests.lock().unwrap(),
        2,
        "one request for each of the two attempts"
    );
    assert_eq!(kinds(&cache), ["content"; 2]);
}

#[test]
fn the_article_is_read_from_the_answer_and_the_run_records_the_tokens_the_provider_counted() {
    let persona = Persona::generate(11);
    let (model, _) = hosted(vec![Step::Answer(good_reply(&persona, 11, 0, 2), 5400)]);
    let cache = cache("reasoning");
    let summary = write_corpus(&persona, 11, &model, &cache, &options(1)).unwrap();
    assert_eq!(summary.written, 1);
    assert_eq!(
        (summary.input_tokens, summary.output_tokens),
        (1200, 5400),
        "reasoning tokens are tokens written"
    );
    let article = &cache.articles().unwrap()[0];
    assert_eq!(article.title, "Season 0", "not the thinking's title");
    let usage = cache.usage().unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (1200, 5400));
}

#[test]
fn a_miner_that_is_rate_limited_is_waited_out_too() {
    let cache = cache("mine-limited");
    cache
        .write_article(&StoredArticle {
            index: 0,
            topic: 0,
            title: "t".into(),
            body: "Alpha was settled by waiting.".into(),
            cases: vec!["Alpha".into()],
            attempts: 1,
        })
        .unwrap();
    let reading = serde_json::json!({"cases": [{
        "case": "Alpha",
        "briefing": "The steward faced a matter that the charter did not clearly settle, \
                     with a little time to spare, and a community that would live with the \
                     outcome for some seasons to come.",
        "offer_made": false,
        "chose": "hold_and_wait",
        "reasons": [],
        "open": ["act_alone", "hold_and_wait"],
    }]});
    let (model, provider) = hosted(vec![
        Step::Fail(RATE_LIMIT.into()),
        Step::Answer(reading.to_string(), 300),
    ]);
    let transit = Transit::new(2, Duration::ZERO);
    assert_eq!(mine_all(&cache, &model, 1, &transit).unwrap(), 1);
    assert_eq!(*provider.requests.lock().unwrap(), 2);
    assert!(matches!(
        read_mined(&cache).unwrap()[0].result,
        MinedResult::Kept(_)
    ));

    let (model, _) = hosted(vec![Step::Fail(RATE_LIMIT.into())]);
    let cache = self::cache("mine-limited-out");
    cache
        .write_article(&StoredArticle {
            index: 0,
            topic: 0,
            title: "t".into(),
            body: "Alpha was settled by waiting.".into(),
            cases: vec!["Alpha".into()],
            attempts: 1,
        })
        .unwrap();
    let error = mine_all(&cache, &model, 1, &Transit::new(0, Duration::ZERO)).unwrap_err();
    let message = format!("{error:#}");
    assert!(
        message.contains("429") && message.contains("CoreWeave"),
        "{message}"
    );
}
