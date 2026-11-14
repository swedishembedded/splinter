// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: on the served path every attempt is its own request to the model of
//! the policy asked for, the request is the family's conversation and nothing
//! else, the reply is graded by the family, and a policy already examined is
//! not examined again.

#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_agent::solve::Model;
use splinter_pipelines::campaign::brain::{Graded, ModelSettings, TaskFamily};
use splinter_pipelines::campaign::served::{
    CallFailure, PolicyModels, ServedCollector, ServedValidator,
};
use splinter_pipelines::campaign::{
    CampaignError, CollectRequest, PolicyRef, RolloutCollector, Validator,
};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

struct Says {
    text: String,
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Says {
    fn name(&self) -> &str {
        "says"
    }
    fn model_name(&self) -> &str {
        "says"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push(request);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.text.clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Host {
    provider: Arc<Says>,
    asked: AtomicUsize,
    greedy: Mutex<Vec<bool>>,
}

impl PolicyModels for Host {
    fn model(&self, _: &PolicyRef, greedy: bool) -> Result<Model, CampaignError> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        self.greedy.lock().unwrap().push(greedy);
        Ok(Model::new(self.provider.clone(), "says"))
    }
}

struct Family;

impl TaskFamily for Family {
    fn describe(&self) -> Value {
        json!({"scripted": true})
    }
    fn pool(&self) -> Vec<String> {
        vec!["a".into(), "b".into()]
    }
    fn dev(&self) -> Vec<String> {
        vec!["d".into()]
    }
    fn retention(&self) -> Vec<String> {
        vec!["k".into()]
    }
    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        Ok(vec![("user".into(), format!("what is {task}?"))])
    }
    fn grade(&self, _: &str, completion: &str) -> Result<Graded, CampaignError> {
        let passed = completion == "7";
        Ok(Graded {
            reward: f64::from(u8::from(passed)),
            passed,
            feedback: None,
            privileged: None,
        })
    }
}

fn host(text: &str) -> Host {
    Host {
        provider: Arc::new(Says {
            text: text.into(),
            seen: Mutex::new(Vec::new()),
        }),
        asked: AtomicUsize::new(0),
        greedy: Mutex::new(Vec::new()),
    }
}

fn settings() -> ModelSettings {
    ModelSettings::new(std::path::Path::new("base"))
}

#[test]
fn every_attempt_is_a_request_with_the_conversation_alone_and_is_graded_by_the_family() {
    let host = host("7");
    let mut collector = ServedCollector::new(&Family, &host, settings()).unwrap();
    let groups = collector
        .collect(&CollectRequest {
            tasks: vec!["a".into(), "b".into()],
            attempts: 3,
            policy: PolicyRef::default(),
            // seed: a served engine draws its own; the request carries it for the record only.
            seed: 1,
        })
        .unwrap();
    assert_eq!(groups.len(), 2);
    assert!(groups
        .iter()
        .all(|g| g.rollouts.len() == 3 && g.rollouts.iter().all(|r| r.passed)));
    let seen = host.provider.seen.lock().unwrap();
    assert_eq!(seen.len(), 6, "one request per attempt");
    assert!(
        seen.iter().all(|r| r.messages.len() == 1),
        "no system prompt is added"
    );
    assert_eq!(
        *host.greedy.lock().unwrap(),
        vec![false],
        "training samples, it is not greedy"
    );
}

#[test]
fn a_policy_already_examined_is_not_examined_again() {
    let host = host("8");
    let mut validator = ServedValidator::new(&Family, &host, settings()).unwrap();
    let policy = PolicyRef::default();
    let first = validator.validate(&policy).unwrap();
    assert_eq!(first.dev.len(), 1);
    assert_eq!(first.retention.len(), 1);
    assert!(!first.dev["d"]);
    let again = validator.validate(&policy).unwrap();
    assert_eq!(first, again);
    assert_eq!(
        host.asked.load(Ordering::SeqCst),
        1,
        "the second look is kept"
    );
    assert_eq!(
        *host.greedy.lock().unwrap(),
        vec![true],
        "the exam is greedy"
    );
}

#[test]
fn a_policy_answers_any_tasks_greedily_with_the_completion_and_the_verdict() {
    let host = host("7");
    let validator = ServedValidator::new(&Family, &host, settings()).unwrap();
    let answers = validator
        .answer(&PolicyRef::default(), &["a".to_string(), "b".to_string()])
        .unwrap();
    assert_eq!(
        answers.iter().map(|a| a.task.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "answers come back in the order asked"
    );
    assert!(answers.iter().all(|a| a.reply.text == "7" && a.passed));
    assert_eq!(*host.greedy.lock().unwrap(), vec![true]);
    assert_eq!(host.provider.seen.lock().unwrap().len(), 2);
}

/// Says "7" to task a, fails the request for task b and is cut by the output
/// limit on task c.
struct Mixed {
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Mixed {
    fn name(&self) -> &str {
        "mixed"
    }
    fn model_name(&self) -> &str {
        "mixed"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let asked = request.messages[0]
            .as_text()
            .unwrap_or_default()
            .to_string();
        self.seen.lock().unwrap().push(request);
        let mut events = vec![Ok(ResponseEvent::TextDelta("7".into()))];
        match asked.as_str() {
            "what is b?" => anyhow::bail!("HTTP 500: the engine failed"),
            "what is c?" => events.push(Ok(ResponseEvent::MaxTokens)),
            _ => {}
        }
        events.push(Ok(ResponseEvent::Done));
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

struct MixedHost(Arc<Mixed>);

impl PolicyModels for MixedHost {
    fn model(&self, _: &PolicyRef, _: bool) -> Result<Model, CampaignError> {
        Ok(Model::new(self.0.clone(), "mixed"))
    }
}

#[test]
fn a_failed_call_is_its_own_outcome_a_cut_reply_says_so_and_a_traced_exam_asks_for_the_trace() {
    let provider = Arc::new(Mixed {
        seen: Mutex::new(Vec::new()),
    });
    let host = MixedHost(provider.clone());
    let validator = ServedValidator::new(&Family, &host, settings())
        .unwrap()
        .traced();
    let model = host.model(&PolicyRef::default(), true).unwrap();
    let tasks = ["a", "b", "c"].map(String::from);
    let attempts = validator.attempt_with(&model, &tasks).unwrap();
    let a = attempts[0].outcome.as_ref().unwrap();
    assert!(a.passed && !a.reply.truncated);
    assert!(
        matches!(&attempts[1].outcome, Err(CallFailure::Server(why)) if why.contains("500")),
        "{:?}",
        attempts[1].outcome
    );
    let c = attempts[2].outcome.as_ref().unwrap();
    assert!(c.reply.truncated, "the output limit cut task c");
    assert!(
        provider
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.want_logprobs),
        "a traced exam asks every reply's trace"
    );
    let refused = validator.answer_with(&model, &tasks).unwrap_err();
    assert!(refused.to_string().contains("task b"), "{refused}");
}
