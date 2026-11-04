// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn what a person teaches
// them in conversation, for its clients. If your team needs expertise in
// continual learning from user feedback, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: an extractor model reads a session and proposes claims, each with
//! the quotes that support it; what it is shown never holds a secret; a
//! reply that is malformed or breaks the claim limits goes through the typed
//! call's own correction, and one that stays unusable is declined with why,
//! never turned into claims.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_agent::claims::{ClaimExtractor, Extraction};
use splinter_agent::solve::Model;
use splinter_core::claim::{ClaimKind, ClaimSubject};
use splinter_core::clock::FixedClock;
use splinter_knowledge::capture::capture_session;
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_knowledge::claims::MAX_STATEMENT_CHARS;
use splinter_knowledge::session::SessionView;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

const API_KEY: &str = "sk-proj-4f9a8c7b2e1d3f6a5b4c3d2e1f0a9b8c";

/// A model that answers each request with the next reply of its script, and
/// keeps the requests it was sent.
struct Scripted {
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    fn new(replies: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<String> {
        self.seen
            .lock()
            .map(|seen| seen.iter().map(|r| format!("{:?}", r.messages)).collect())
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "extractor-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .push(req);
        let reply = self
            .replies
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("the script has no reply left"))?;
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn session() -> anyhow::Result<SessionView> {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.steps = vec![
        TraceStep::new(
            1,
            StepOrigin::User,
            "Which port does the Tessera dashboard use?",
        ),
        TraceStep::new(2, StepOrigin::Agent, "It uses port 8080."),
        TraceStep::new(
            3,
            StepOrigin::User,
            format!("Wrong, it is 9090. By the way my key is {API_KEY}, do not repeat it."),
        ),
        TraceStep::new(4, StepOrigin::Agent, "Understood."),
    ];
    let captured = capture_session(
        &serde_json::to_vec(&t)?,
        &FixedClock::new("2026-10-01T08:00:00.000Z"),
    )?;
    Ok(SessionView::of(&captured)?)
}

fn claim(statement: &str) -> serde_json::Value {
    json!({
        "kind": "correction",
        "statement": statement,
        "question": "Which port does the Tessera dashboard use?",
        "quotes": [{"step": 1, "quote": "the Tessera dashboard"}, {"step": 3, "quote": "it is 9090"}],
        "said_wrong": "port 8080",
        "subject": "world",
    })
}

fn reply(claims: Vec<serde_json::Value>) -> String {
    json!({ "claims": claims }).to_string()
}

fn extractor(model: &Arc<Scripted>) -> ClaimExtractor {
    ClaimExtractor::new(Model::new(model.clone(), "scripted/extractor-1")).with_policy(
        ExtractionPolicy {
            repairs: 1,
            ..ExtractionPolicy::default()
        },
    )
}

#[tokio::test]
async fn the_model_proposes_claims_with_their_quotes_from_a_session_without_secrets(
) -> anyhow::Result<()> {
    let model = Scripted::new(vec![reply(vec![claim(
        "The Tessera dashboard uses port 9090.",
    )])]);
    let Extraction::Proposals(proposals) = extractor(&model).extract(&session()?).await? else {
        anyhow::bail!("declined");
    };
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];
    assert_eq!(p.kind, ClaimKind::Correction);
    assert_eq!(p.statement, "The Tessera dashboard uses port 9090.");
    assert_eq!(
        p.quotes
            .iter()
            .map(|q| (q.step, q.text.as_str()))
            .collect::<Vec<_>>(),
        [(1, "the Tessera dashboard"), (3, "it is 9090")]
    );
    assert_eq!(p.said_wrong.as_deref(), Some("port 8080"));

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].contains("Wrong, it is 9090"),
        "the model is shown the session"
    );
    assert!(
        !requests[0].contains(API_KEY),
        "no secret reaches the model"
    );
    Ok(())
}

#[tokio::test]
async fn a_malformed_or_over_long_reply_is_sent_back_for_correction() -> anyhow::Result<()> {
    let too_long = "x".repeat(MAX_STATEMENT_CHARS + 1);
    let model = Scripted::new(vec![
        "Sure! Here are the claims.".into(),
        reply(vec![claim(&too_long)]),
        reply(vec![claim("The Tessera dashboard uses port 9090.")]),
    ]);
    let extractor = ClaimExtractor::new(Model::new(model.clone(), "scripted/extractor-1"))
        .with_policy(ExtractionPolicy {
            repairs: 2,
            ..ExtractionPolicy::default()
        });
    let Extraction::Proposals(proposals) = extractor.extract(&session()?).await? else {
        anyhow::bail!("declined");
    };
    assert_eq!(proposals.len(), 1);
    assert_eq!(
        model.requests().len(),
        3,
        "two corrections, then a usable reply"
    );
    Ok(())
}

#[tokio::test]
async fn a_claim_without_its_subject_is_sent_back_for_correction() -> anyhow::Result<()> {
    let mut unsure = claim("The Tessera dashboard uses port 9090.");
    unsure.as_object_mut().map(|m| m.remove("subject"));
    let model = Scripted::new(vec![
        reply(vec![unsure]),
        reply(vec![claim("The Tessera dashboard uses port 9090.")]),
    ]);
    let Extraction::Proposals(proposals) = extractor(&model).extract(&session()?).await? else {
        anyhow::bail!("declined");
    };
    assert_eq!(proposals[0].subject, Some(ClaimSubject::World));
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "one correction");
    assert!(
        requests[1].contains("subject"),
        "the correction names the field"
    );
    Ok(())
}

#[tokio::test]
async fn a_reply_that_stays_unusable_is_declined_with_why() -> anyhow::Result<()> {
    let model = Scripted::new(vec!["not json".into(), "{\"claims\": [".into()]);
    let declined = extractor(&model).extract(&session()?).await?;
    let Extraction::Declined(why) = declined else {
        anyhow::bail!("{declined:?}");
    };
    assert!(why.contains("attempt"), "{why}");
    Ok(())
}
