// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a model's proposed edit to a harness document counts only when the
//! document it makes compiles under sven's compiler. A reply that breaks the
//! document is sent back with the compiler's reason, and one that is never
//! fixed is an error, not a candidate.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::harness_edit::{propose_edit, EditRequest};
use splinter_agent::solve::Model;
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};
use sven_sdk::CallError;

const DOC: &str =
    "api_version: \"1\"\nname: h\nentry: done\nsteps:\n  done:\n    return:\n      value: 1\n";

/// Answers each request with the next reply, keeping what it was sent.
struct Replies {
    replies: Vec<String>,
    next: AtomicUsize,
    seen: Mutex<Vec<String>>,
}

impl Replies {
    fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: replies.iter().map(ToString::to_string).collect(),
            next: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Replies {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "editor-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = req
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.seen.lock().unwrap().push(text);
        let at = self
            .next
            .fetch_add(1, Ordering::SeqCst)
            .min(self.replies.len() - 1);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.replies[at].clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn request() -> EditRequest<'static> {
    EditRequest {
        document: DOC,
        feedback: "regressed: parsing",
        deadline: Duration::from_secs(30),
        repairs: 2,
        cancel: None,
    }
}

const BREAKS: &str = r#"{"rationale": "r", "edits": [{"path": ["entry"], "value": "nowhere"}]}"#;
const MENDS: &str =
    r#"{"rationale": "better", "edits": [{"path": ["description"], "value": "tuned"}]}"#;

#[tokio::test]
async fn a_reply_that_breaks_the_document_is_sent_back_with_the_compilers_reason() {
    let model = Replies::new(&[BREAKS, MENDS]);
    let edited = propose_edit(&Model::new(model.clone(), "scripted/editor-1"), request())
        .await
        .unwrap();

    assert!(
        edited.document.contains("description: tuned"),
        "{}",
        edited.document
    );
    assert_eq!(edited.rationale, "better");
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one correction");
    assert!(
        seen[1].contains("does not compile"),
        "the compiler's reason went back: {}",
        seen[1]
    );
    assert!(
        seen[0].contains("regressed: parsing"),
        "the feedback reached the model"
    );
}

#[tokio::test]
async fn edits_that_are_never_fixed_are_an_error_not_a_candidate() {
    let model = Replies::new(&[BREAKS]);
    let result = propose_edit(&Model::new(model, "scripted/editor-1"), request()).await;
    assert!(
        matches!(result, Err(CallError::Postcondition { .. })),
        "{result:?}"
    );
}
