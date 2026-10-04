// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in testing model pipelines without a model, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What the specifications share: a document builder, distinct words, and a
//! helper model that answers from a script.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};

use crate::curate::Document;
use crate::document::{Authorship, Date, Period};
use crate::helper::Helper;

/// A document of his own, dated `year`, written to `recipient`.
pub fn doc(id: &str, year: u16, recipient: &str, body: &str) -> Document {
    Document {
        schema: crate::curate::SCHEMA,
        id: id.into(),
        source_id: "cushing-1".into(),
        heading: format!("TO {}.", recipient.to_uppercase()),
        recipient: Some(recipient.into()),
        note: "[MS.]".into(),
        date: Date {
            year,
            month: None,
            day: None,
        },
        period: Period::of(year),
        authorship: Authorship::DraftInHand,
        authorship_confidence: 0.8,
        authorship_basis: "test".into(),
        temporal_holdout: false,
        body: body.into(),
    }
}

/// `n` distinct words that start with `prefix`.
pub fn words(prefix: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{prefix}{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Chooses a reply by what the model was asked.
type Responder = Box<dyn Fn(&str) -> String + Send + Sync>;

/// A model that answers each request with the next reply of its script, the
/// last one repeating, and keeps what it was sent.
pub struct Scripted {
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<String>>,
    /// When set, the reply is chosen by what was asked, not by order.
    responder: Option<Responder>,
}

impl Scripted {
    /// Everything the model was sent, joined.
    pub fn sent(&self) -> String {
        self.seen.lock().unwrap().join(" ")
    }

    /// How many requests the model has had.
    pub fn requests(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "scripted-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let asked = format!("{:?}", req.messages);
        self.seen.lock().unwrap().push(asked.clone());
        let reply = match &self.responder {
            Some(responder) => responder(&asked),
            None => {
                let mut replies = self.replies.lock().unwrap();
                if replies.len() > 1 {
                    replies.pop_front()
                } else {
                    replies.front().cloned()
                }
                .unwrap_or_default()
            }
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A helper that replies from `replies`, and the model behind it.
pub fn helper(replies: &[&str]) -> (Helper, Arc<Scripted>) {
    let provider = Arc::new(Scripted {
        replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
        seen: Mutex::new(Vec::new()),
        responder: None,
    });
    (
        Helper::with_model(Model::new(provider.clone(), "scripted")),
        provider,
    )
}

/// A helper whose reply is `reply(what was asked)`.
pub fn responding(
    reply: impl Fn(&str) -> String + Send + Sync + 'static,
) -> (Helper, Arc<Scripted>) {
    let provider = Arc::new(Scripted {
        replies: Mutex::new(VecDeque::new()),
        seen: Mutex::new(Vec::new()),
        responder: Some(Box::new(reply)),
    });
    (
        Helper::with_model(Model::new(provider.clone(), "scripted")),
        provider,
    )
}
