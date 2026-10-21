// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: what a busy server does to a run is measured at the model, apart from
//! what the agent did: requests, the wait for the first stream item (the
//! longest and in all) and the time the streams took.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::Scripted;
use futures::StreamExt;
use splinter_agent::timed::{Stats, Timed};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseStream};

/// Waits before it answers, as a busy server does.
struct Slow(Arc<dyn ModelProvider>, Duration);

#[async_trait::async_trait]
impl ModelProvider for Slow {
    fn name(&self) -> &str {
        "slow"
    }
    fn model_name(&self) -> &str {
        "slow"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        tokio::time::sleep(self.1).await;
        self.0.complete(req).await
    }
}

async fn ask(model: &Arc<dyn ModelProvider>) {
    let mut stream = model.complete(CompletionRequest::default()).await.unwrap();
    while stream.next().await.is_some() {}
}

#[tokio::test]
async fn the_wait_for_the_server_is_measured_per_request() {
    let stats = Arc::new(Mutex::new(Stats::default()));
    let slow = Arc::new(Slow(
        Scripted::new(vec!["a".into(), "b".into()]),
        Duration::from_millis(200),
    ));
    let timed = Timed::wrap(slow, Arc::clone(&stats));
    ask(&timed).await;
    ask(&timed).await;
    let seen = stats.lock().unwrap().clone();
    assert_eq!(seen.requests, 2);
    assert!(seen.max_wait_seconds >= 0.2, "{seen:?}");
    assert!(seen.wait_seconds >= 0.4, "{seen:?}");
    assert!(seen.model_seconds >= seen.wait_seconds - 1e-9, "{seen:?}");
}

#[tokio::test]
async fn a_wrapped_model_keeps_its_name_and_a_fresh_stats_counts_from_zero() {
    let stats = Arc::new(Mutex::new(Stats::default()));
    let timed = Timed::wrap(Scripted::new(vec![]), Arc::clone(&stats));
    assert_eq!(timed.name(), "scripted");
    assert_eq!(timed.model_name(), "generator-1");
    assert_eq!(*stats.lock().unwrap(), Stats::default());
}
