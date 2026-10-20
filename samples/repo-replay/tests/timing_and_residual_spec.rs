// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: what the server's latency did to an attempt is recorded apart from
//! what the agent did, and a file the commit removes that the agent left is
//! not called junk.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::ledger::{self, Outcome};
use splinter_repo_replay::plan_prompt::PlanStyle;
use splinter_repo_replay::scripted::{Behaviour, ScriptedAgent, Specificity};
use splinter_repo_replay::timed::{Stats, Timed};
use splinter_sdk::agent::sven::model::{CompletionRequest, ModelProvider};

/// Waits before it answers, as a busy server does.
struct Slow(Arc<dyn ModelProvider>);

#[async_trait::async_trait]
impl ModelProvider for Slow {
    fn name(&self) -> &str {
        "slow"
    }
    fn model_name(&self) -> &str {
        "slow"
    }
    async fn complete(
        &self,
        req: CompletionRequest,
    ) -> anyhow::Result<splinter_sdk::agent::sven::model::ResponseStream> {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        self.0.complete(req).await
    }
}

#[test]
fn the_wait_for_the_server_is_measured_per_request() {
    let stats = Arc::new(Mutex::new(Stats::default()));
    let timed = Timed::wrap(
        Arc::new(Slow(ScriptedAgent::provider(Behaviour::Honest, vec![]))),
        Arc::clone(&stats),
    );
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        use futures::StreamExt;
        let mut stream = timed.complete(CompletionRequest::default()).await.unwrap();
        while stream.next().await.is_some() {}
    });
    let seen = stats.lock().unwrap().clone();
    assert_eq!(seen.requests, 1);
    assert!(seen.max_wait_seconds >= 0.3, "{seen:?}");
}

fn run(behaviour: Behaviour, commit: usize) -> Vec<ledger::Entry> {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[commit - 1].clone());
    config.to = Some(ids[commit].clone());
    config.styles = vec![PlanStyle::Terse];
    run_blocking(
        &ctx,
        &scripted_models(behaviour, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap()
    .entries
}

#[test]
fn an_attempt_records_its_requests_and_the_plans_code_share() {
    let entries = run(Behaviour::Honest, 2);
    let e = entries
        .iter()
        .find(|e| e.outcome == Outcome::Passed)
        .unwrap();
    assert!(e.requests >= 2, "{e:?}");
    assert!(e.model_seconds >= 0.0 && e.max_wait_seconds >= 0.0);
    assert_eq!(e.plan_code_percent, Some(0));
}

#[test]
fn a_file_the_commit_removes_and_the_agent_left_is_a_mismatch_not_junk() {
    // Commit 4 of the fixture deletes shapes.py; this agent skips it.
    let entries = run(Behaviour::MissingEdit, 4);
    let e = entries
        .iter()
        .find(|e| e.outcome == Outcome::Failed)
        .unwrap();
    assert!(e.classes.iter().any(|c| c == "tree_mismatch"), "{e:?}");
    assert!(!e.classes.iter().any(|c| c == "junk"), "{e:?}");
    assert!(
        e.detail
            .as_deref()
            .unwrap_or_default()
            .contains("still present")
            || e.classes.len() > 1,
        "{e:?}"
    );
}
