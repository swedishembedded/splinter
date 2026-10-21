// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the harness document drives the replay and the models see exactly
//! what the Rust path shows them. Every prompt of the document (the plan
//! writer's and the agent's system prompts, the user turns) is compared with
//! what the Rust path sent for the same commits, request by request.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::control::Control;
use splinter_repo_replay::fixture;
use splinter_repo_replay::harness::{
    Decoding, HarnessConfig, DEFAULT_DOCUMENT, DEFAULT_PLAN_ONLY_DOCUMENT,
};
use splinter_repo_replay::pipeline::{Models, RunConfig};
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseStream, Role,
};

/// A provider that notes the system message and the first user message of
/// every request it forwards.
struct Recording {
    inner: Arc<dyn ModelProvider>,
    seen: Arc<Mutex<Vec<(String, String)>>>,
}

fn text_of(request: &CompletionRequest, role: Role) -> String {
    request
        .messages
        .iter()
        .find(|m| m.role == role)
        .and_then(|m| match &m.content {
            MessageContent::Text(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[async_trait::async_trait]
impl ModelProvider for Recording {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push((
            text_of(&request, Role::System),
            text_of(&request, Role::User),
        ));
        self.inner.complete(request).await
    }
}

/// The system and the first user message of each request, sorted.
type Asked = Vec<(String, String)>;

/// Runs the fixture history and returns, sorted, what was asked of the planner and of the agent.
fn requests(control: Control, harness: bool) -> (Asked, Asked) {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let scripted = scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths);
    let (planner_seen, agent_seen) = (Arc::default(), Arc::default());
    let models = Models {
        planner: Model::new(
            Arc::new(Recording {
                inner: Arc::clone(&scripted.planner.provider),
                seen: Arc::clone(&planner_seen),
            }),
            "scripted/planner",
        ),
        agent: Arc::new(Recording {
            inner: scripted.agent,
            seen: Arc::clone(&agent_seen),
        }),
        ..scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths)
    };
    let config = RunConfig {
        control,
        harness: harness.then(|| HarnessConfig {
            document: DEFAULT_DOCUMENT.to_string(),
            plan_only_document: DEFAULT_PLAN_ONLY_DOCUMENT.to_string(),
            check_bin: env!("CARGO_BIN_EXE_repo-replay").into(),
            decoding: Decoding {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
                seed: 1, // seed: the scripted models ignore it
                thinking: false,
            },
            thinking_budget: None,
        }),
        ..default_config(&source, &state)
    };
    run_blocking(&ctx, &models, &config).unwrap();
    let sorted = |seen: Arc<Mutex<Vec<(String, String)>>>| {
        let mut all = seen.lock().unwrap().clone();
        all.sort();
        all
    };
    (sorted(planner_seen), sorted(agent_seen))
}

#[test]
fn the_document_shows_the_models_what_the_rust_path_shows_them() {
    for control in [Control::Grounded, Control::PlanOnly] {
        let (rust_planner, rust_agent) = requests(control, false);
        let (document_planner, document_agent) = requests(control, true);
        assert!(!rust_planner.is_empty() && !rust_agent.is_empty());
        assert_eq!(
            rust_planner, document_planner,
            "planner prompts, {control:?}"
        );
        // The agent's first turn of every attempt: system prompt and task.
        let first_turns = |all: &[(String, String)]| {
            let mut firsts: Vec<(String, String)> = all
                .iter()
                .filter(|(_, user)| !user.starts_with("You stopped before finishing"))
                .cloned()
                .collect();
            firsts.dedup();
            firsts
        };
        assert_eq!(
            first_turns(&rust_agent),
            first_turns(&document_agent),
            "agent prompts, {control:?}"
        );
    }
}
