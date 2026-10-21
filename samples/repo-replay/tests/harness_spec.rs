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
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
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
fn requests(control: Control, harness: bool, corrections: usize) -> (Asked, Asked, Asked) {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let behaviour = if corrections > 0 {
        Behaviour::FixesAfterCorrection
    } else {
        Behaviour::Honest
    };
    let scripted = scripted_models(behaviour.clone(), Specificity::NamesFiles, &truths);
    let (planner_seen, agent_seen, supervisor_seen) =
        (Arc::default(), Arc::default(), Arc::default());
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
        supervisor: Model::new(
            Arc::new(Recording {
                inner: Arc::clone(&scripted.supervisor.provider),
                seen: Arc::clone(&supervisor_seen),
            }),
            "scripted/supervisor",
        ),
        ..scripted_models(behaviour, Specificity::NamesFiles, &truths)
    };
    let config = RunConfig {
        control,
        corrections,
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
                thinking_budget: None,
            },
        }),
        ..default_config(&source, &state)
    };
    run_blocking(&ctx, &models, &config).unwrap();
    let sorted = |seen: Arc<Mutex<Vec<(String, String)>>>| {
        let mut all = seen.lock().unwrap().clone();
        all.sort();
        all
    };
    (
        sorted(planner_seen),
        sorted(agent_seen),
        sorted(supervisor_seen),
    )
}

#[test]
fn the_document_shows_the_models_what_the_rust_path_shows_them() {
    for control in [Control::Grounded, Control::PlanOnly] {
        let (rust_planner, rust_agent, _) = requests(control, false, 0);
        let (document_planner, document_agent, _) = requests(control, true, 0);
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

#[test]
fn the_supervisor_is_asked_what_the_rust_path_asks_it_and_the_agent_is_told_the_same() {
    let (_, rust_agent, rust_supervisor) = requests(Control::Grounded, false, 3);
    let (_, document_agent, document_supervisor) = requests(Control::Grounded, true, 3);
    assert!(
        !rust_supervisor.is_empty(),
        "the scripted run needed a correction"
    );
    assert_eq!(rust_supervisor, document_supervisor, "supervisor prompts");
    assert_eq!(rust_agent.len(), document_agent.len(), "agent requests");
}

/// A planner that writes a plan too short to be one until it has been refused
/// (`always`: never writes a good one).
struct StubbornPlanner {
    inner: Arc<dyn ModelProvider>,
    always: bool,
}

#[async_trait::async_trait]
impl ModelProvider for StubbornPlanner {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let refused_before = request
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .count()
            > 1;
        if self.always || !refused_before {
            return Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::TextDelta("Do it.".into())),
                Ok(ResponseEvent::Done),
            ])));
        }
        self.inner.complete(request).await
    }
}

/// The outcome and classes of every attempt of the fixture history under a planner that refuses first.
fn under_a_stubborn_planner(harness: bool, always: bool) -> Vec<(String, Vec<String>)> {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let scripted = scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths);
    let models = Models {
        planner: Model::new(
            Arc::new(StubbornPlanner {
                inner: Arc::clone(&scripted.planner.provider),
                always,
            }),
            "scripted/planner",
        ),
        ..scripted
    };
    let config = RunConfig {
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
                thinking_budget: None,
            },
        }),
        ..default_config(&source, &state)
    };
    let report = run_blocking(&ctx, &models, &config).unwrap();
    report
        .entries
        .into_iter()
        .filter(|e| e.outcome != splinter_repo_replay::ledger::Outcome::Skipped)
        .map(|e| (format!("{:?}", e.outcome), e.classes))
        .collect()
}

#[test]
fn a_refused_plan_is_written_again_and_a_plan_refused_twice_ends_the_attempt_as_on_the_rust_path() {
    for always in [false, true] {
        let rust = under_a_stubborn_planner(false, always);
        let document = under_a_stubborn_planner(true, always);
        assert!(!rust.is_empty());
        assert_eq!(rust, document, "always refusing: {always}");
    }
}
