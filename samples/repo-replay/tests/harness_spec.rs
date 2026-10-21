// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the harness documents drive the replay and the models are told what
//! the documents say: the system prompts and the user turns are the documents'
//! text with the run's inputs in the holes, and nothing else.

#![allow(clippy::unwrap_used)]

mod support;

use std::sync::{Arc, Mutex};

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::control::Control;
use splinter_repo_replay::fixture;
use splinter_repo_replay::pipeline::Models;
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

struct Seen {
    planner: Asked,
    agent: Asked,
    supervisor: Asked,
}

/// Runs the fixture history and returns, sorted, what each model was asked.
fn requests(control: Control, corrections: usize) -> Seen {
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
    let mut config = default_config(&source, &state);
    config.control = control;
    config.corrections = corrections;
    run_blocking(&ctx, &models, &config).unwrap();
    let sorted = |seen: Arc<Mutex<Asked>>| {
        let mut all = seen.lock().unwrap().clone();
        all.sort();
        all
    };
    Seen {
        planner: sorted(planner_seen),
        agent: sorted(agent_seen),
        supervisor: sorted(supervisor_seen),
    }
}

/// The text of `template` before its first hole and after its last one.
fn around_holes(template: &str) -> (String, String) {
    let first = template.find("{{").unwrap();
    let last = template.rfind("}}").unwrap() + 2;
    (template[..first].to_string(), template[last..].to_string())
}

#[test]
fn the_planner_is_asked_what_the_document_says() {
    let seen = requests(Control::Grounded, 0);
    assert!(!seen.planner.is_empty());
    for style in ["terse", "detailed"] {
        let system = support::plan_system(style);
        let asked: Vec<_> = seen.planner.iter().filter(|(s, _)| *s == system).collect();
        assert!(
            !asked.is_empty(),
            "{style}: no request carried the document's system prompt"
        );
        for (_, user) in asked {
            assert!(
                user.starts_with("Commit message:\n") && user.ends_with("\n\nWrite the request."),
                "{user}"
            );
        }
    }
    // Nothing else was said to the planner as a system prompt.
    assert!(seen.planner.iter().all(|(s, _)| {
        s == &support::plan_system("terse") || s == &support::plan_system("detailed")
    }));
}

#[test]
fn the_agent_is_told_what_the_document_of_its_control_says() {
    for (control, variant) in [
        (Control::Grounded, "grounded"),
        (Control::PlanOnly, "plan_only"),
    ] {
        let template = support::agent_system(variant);
        let (before, after) = around_holes(&template);
        let seen = requests(control, 0);
        assert!(!seen.agent.is_empty());
        for (system, _) in &seen.agent {
            assert!(system.starts_with(&before), "{control:?}: {system}");
            assert!(system.ends_with(&after), "{control:?}: {system}");
            assert!(!system.contains("{{"), "a hole stayed open: {system}");
        }
        let grounded = matches!(control, Control::Grounded);
        assert_eq!(seen.agent[0].0.contains("The change:"), grounded);
    }
}

#[test]
fn the_supervisor_is_asked_what_the_document_says() {
    let seen = requests(Control::Grounded, 3);
    assert!(
        !seen.supervisor.is_empty(),
        "the scripted run needed a correction"
    );
    let document = support::document();
    let author = &support::task(&document, "work")["corrections"]["author"];
    let system = author["system"].as_str().unwrap();
    let (before, after) = around_holes(author["task"].as_str().unwrap());
    for (asked_system, user) in &seen.supervisor {
        assert_eq!(asked_system, system);
        assert!(
            user.starts_with(&before) && user.ends_with(&after),
            "{user}"
        );
        assert!(user.contains("What the engineer changed:"), "{user}");
    }
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
fn under_a_stubborn_planner(always: bool) -> Vec<(String, Vec<String>)> {
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
    let report = run_blocking(&ctx, &models, &default_config(&source, &state)).unwrap();
    report
        .entries
        .into_iter()
        .filter(|e| e.outcome != splinter_repo_replay::ledger::Outcome::Skipped)
        .map(|e| (format!("{:?}", e.outcome), e.classes))
        .collect()
}

#[test]
fn a_refused_plan_is_written_again() {
    let outcomes = under_a_stubborn_planner(false);
    assert!(!outcomes.is_empty());
    assert!(
        outcomes.iter().all(|(o, c)| o == "Passed" && c.is_empty()),
        "{outcomes:?}"
    );
}

#[test]
fn a_plan_refused_twice_ends_the_attempt_as_refused_with_the_rule_that_refused_it() {
    let outcomes = under_a_stubborn_planner(true);
    assert!(!outcomes.is_empty());
    assert!(
        outcomes
            .iter()
            .all(|(o, c)| o == "PlanRefused" && c == &["plan_length".to_string()]),
        "{outcomes:?}"
    );
}
