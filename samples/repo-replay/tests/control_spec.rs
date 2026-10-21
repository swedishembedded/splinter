// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the plan-only control. The same commits are replayed with the plan
//! and the repository and nothing else; the pass rates of the grounded and
//! the plan-only run are reported side by side, because the gap between them
//! is the hidden information the grounding diff carries. A plan-only pass is
//! the purest record: nothing but the plan led to it.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::compare::Comparison;
use splinter_repo_replay::control::Control;
use splinter_repo_replay::fixture;
use splinter_repo_replay::git::Git;
use splinter_repo_replay::ledger::{self, Outcome};
use splinter_repo_replay::pipeline::Models;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::style::PlanStyle;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};

/// An agent that writes one file at once and records the system prompt it ran under.
struct WritesFile {
    path: &'static str,
    text: String,
    systems: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl ModelProvider for WritesFile {
    fn name(&self) -> &str {
        "writes-file"
    }
    fn model_name(&self) -> &str {
        "agent"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let system = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.as_text().map(str::to_string))
            .unwrap_or_default();
        self.systems.lock().unwrap().push(system);
        let answered = req
            .messages
            .iter()
            .any(|m| matches!(m.content, MessageContent::ToolResult { .. }));
        let events = if answered {
            vec![
                ResponseEvent::TextDelta("I added the module.".into()),
                ResponseEvent::Done,
            ]
        } else {
            vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: "w-0".into(),
                    name: "write_file".into(),
                    arguments: serde_json::json!({"path": self.path, "text": self.text})
                        .to_string(),
                },
                ResponseEvent::Done,
            ]
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

struct Setup {
    work: tempfile::TempDir,
    ids: Vec<String>,
    truths: Vec<String>,
    systems: Arc<Mutex<Vec<String>>>,
    shapes: String,
}

fn setup() -> Setup {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let shapes = Git::at(&source)
        .text(&["show", &format!("{}:shapes.py", ids[1])])
        .unwrap();
    Setup {
        work,
        ids,
        truths,
        systems: Arc::default(),
        shapes,
    }
}

impl Setup {
    /// A run of the commit that adds shapes.py into `name`, under `control`.
    fn run(&self, name: &str, control: Control, grounded_agent: bool) -> Vec<ledger::Entry> {
        let source = self.work.path().join("source");
        let state = self.work.path().join(name);
        let ctx = open_context(&state).unwrap();
        let mut config = default_config(&source, &state);
        config.from = Some(self.ids[0].clone());
        config.to = Some(self.ids[1].clone());
        config.styles = vec![PlanStyle::Terse];
        config.control = control;
        let mut models = scripted_models(Behaviour::Honest, Specificity::NamesFiles, &self.truths);
        if !grounded_agent {
            models = Models {
                agent: Arc::new(WritesFile {
                    path: "shapes.py",
                    text: self.shapes.clone(),
                    systems: Arc::clone(&self.systems),
                }),
                ..models
            };
        }
        run_blocking(&ctx, &models, &config).unwrap();
        ledger::read(&state).unwrap()
    }
}

fn attempts(entries: &[ledger::Entry]) -> Vec<&ledger::Entry> {
    entries
        .iter()
        .filter(|e| e.outcome != Outcome::Skipped)
        .collect()
}

#[test]
fn a_plan_only_attempt_is_never_shown_the_change_and_its_pass_is_flagged() {
    let s = setup();
    let entries = s.run("plan-only", Control::PlanOnly, false);
    let attempts = attempts(&entries);
    assert_eq!(attempts.len(), 1, "{entries:?}");
    assert_eq!(attempts[0].outcome, Outcome::Passed, "{:?}", attempts[0]);
    assert_eq!(attempts[0].control, Control::PlanOnly);

    let systems = s.systems.lock().unwrap();
    assert!(!systems.is_empty());
    for system in systems.iter() {
        for hidden in ["circle_area", "diff --git", "The change:", "xxx", "math.pi"] {
            assert!(!system.contains(hidden), "{hidden} in the prompt: {system}");
        }
        assert!(system.contains("working directory is the repository root"));
    }
}

#[test]
fn one_state_holds_one_control() {
    let s = setup();
    s.run("grounded", Control::Grounded, true);
    let source = s.work.path().join("source");
    let state = s.work.path().join("grounded");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.control = Control::PlanOnly;
    let err = run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &s.truths),
        &config,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("separate --state"), "{err:#}");
}

#[test]
fn the_comparison_puts_grounded_and_plan_only_side_by_side_on_the_same_commits() {
    let s = setup();
    let grounded = s.run("grounded", Control::Grounded, true);
    // A plan-only agent that cannot know the file fails.
    let blind = s.run("blind", Control::PlanOnly, true);
    let comparison = Comparison::of(&grounded, &blind);
    assert_eq!(comparison.grounded.passed, 1);
    assert_eq!(comparison.plan_only.attempts, 1);
    assert_eq!(comparison.plan_only.passed, 0);
    assert_eq!(comparison.paired, 1);
    assert_eq!(comparison.only_grounded, 1);
    assert_eq!(comparison.only_plan_only, 0);
    let text = comparison.to_string();
    assert!(
        text.contains("grounded") && text.contains("plan-only"),
        "{text}"
    );
    assert!(text.contains("1/1") && text.contains("0/1"), "{text}");

    let pure = s.run("pure", Control::PlanOnly, false);
    let both = Comparison::of(&grounded, &pure);
    assert_eq!((both.only_grounded, both.only_plan_only), (0, 0));
    assert_eq!(both.passed_in_both, 1);
}
