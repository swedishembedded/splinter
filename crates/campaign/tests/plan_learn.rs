// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that survey a body of writing and
// plan what to learn from it, for its clients. If your team needs expertise
// in turning an unfamiliar corpus into a learning plan, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: `learn --plan` lets a model decide how to learn. After the sources
//! are captured the run surveys them, the planner chooses the kinds of task
//! (and whether to distil), and the tasks stage generates exactly those; the
//! survey and the plan are recorded in the report. Naming kinds as well as
//! asking for a plan is refused: one of the two decides.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::manual::{manual_reply, MANUAL};
use common::Scripted;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_campaign::learn::auto_steps;
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::release::arm;
use splinter_campaign::train::DEFAULT_STEPS;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_core::model_ref::ModelRef;
use splinter_core::role::Role;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};

const RECALL_ONLY: &str = r#"{"persona": null, "kinds": ["recall"], "distill": false, "rationale": "the manual states facts"}"#;
const DISTILLED: &str = r#"{"persona": null, "kinds": ["recall"], "distill": true, "rationale": "the manual is not known to the learner"}"#;

struct Student {
    plans: Mutex<Vec<TrainPlan>>,
}

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.plans.lock().unwrap().push(plan.clone());
        let trained = FakeTrainer::knowing(&[ANCHOR]).train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(
                Arc::new(Scripted::new(|_| "I do not know.".into())),
                "scripted/learned",
            ),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        panic!("learn trains chat datasets only")
    }
}

/// The manual-learning policy, which also answers the planner's call with
/// `plan`.
fn policy(plan: &'static str) -> Scripted {
    Scripted::new(move |prompt| {
        if prompt.contains("You plan how a learning system should learn") {
            plan.to_string()
        } else {
            manual_reply(prompt)
        }
    })
}

fn run(
    test: &str,
    plan: &'static str,
    request: impl FnOnce(String) -> LearnRequest,
) -> Result<(splinter_campaign::learn::LearnReport, common::Scratch), OrchestratorError> {
    let (scratch, ctx) = gate_context(test, Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy(plan)), common::POLICY),
    );
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    match learn(&ctx, &request(manual.display().to_string()), &student)? {
        Learned::Ran(ran) => Ok((ran.report, scratch)),
        Learned::Planned(_) => panic!("not a dry run"),
    }
}

#[test]
fn a_planned_learn_generates_the_kinds_the_planner_chose_and_records_the_plan() {
    let (report, _scratch) = run("plan-learn", RECALL_ONLY, |manual| LearnRequest {
        sources: vec![manual],
        plan: true,
        no_release: true,
        ..LearnRequest::default()
    })
    .unwrap();

    let planned = report.plan.as_ref().unwrap();
    assert_eq!(planned.plan.kinds, ["recall"]);
    assert_eq!(planned.survey.parts, 1);
    assert_eq!(planned.survey.advice_sections, 0);
    let generated = report.tasks.as_ref().unwrap();
    assert_eq!(generated.per_kind.keys().collect::<Vec<_>>(), ["recall"]);
    assert!(report.solve.is_some(), "the student made its attempts");
}

#[test]
fn a_plan_that_distils_skips_the_students_attempts() {
    let (report, _scratch) = run("plan-learn-distill", DISTILLED, |manual| LearnRequest {
        sources: vec![manual],
        plan: true,
        no_release: true,
        ..LearnRequest::default()
    })
    .unwrap();
    assert!(report.plan.as_ref().unwrap().plan.distill);
    assert!(report.solve.is_none() && report.verify.is_none());
    assert!(report.teach.is_some());
}

#[test]
fn naming_kinds_and_asking_for_a_plan_is_refused() {
    let error = run("plan-learn-both", RECALL_ONLY, |manual| LearnRequest {
        sources: vec![manual],
        plan: true,
        kinds: vec!["recall".into()],
        ..LearnRequest::default()
    })
    .err()
    .unwrap();
    assert!(error.is_refusal(), "{error}");
}

/// The configuration names the stronger model that plans, writes the tasks
/// and teaches, so a sentence needs no flag for it; a model named on the
/// command wins over it.
#[test]
fn the_configured_assistant_is_the_default_planner_generator_and_teacher() {
    let scratch = common::Scratch::new("plan-learn-assistant");
    let mut settings = common::config(&scratch);
    settings.assistant_model = Some("local:Qwen/Qwen3-8B".into());
    let ctx = splinter_orchestrator::Context::new(settings, false).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    let dry = |generator: Option<ModelRef>| {
        let Learned::Planned(plan) = learn(
            &ctx,
            &LearnRequest {
                sources: vec![scratch.0.display().to_string()],
                plan: true,
                dry_run: true,
                roles: generator
                    .map(|g| (Role::Generator, g))
                    .into_iter()
                    .collect(),
                ..LearnRequest::default()
            },
            &student,
        )
        .unwrap() else {
            panic!("a dry run plans");
        };
        plan
    };
    let plan = dry(None);
    assert!(plan.generator.contains("Qwen3-8B"), "{}", plan.generator);
    assert_eq!(plan.teacher, plan.generator);
    assert_eq!(plan.planner.as_deref(), Some(plan.generator.as_str()));
    let named = dry(Some("local:./other".parse().unwrap()));
    assert!(named.generator.contains("other"), "{}", named.generator);
    assert!(
        named.teacher.contains("Qwen3-8B"),
        "the teacher is still the assistant"
    );
}

#[test]
fn an_assistant_that_is_no_model_reference_is_refused_by_name() {
    let scratch = common::Scratch::new("plan-learn-assistant-bad");
    let mut settings = common::config(&scratch);
    settings.assistant_model = Some("not a reference".into());
    let ctx = splinter_orchestrator::Context::new(settings, false).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    let error = learn(
        &ctx,
        &LearnRequest {
            sources: vec![scratch.0.display().to_string()],
            dry_run: true,
            ..LearnRequest::default()
        },
        &student,
    )
    .err()
    .unwrap();
    assert!(
        error.is_refusal() && error.to_string().contains("not a reference"),
        "{error}"
    );
}

/// A run trains for about two passes over what it has learned, never fewer
/// steps than the default and never more than a day's work.
#[test]
fn the_steps_follow_the_size_of_the_dataset() {
    assert_eq!(
        auto_steps(2),
        DEFAULT_STEPS,
        "a tiny set trains for the default"
    );
    assert_eq!(auto_steps(100), 200, "two passes over a hundred records");
    assert_eq!(auto_steps(1_000_000), auto_steps(5_000), "bounded above");
}
