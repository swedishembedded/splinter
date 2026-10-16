// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of what an agent really
// completed, and at what cost, for its clients. If your team needs expertise
// in benchmarking coding agents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a task outcome is derived from the experience, its annotations and
//! the attempt's report, and absent measurements stay absent. Quality,
//! cost and autonomy are reported with their raw counts and family-clustered
//! intervals; an environment failure leaves the denominators and is counted.

#![allow(clippy::unwrap_used)]

mod common;

use atif::{FinalMetrics, StepOrigin, ToolInvocation, TraceStep};
use common::{experience, task};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, ExperienceId};
use splinter_core::outcome::{AttemptReport, FailureClass, TaskOutcome};
use splinter_eval::competence::{competence, outcome_of, Attempt};

fn note(experience: &ExperienceId, body: AnnotationBody) -> Annotation {
    Annotation {
        experience: experience.clone(),
        producer: Producer {
            name: "t".into(),
            version: "1".into(),
        },
        body,
    }
}

fn verdict(outcome: Outcome, strength: Strength) -> AnnotationBody {
    AnnotationBody::Verdict {
        outcome,
        strength,
        evidence: serde_json::json!({}),
    }
}

#[test]
fn an_outcome_is_derived_from_the_experience_its_verdicts_and_the_attempt_report() {
    let task = task("workspace-change", Environment::closed_book(), vec![]);
    let mut exp = experience(&task, Some("patch"), "agent");
    exp.trajectory.final_metrics = Some(FinalMetrics {
        total_prompt_tokens: Some(900),
        total_completion_tokens: Some(40),
        ..FinalMetrics::default()
    });
    let mut step = TraceStep::new(1, StepOrigin::Agent, "run the tests");
    step.tool_calls = Some(vec![
        ToolInvocation::new("a", "shell"),
        ToolInvocation::new("b", "read"),
    ]);
    exp.trajectory.steps.push(step);
    let id = exp.id().unwrap();
    let report = AttemptReport {
        wall_seconds: Some(12.5),
        tool_errors: Some(1),
        failure: None,
        ..AttemptReport::default()
    };
    let notes = [
        note(&id, verdict(Outcome::Pass, Strength::Executable)),
        note(&id, AnnotationBody::AttemptReport(report)),
    ];
    let out = outcome_of(&exp, &notes).unwrap();
    assert_eq!(out.task_id, task.task.id);
    assert_eq!(out.experience_id, id);
    assert!(out.verified_success && out.submitted);
    assert_eq!(out.verification_strength, Some(Strength::Executable));
    assert_eq!(
        (out.prompt_tokens, out.completion_tokens),
        (Some(900), Some(40))
    );
    // The trajectory counts the calls when the report does not say.
    assert_eq!(out.tool_calls, Some(2));
    assert_eq!((out.wall_seconds, out.tool_errors), (Some(12.5), Some(1)));
    // Nothing recorded these: absent, not zero.
    assert_eq!(out.gpu_seconds, None);
    assert_eq!(out.human_minutes, None);
    assert_eq!(out.attempted_actions, None);
    assert_eq!(out.unauthorised_actions, None);
}

#[test]
fn an_attempt_nothing_decided_is_not_verified_and_an_answer_without_a_run_has_no_costs() {
    let task = task("workspace-change", Environment::closed_book(), vec![]);
    let exp = experience(&task, None, "agent");
    let id = exp.id().unwrap();
    // A constraint that passed counts as an abstention: nothing decided.
    let notes = [note(&id, verdict(Outcome::Pass, Strength::Constraint))];
    let out = outcome_of(&exp, &notes).unwrap();
    assert!(!out.verified_success && !out.submitted);
    assert_eq!(out.verification_strength, None);
    assert_eq!(
        (out.prompt_tokens, out.tool_calls, out.wall_seconds),
        (None, None, None)
    );
}

fn attempt(
    family: &str,
    verified: bool,
    wall: Option<f64>,
    prompt: u64,
    human: Option<f64>,
) -> Attempt {
    Attempt {
        family: family.into(),
        outcome: TaskOutcome {
            task_id: Digest::of(family.as_bytes()),
            experience_id: ExperienceId(Digest::of(
                format!("{family}{verified}{prompt}").as_bytes(),
            )),
            verified_success: verified,
            verification_strength: Some(Strength::Executable),
            submitted: true,
            wall_seconds: wall,
            gpu_seconds: None,
            prompt_tokens: Some(prompt),
            completion_tokens: Some(prompt / 10),
            tool_calls: Some(4),
            tool_errors: None,
            attempted_actions: None,
            unauthorised_actions: None,
            failure: None,
            human_minutes: human,
        },
    }
}

#[test]
fn quality_cost_and_autonomy_are_computed_by_hand_with_absent_costs_left_absent() {
    let mut env = attempt("c", false, None, 999, None);
    env.outcome.failure = Some(FailureClass::Environment);
    let attempts = [
        attempt("a", true, Some(10.0), 100, None),
        attempt("a", false, Some(20.0), 200, None),
        attempt("b", true, Some(30.0), 300, Some(5.0)),
        attempt("b", true, Some(40.0), 400, None),
        env,
    ];
    let report = competence(&attempts, 7);
    let c = &report.counts;
    assert_eq!(
        (c.attempted, c.verified, c.verified_without_human),
        (4, 3, 2)
    );
    assert_eq!((c.excluded_environment, c.families), (1, 2));
    // Q = 3 / 4, A = 2 / 4.
    assert_eq!(report.quality.value, Some(0.75));
    assert_eq!(report.autonomy.value, Some(0.5));
    // C per verified: totals over the four attempted, divided by three.
    assert_eq!(report.cost.wall_seconds.total, Some(100.0));
    assert!((report.cost.wall_seconds.per_verified.value.unwrap() - 100.0 / 3.0).abs() < 1e-9);
    assert_eq!(report.cost.prompt_tokens.total, Some(1000.0));
    assert_eq!(report.cost.tool_calls.total, Some(16.0));
    // No attempt recorded device time: absent, with every attempt missing it.
    assert_eq!(report.cost.gpu_seconds.total, None);
    assert_eq!(report.cost.gpu_seconds.missing, 4);
    assert_eq!(report.cost.gpu_seconds.per_verified.value, None);
    // Each estimate carries an interval around it, the same for the same seed.
    let (low, high) = (report.quality.low.unwrap(), report.quality.high.unwrap());
    assert!((0.0..=0.75).contains(&low) && (0.75..=1.0).contains(&high));
    assert_eq!(report, competence(&attempts, 7));
}

#[test]
fn a_cost_missing_from_one_attempt_has_no_total_and_nothing_verified_has_no_cost_per_verified() {
    let attempts = [
        attempt("a", false, Some(10.0), 100, None),
        attempt("b", false, None, 100, None),
    ];
    let report = competence(&attempts, 1);
    assert_eq!(report.quality.value, Some(0.0));
    assert_eq!(report.cost.wall_seconds.total, None);
    assert_eq!(report.cost.wall_seconds.missing, 1);
    assert_eq!(report.cost.prompt_tokens.total, Some(200.0));
    assert_eq!(report.cost.prompt_tokens.per_verified.value, None);

    let only_environment = {
        let mut a = attempt("a", false, None, 1, None);
        a.outcome.failure = Some(FailureClass::Environment);
        competence(&[a], 1)
    };
    assert_eq!(only_environment.quality.value, None);
    assert_eq!(only_environment.counts.excluded_environment, 1);
}
