// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! Every way a Harbor trial ends maps to one status, an infrastructure
//! failure is never the agent's 0, and a leak is a veto whatever the score.
//! The cases are the task repository's golden trial records, unchanged.

#![allow(clippy::expect_used)]

use serde_json::Value;
use splinter_core::factory::trial::{
    grade, GradingPolicy, ModelRequest, TimeoutPolicy, TrialEvidence, TrialStatus,
};

fn cases() -> Vec<Value> {
    serde_json::from_str(include_str!("harbor-trials.json")).expect("the fixture is JSON")
}

fn evidence(case: &Value) -> (TrialEvidence, GradingPolicy) {
    let record = &case["record"];
    let mut evidence: TrialEvidence =
        serde_json::from_value(record.clone()).expect("a trial record");
    evidence.requests =
        serde_json::from_value::<Vec<ModelRequest>>(case["requests"].clone()).expect("requests");
    let timeout = match record
        .pointer("/entry/timeout_policy")
        .and_then(Value::as_str)
    {
        Some("excluded") => TimeoutPolicy::Excluded,
        _ => TimeoutPolicy::Graded,
    };
    (
        evidence,
        GradingPolicy {
            timeout,
            budget: None,
        },
    )
}

#[test]
fn every_way_a_harbor_trial_ends_has_its_status() {
    let cases = cases();
    assert_eq!(cases.len(), 30, "every golden case is checked");
    for case in &cases {
        let (evidence, policy) = evidence(case);
        let graded = grade(&evidence, &policy);
        let expect = &case["expect"];
        let name = case["name"].as_str().expect("a name");
        let status: TrialStatus =
            serde_json::from_value(expect["status"].clone()).expect("a status");
        assert_eq!(graded.status, status, "{name}: {}", case["why"]);
        assert_eq!(
            Value::Bool(graded.counted),
            expect["counted"],
            "{name}: counted"
        );
        assert_eq!(
            serde_json::to_value(graded.passed).expect("json"),
            expect["pass"],
            "{name}: pass"
        );
        if let Some(score) = expect.get("score") {
            assert_eq!(graded.score, score.as_f64(), "{name}: score");
        }
        if let Some(code) = expect.get("cause").and_then(Value::as_str) {
            assert_eq!(
                graded.cause.as_ref().map(|c| c.code.as_str()),
                Some(code),
                "{name}: cause"
            );
        }
        if graded.status != TrialStatus::Graded {
            assert!(
                graded.cause.as_ref().is_some_and(|c| !c.message.is_empty()),
                "{name}: a person can read why"
            );
        }
        if !matches!(
            graded.status,
            TrialStatus::Graded | TrialStatus::Timeout | TrialStatus::SecurityViolation
        ) {
            assert!(
                graded.score.is_none() && graded.metrics.is_empty(),
                "{name}: no score on infrastructure"
            );
        }
    }
}

#[test]
fn a_timeout_names_the_tasks_policy() {
    let cases = cases();
    let policy_of = |name: &str| {
        let case = cases.iter().find(|c| c["name"] == name).expect("the case");
        let (evidence, policy) = evidence(case);
        grade(&evidence, &policy).cause.and_then(|c| c.policy)
    };
    assert_eq!(policy_of("sven_step_timeout"), Some(TimeoutPolicy::Graded));
    assert_eq!(
        policy_of("timeout_excluded_by_policy"),
        Some(TimeoutPolicy::Excluded)
    );
}

#[test]
fn a_leak_is_a_hard_veto_whatever_the_score() {
    let cases = cases();
    let case = cases
        .iter()
        .find(|c| c["name"] == "leak")
        .expect("the case");
    let (evidence, policy) = evidence(case);
    let graded = grade(&evidence, &policy);
    assert_eq!((graded.score, graded.passed), (Some(1.0), Some(false)));
    assert_eq!(graded.vetoes, vec!["shell: cat /tests/test.sh".to_owned()]);
}

#[test]
fn a_run_budget_is_a_counted_fail_and_a_wall_clock_end_is_the_infrastructures_only_when_the_model_never_answered(
) {
    let cases = cases();
    let passed = cases
        .iter()
        .find(|c| c["name"] == "passed")
        .expect("the case");
    let (within, _) = evidence(passed);
    let tight = GradingPolicy {
        timeout: TimeoutPolicy::Graded,
        budget: Some(splinter_core::factory::trial::RunBudget {
            completion_tokens: 10,
            turns: 100,
        }),
    };
    let over = grade(&within, &tight);
    assert_eq!(
        (over.status, over.counted, over.passed),
        (TrialStatus::BudgetExhausted, true, Some(false))
    );
    let timed = cases
        .iter()
        .find(|c| c["name"] == "harbor_agent_timeout")
        .expect("the case");
    let (timed_out, _) = evidence(timed);
    let roomy = GradingPolicy {
        timeout: TimeoutPolicy::Graded,
        budget: Some(splinter_core::factory::trial::RunBudget {
            completion_tokens: 1_000_000,
            turns: 1000,
        }),
    };
    // The model never answered in this trial: the infrastructure stalled it.
    let mut stalled = timed_out.clone();
    stalled.requests.clear();
    stalled.turns = None;
    let wall = grade(&stalled, &roomy);
    assert_eq!(
        (wall.status, wall.counted),
        (TrialStatus::InfraTimeout, false)
    );
    assert!(wall.status.is_retryable());
    // The model answered and the time ran out: out of time, graded as it
    // stands and counted, as a person at an exam is.
    let mut working = timed_out;
    working.requests = vec![ModelRequest {
        status: Some(200),
        ..ModelRequest::default()
    }];
    let late = grade(&working, &roomy);
    assert_eq!(late.status, TrialStatus::Timeout);
    assert!(late.counted, "{late:?}");
    assert!(!late.status.is_retryable());
}

#[test]
fn an_endpoint_that_answered_no_request_is_the_infrastructures_not_the_agents_zero() {
    let cases = cases();
    let failed = cases
        .iter()
        .find(|c| c["name"] == "quit_after_one_turn")
        .expect("the case");
    let (mut evidence, policy) = evidence(failed);
    let rejected = |status| ModelRequest {
        status: Some(status),
        error: Some("the anthropic-beta header is not supported on this API".into()),
        ..ModelRequest::default()
    };
    evidence.requests = vec![rejected(400), rejected(400), rejected(400)];
    let graded = grade(&evidence, &policy);
    assert_eq!(
        (graded.status, graded.counted, graded.passed),
        (TrialStatus::ModelError, false, None)
    );
    let cause = graded.cause.expect("a cause");
    assert_eq!(cause.code, "endpoint_rejected");
    assert!(
        cause.message.contains("anthropic-beta"),
        "{}",
        cause.message
    );
    assert!(graded.status.is_retryable());

    // Once the model answered, a rejected request is the agent's own doing.
    evidence.requests.insert(
        0,
        ModelRequest {
            status: Some(200),
            ..ModelRequest::default()
        },
    );
    assert_eq!(grade(&evidence, &policy).status, TrialStatus::Graded);
}

#[test]
fn an_endpoint_rejecting_every_request_with_any_4xx_status_is_an_endpoint_rejection() {
    let cases = cases();
    let failed = cases
        .iter()
        .find(|c| c["name"] == "quit_after_one_turn")
        .expect("the case");
    let (mut evidence, policy) = evidence(failed);
    // 400 and 498 bound the statuses an endpoint answers with; 499 is the gateway's record of the agent hanging up.
    for status in [400, 498] {
        evidence.requests = vec![ModelRequest {
            status: Some(status),
            error: Some("rejected".into()),
            ..ModelRequest::default()
        }];
        let graded = grade(&evidence, &policy);
        assert_eq!(
            graded.cause.map(|c| c.code),
            Some("endpoint_rejected".to_owned()),
            "HTTP {status}"
        );
    }
}

#[test]
fn an_agent_that_hung_up_on_its_model_was_not_rejected_by_the_endpoint() {
    let cases = cases();
    let failed = cases
        .iter()
        .find(|c| c["name"] == "quit_after_one_turn")
        .expect("the case");
    let (mut evidence, policy) = evidence(failed);
    // The gateway records 499 when the agent closed the stream before the answer ended: sven does that when the
    // model loops inside a tool call's arguments. The loop is the model's failure, counted, and not asked again.
    evidence.requests = vec![ModelRequest {
        status: Some(499),
        error: Some("the agent closed the stream before the answer ended".into()),
        ..ModelRequest::default()
    }];
    let hung_up = grade(&evidence, &policy);
    assert_ne!(
        hung_up.cause.map(|c| c.code),
        Some("endpoint_rejected".to_owned())
    );
    evidence.agent_reasoning_loop = true;
    let looped = grade(&evidence, &policy);
    assert_eq!(looped.status, TrialStatus::ReasoningLoop);
    assert!(looped.counted && looped.passed == Some(false));
    assert!(!looped.status.is_retryable());
}

#[test]
fn a_verifier_metric_that_is_no_valid_measurement_makes_the_trial_unusable_not_metricless() {
    let cases = cases();
    let partial = cases
        .iter()
        .find(|c| c["name"] == "partial_credit")
        .expect("the case");
    for (metric, value) in [
        ("invariants_hold", serde_json::json!(1.5)),
        ("invariants_hold", serde_json::json!(-0.1)),
        ("invariants_hold", serde_json::json!("high")),
    ] {
        let (mut evidence, policy) = evidence(partial);
        evidence.result["verifier_result"]["rewards"][metric] = value.clone();
        let graded = grade(&evidence, &policy);
        assert_eq!(
            (graded.status, graded.counted, graded.score),
            (TrialStatus::VerifierError, false, None),
            "{metric}={value}"
        );
        let cause = graded.cause.expect("a cause");
        assert_eq!(cause.code, "malformed_reward", "{metric}={value}");
        assert!(cause.message.contains(metric), "{}", cause.message);
    }
}

/// An agent that never asked its model anything and completed no turn failed to run, not the task.
#[test]
fn an_agent_that_never_asked_its_model_is_not_graded_a_failure() {
    let cases = cases();
    let failed = cases
        .iter()
        .find(|c| c["name"] == "quit_after_one_turn")
        .expect("the case");
    let (mut evidence, policy) = evidence(failed);
    evidence.requests = Vec::new();
    evidence.turns = Some(0);
    evidence.model_responses = None;
    let graded = grade(&evidence, &policy);
    assert_eq!(
        (graded.status, graded.counted, graded.passed),
        (TrialStatus::AgentError, false, None)
    );
    assert_eq!(graded.cause.expect("a cause").code, "no_model_request");
    // With a completed turn the same trial is the model's.
    evidence.turns = Some(1);
    assert_eq!(grade(&evidence, &policy).status, TrialStatus::Graded);
}
