// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every record is
// content-addressed and traceable to its source, for its clients. If your
// team needs expertise in training-data provenance or reproducible
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: which model plays which role is decided in one place, by the same
//! rule for every pipeline.
//!
//! A role a command names wins; then the configured assistant for the roles
//! an assistant can play; then the policy itself. The planner falls back to
//! the generator before the policy, and the router to the model configured
//! for the front door. The policy is always the policy.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Fallbacks, ModelAssignments, Role};

fn model(text: &str) -> ModelRef {
    text.parse().unwrap()
}

fn assigned(named: &[(Role, &str)], fallbacks: &Fallbacks) -> ModelAssignments {
    let named: BTreeMap<Role, ModelRef> = named
        .iter()
        .map(|(role, text)| (*role, model(text)))
        .collect();
    ModelAssignments::resolve(&named, fallbacks)
}

#[test]
fn with_nothing_configured_every_role_is_the_policy() {
    let assignments = assigned(&[], &Fallbacks::default());
    for role in Role::ALL {
        assert_eq!(assignments.get(role), &ModelRef::policy_default(), "{role}");
    }
}

#[test]
fn the_assistant_plays_the_roles_that_ask_for_one() {
    let fallbacks = Fallbacks {
        assistant: Some(model("local:/models/assistant")),
        front_door: None,
    };
    let assignments = assigned(&[], &fallbacks);
    let assistant = model("local:/models/assistant");
    for role in [Role::Teacher, Role::Generator, Role::Planner, Role::Judge] {
        assert_eq!(assignments.get(role), &assistant, "{role}");
    }
    for role in [Role::Policy, Role::Critic, Role::Router] {
        assert_eq!(assignments.get(role), &ModelRef::policy_default(), "{role}");
    }
}

#[test]
fn a_role_a_command_names_wins_over_the_assistant() {
    let fallbacks = Fallbacks {
        assistant: Some(model("local:/models/assistant")),
        front_door: None,
    };
    let assignments = assigned(&[(Role::Teacher, "remote:openrouter/big")], &fallbacks);
    assert_eq!(
        assignments.get(Role::Teacher),
        &model("remote:openrouter/big")
    );
    assert_eq!(
        assignments.get(Role::Generator),
        &model("local:/models/assistant")
    );
}

#[test]
fn the_planner_falls_back_to_the_generator_before_the_policy() {
    let named = [(Role::Generator, "local:/models/writer")];
    let assignments = assigned(&named, &Fallbacks::default());
    assert_eq!(
        assignments.get(Role::Planner),
        &model("local:/models/writer")
    );
}

#[test]
fn the_router_is_the_model_configured_for_the_front_door() {
    let fallbacks = Fallbacks {
        assistant: Some(model("local:/models/assistant")),
        front_door: Some(model("local:/models/router")),
    };
    let assignments = assigned(&[], &fallbacks);
    assert_eq!(
        assignments.get(Role::Router),
        &model("local:/models/router")
    );
}

#[test]
fn the_policy_is_always_the_policy() {
    let assignments = assigned(
        &[(Role::Policy, "local:/models/other")],
        &Fallbacks::default(),
    );
    assert_eq!(assignments.get(Role::Policy), &ModelRef::policy_default());
}

#[test]
fn assignments_record_as_role_to_reference() {
    let assignments = assigned(&[(Role::Teacher, "local:/models/t")], &Fallbacks::default());
    let json = serde_json::to_value(&assignments).unwrap();
    assert_eq!(json["teacher"], "local:/models/t");
    assert_eq!(json["policy"], "policy:default");
    assert_eq!(json.as_object().unwrap().len(), Role::ALL.len());
}
