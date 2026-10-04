// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuned models that answer under the
// prompt they were trained under, for its clients. If your team needs
// expertise in training and serving persona models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a model that answers under a system prompt of its own is asked
//! under it, wherever it is asked; a model with none is asked under the
//! default, so the teacher and the judge never speak as the person the policy
//! was trained to be.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{scratch_context, Scripted};
use splinter_agent::solve::{Model, SYSTEM_PROMPT};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::ask::ask;

const PERSONA: &str = "You are a surveyor of the old school. Answer as one.";

#[test]
fn a_model_is_asked_under_its_own_system_prompt_and_another_under_the_default() {
    let (_scratch, ctx) = scratch_context("system-prompt", Scripted::new(|_| String::new()), false);
    let policy = Scripted::new(|_| "Measure twice.".into());
    let other = Scripted::new(|_| "Measure once.".into());
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), "scripted/policy").with_system(PERSONA),
    );
    let assistant: ModelRef = "local:test/assistant".parse().unwrap();
    ctx.add_model(
        assistant.clone(),
        Model::new(Arc::new(other.clone()), "scripted/assistant"),
    );
    ask(
        &ctx,
        "How do you measure?",
        None,
        &ModelRef::policy_default(),
    )
    .unwrap();
    ask(&ctx, "How do you measure?", None, &assistant).unwrap();
    assert_eq!(policy.systems.lock().unwrap()[0], [PERSONA]);
    assert_eq!(other.systems.lock().unwrap()[0], [SYSTEM_PROMPT]);
}
