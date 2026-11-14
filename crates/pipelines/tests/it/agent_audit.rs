// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose agent sessions keep
// their audit trail with the run that made them, for its clients. If your
// team needs expertise in auditable agent pipelines, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the agent sessions a command runs keep their audit trail in the
//! command's own state, so two runs (or two tests) never share one audit
//! file and its lock, and nothing about a run is written to the user's
//! shared state directory.

#![allow(clippy::unwrap_used)]

use crate::common;

use common::{scratch_context, Scripted};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::ask::ask;

#[test]
fn a_question_asked_of_a_handed_in_model_is_audited_in_the_runs_own_state() {
    let (scratch, ctx) = scratch_context("agent-audit", Scripted::new(|_| "Paris".into()), false);
    let answer = ask(
        &ctx,
        "What is the capital of France?",
        None,
        &ModelRef::policy_default(),
    )
    .unwrap();
    assert!(answer.answer.contains("Paris"), "{}", answer.answer);
    let log = scratch.state().agent_audit_log();
    let trail = std::fs::read_to_string(&log)
        .unwrap_or_else(|e| panic!("no audit trail at {}: {e}", log.display()));
    assert!(
        !trail.trim().is_empty(),
        "the session wrote no audit record"
    );
}
