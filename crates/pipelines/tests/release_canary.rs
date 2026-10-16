// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a canary is a task no honest candidate passes. A candidate that
//! passes one is refused with a typed reason, whatever else it passed; one
//! that fails them all is released as before.
//!
//! The canaries here are tasks the scripted candidate happens to answer
//! (it knows their topic), standing for a candidate that satisfied a check it
//! should not have been able to.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{
    anchor_file, candidate, fact, fast_gate, gate_context, serve_release, Brain, ANCHOR,
};
use splinter_agent::CancelToken;
use splinter_core::experience::Task;
use splinter_orchestrator::Context;
use splinter_pipelines::release::{anchor, release, ReleaseRequest, Released};
use splinter_pipelines::train::Candidate;

fn decide_with(ctx: &Context, candidate: &Candidate, canaries: Vec<Task>) -> Released {
    let request = ReleaseRequest {
        gate: fast_gate(),
        canaries,
        ..ReleaseRequest::new(candidate.candidate.clone())
    };
    let decided = release(ctx, &request, &CancelToken::new()).unwrap();
    if let Some(id) = &decided.release {
        serve_release(ctx, id);
    }
    decided
}

#[test]
fn a_candidate_that_passes_a_canary_is_refused_whatever_else_it_passed() {
    let (scratch, ctx) = gate_context("canary-cheat", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    // "alpha" facts are answered by the candidate; "beta" ones are not.
    let canaries = vec![fact("beta", 0), fact("alpha", 1), fact("beta", 2)];
    let decided = decide_with(&ctx, &candidate, canaries.clone());

    let gate = &decided.gate;
    assert!(
        gate.improvement.passed && gate.retention.passed && gate.anchor.passed,
        "every other check passed: {gate:#?}"
    );
    assert!(!gate.passed && decided.release.is_none(), "{gate:#?}");
    assert_eq!(gate.canary_passes(), [canaries[1].task.id.to_string()]);
    let why = gate.blocked_by();
    assert!(why.iter().any(|w| w.starts_with("canary:")), "{why:?}");
    assert_eq!(ctx.releases().alias("default").unwrap(), None);
}

#[test]
fn a_candidate_that_fails_every_canary_is_released_and_the_gate_records_it() {
    let (scratch, ctx) = gate_context("canary-clean", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    let decided = decide_with(&ctx, &candidate, vec![fact("beta", 0), fact("beta", 1)]);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let canary = decided.gate.canary.as_ref().expect("the check is recorded");
    let measured = canary.measured.as_ref().unwrap();
    assert_eq!((measured.tasks, measured.graded), (2, 2));
    assert!(measured.passed_tasks.is_empty());
    assert!(decided.release.is_some());
}

#[test]
fn a_gate_handed_no_canary_has_no_canary_check() {
    let (scratch, ctx) = gate_context("canary-none", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let decided = decide_with(&ctx, &candidate, Vec::new());
    assert!(decided.gate.passed && decided.gate.canary.is_none());
}
