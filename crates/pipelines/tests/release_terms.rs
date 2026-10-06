// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: the terms data came under travel from a source to every release
//! made from it, and bind there.
//!
//! * A source's terms are part of its identity; a dataset states the combined
//!   terms of its sources, an unstated source counting as unknown.
//! * A candidate carries the combination of its datasets' terms and those of
//!   the release it continues, so the strongest restriction anywhere upstream
//!   is the one on every release after it.
//! * An unrestricted release is refused, before anything is measured, unless
//!   every axis of those terms is allowed; unknown terms never are. The same
//!   candidate may still be released restricted, and the release records the
//!   restriction.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{
    anchor_file, candidate_on, dataset_with_terms, fast_gate, gate_context, serve_release, Brain,
    ANCHOR, FACTS,
};
use common::Scratch;
use splinter_agent::CancelToken;
use splinter_core::terms::{Distribution, Permission, Terms, UsagePolicy};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::datasets::source_terms;
use splinter_pipelines::release::{anchor, release, ReleaseRequest, Released};
use splinter_pipelines::sources::{add_with_terms, SourceTarget};
use splinter_pipelines::train::Candidate;

fn policy(label: UsagePolicy) -> Terms {
    label.terms(label.as_str())
}

fn request(candidate: &Candidate, distribution: Distribution) -> ReleaseRequest {
    ReleaseRequest {
        gate: fast_gate(),
        distribution,
        ..ReleaseRequest::new(candidate.candidate.clone())
    }
}

fn decide(ctx: &Context, candidate: &Candidate, distribution: Distribution) -> Released {
    let decided = release(ctx, &request(candidate, distribution), &CancelToken::new()).unwrap();
    if let Some(id) = &decided.release {
        serve_release(ctx, id);
    }
    decided
}

fn refusal(ctx: &Context, candidate: &Candidate) -> String {
    match release(
        ctx,
        &request(candidate, Distribution::Unrestricted),
        &CancelToken::new(),
    ) {
        Err(OrchestratorError::Refused(why)) => why,
        other => panic!("an unrestricted release must be refused, got {other:?}"),
    }
}

fn context(test: &str) -> (Scratch, Context) {
    let (scratch, ctx) = gate_context(test, Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    (scratch, ctx)
}

fn add_source(
    scratch: &Scratch,
    ctx: &Context,
    text: &str,
    terms: Option<Terms>,
) -> splinter_core::source::SourceId {
    let path = scratch.0.join(format!("{}.txt", text.replace(' ', "-")));
    std::fs::write(&path, text).unwrap();
    let target = SourceTarget::from_learn_arg(&path.display().to_string()).unwrap();
    add_with_terms(ctx, &target, terms).unwrap().source.id
}

#[test]
fn a_dataset_states_the_strongest_restriction_of_its_sources() {
    let (scratch, ctx) = context("terms-sources");
    let open = add_source(
        &scratch,
        &ctx,
        "open text",
        Some(policy(UsagePolicy::Redistributable)),
    );
    let nc = add_source(
        &scratch,
        &ctx,
        "noncommercial text",
        Some(policy(UsagePolicy::Noncommercial)),
    );
    let unstated = add_source(&scratch, &ctx, "unstated text", None);

    assert_eq!(source_terms(&ctx, [&unstated]).unwrap(), None);
    let both = source_terms(&ctx, [&open, &nc]).unwrap().unwrap();
    assert_eq!(both.commercial_use, Permission::Forbidden);
    assert_eq!(both.redistribution, Permission::Allowed);
    let with_unstated = source_terms(&ctx, [&open, &unstated]).unwrap().unwrap();
    assert_eq!(
        with_unstated.training,
        Permission::Unknown,
        "one unread licence among stated ones fails closed"
    );
    // The same bytes under other terms are another source.
    let again = add_source(
        &scratch,
        &ctx,
        "open text",
        Some(policy(UsagePolicy::ResearchOnly)),
    );
    assert_ne!(again, open);
}

#[test]
fn the_strongest_restriction_binds_every_release_after_it() {
    let (_scratch, ctx) = context("terms-hops");
    let research = policy(UsagePolicy::ResearchOnly);
    let first_data = dataset_with_terms(&ctx, "alpha", FACTS, Some(research.clone()));
    let (first, _) = candidate_on(&ctx, first_data, &[ANCHOR, "alpha"]);
    assert_eq!(first.terms.as_ref(), Some(&research));

    // Refused as distributable before anything is measured; allowed as
    // restricted, and the release records the restriction.
    let why = refusal(&ctx, &first);
    assert!(why.contains("Redistribution is Forbidden"), "{why}");
    let decided = decide(&ctx, &first, Distribution::Restricted);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let first_release = ctx.releases().get(&decided.release.unwrap()).unwrap();
    assert_eq!(
        first_release.manifest.distribution,
        Distribution::Restricted
    );
    assert_eq!(
        first_release.manifest.terms.commercial_use,
        Permission::Forbidden
    );
    assert_eq!(
        first_release.manifest.training.terms.as_ref(),
        Some(&research)
    );

    // A second hop trained on freely redistributable data alone still
    // carries the restriction of the release it continues.
    let open = policy(UsagePolicy::Redistributable);
    let second_data = dataset_with_terms(&ctx, "beta", FACTS, Some(open));
    let (second, _) = candidate_on(&ctx, second_data, &[ANCHOR, "alpha", "beta"]);
    let carried = second.terms.as_ref().unwrap();
    assert_eq!(carried.commercial_use, Permission::Forbidden);
    assert_eq!(carried.redistribution, Permission::Forbidden);
    let why = refusal(&ctx, &second);
    assert!(why.contains("unrestricted"), "{why}");
    let decided = decide(&ctx, &second, Distribution::Restricted);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let second_release = ctx.releases().get(&decided.release.unwrap()).unwrap();
    assert_eq!(
        second_release.manifest.terms.commercial_use,
        Permission::Forbidden
    );
}

#[test]
fn unknown_terms_fail_closed_for_an_unrestricted_release() {
    let (_scratch, ctx) = context("terms-unknown");
    let data = dataset_with_terms(&ctx, "alpha", FACTS, None);
    let (unstated, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    assert_eq!(unstated.terms, None);
    let why = refusal(&ctx, &unstated);
    assert!(why.contains("Unknown"), "{why}");
    assert!(
        ctx.releases()
            .of_candidate(&unstated.candidate)
            .unwrap()
            .is_none(),
        "a refused release leaves nothing behind"
    );
    let decided = decide(&ctx, &unstated, Distribution::Restricted);
    let id = decided.release.expect("a restricted release is allowed");
    let manifest = ctx.releases().get(&id).unwrap().manifest;
    assert_eq!(manifest.terms.training, Permission::Unknown);
}

#[test]
fn terms_that_allow_everything_permit_an_unrestricted_release() {
    let (_scratch, ctx) = context("terms-open");
    let data = dataset_with_terms(
        &ctx,
        "alpha",
        FACTS,
        Some(policy(UsagePolicy::Redistributable)),
    );
    let (open, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    let decided = decide(&ctx, &open, Distribution::Unrestricted);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let manifest = ctx
        .releases()
        .get(&decided.release.unwrap())
        .unwrap()
        .manifest;
    assert_eq!(manifest.distribution, Distribution::Unrestricted);
}
