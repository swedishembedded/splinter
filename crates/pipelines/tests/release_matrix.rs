// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and what it loses, for its clients. If your team needs
// expertise in continual-learning evaluation for fine-tuned models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the results matrix of an alias is read from the gate reports of its
//! releases. With nothing released there is no matrix (refused, not an empty
//! one); with a release, its own cell holds what its gate measured on its own
//! suite and nothing else is filled in.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{anchor_file, candidate, fast_gate, gate_context, serve_release, Brain, ANCHOR};
use splinter_agent::CancelToken;
use splinter_pipelines::release::continual::matrix;
use splinter_pipelines::release::{anchor, release, ReleaseRequest};

#[test]
fn an_alias_with_no_release_has_no_matrix() {
    let (_scratch, ctx) = gate_context("matrix-none", Brain::Honest);
    let refused = matrix(&ctx, "default").unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
}

#[test]
fn a_released_candidate_fills_only_the_cell_its_gate_measured() {
    let (scratch, ctx) = gate_context("matrix-one", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let request = ReleaseRequest {
        gate: fast_gate(),
        ..ReleaseRequest::new(candidate.candidate.clone())
    };
    let decided = release(&ctx, &request, &CancelToken::new()).unwrap();
    let id = decided.release.clone().expect("released");
    serve_release(&ctx, &id);

    let report = matrix(&ctx, "default").unwrap();
    assert_eq!(report.releases, [id]);
    assert_eq!(report.matrix.generations(), 1);
    let own = decided
        .gate
        .improvement
        .measured
        .as_ref()
        .and_then(|m| m.comparison.candidate_accuracy);
    assert!(own.is_some());
    assert_eq!(report.matrix.cell(0, 0), own);
}
