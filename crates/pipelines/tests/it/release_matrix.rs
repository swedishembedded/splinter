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

use crate::common;

use common::gate::{gate_context, Brain};
use splinter_pipelines::release::continual::matrix;

#[test]
fn an_alias_with_no_release_has_no_matrix() {
    let (_scratch, ctx) = gate_context("matrix-none", Brain::Honest);
    let refused = matrix(&ctx, "default").unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
}
