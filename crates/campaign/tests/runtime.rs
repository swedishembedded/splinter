// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a runtime outlives its commands, and each command resolves what
//! `policy:<alias>` means for itself.
//!
//! A REPL runs many commands in one process. The release an alias points at
//! is pinned for the length of one command, so a command is never half on
//! one release and half on another; the next command must see the alias
//! where it now is, not where the process first found it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{anchor_file, gate_context, released, Brain, ANCHOR};
use splinter_campaign::release::{anchor, arm};

#[test]
fn a_new_command_sees_the_alias_where_the_last_one_left_it() {
    let (scratch, ctx) = gate_context("runtime-pins", Brain::Honest);
    anchor::freeze(&ctx, &anchor_file(&scratch.0, 4)).unwrap();
    let runtime = ctx.runtime().clone();

    let first = released(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let during = runtime.context();
    assert_eq!(
        during.policy_pin("default").unwrap().unwrap().release,
        first
    );

    // Another command moves the alias while `during` is still running.
    let second = released(&ctx, "beta", &[ANCHOR, "alpha", "beta"]);
    assert_ne!(first, second);

    assert_eq!(
        during.policy_pin("default").unwrap().unwrap().release,
        first,
        "a command keeps the release it first resolved"
    );
    assert_eq!(
        runtime
            .context()
            .policy_pin("default")
            .unwrap()
            .unwrap()
            .release,
        second,
        "the next command resolves the alias afresh"
    );
}

#[test]
fn a_model_handed_to_the_runtime_answers_every_command() {
    let (_scratch, ctx) = gate_context("runtime-handed", Brain::Honest);
    let runtime = ctx.runtime().clone();
    // The base arm the gate context handed in, with no adapter.
    let base = arm(ctx.config(), None);

    let identity = |context: &splinter_campaign::Context| context.model(&base).unwrap().identity;
    let before = identity(&runtime.context());
    assert_eq!(identity(&runtime.context()), before);
}
