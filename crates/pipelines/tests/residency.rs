// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the device goes to the work that needs it.
//!
//! * The policy and the release gate's arms share one resident base.
//! * Training and the gate's serve check start with no resident base held:
//!   a fine-tune loads its own copy and `brain serve` is another process,
//!   so every base the context keeps resident is released first, and the
//!   next use of a model loads its base again.
//!
//! Bases are scripted (no weights): the loader logs each load and free,
//! and a file stands for the device being held while any base is
//! resident - the `brain` stand-in refuses to start while it exists, as
//! brain on a device another process fills would.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::gate::{candidate, decide, device_lock, gate_context, policy, Brain};
use splinter_core::model_ref::ModelRef;
use splinter_model::residency::scripted::{ScriptedEvent, ScriptedLoader};
use splinter_model::Residency;
use splinter_orchestrator::Context;
use splinter_pipelines::release::arm;

/// A scripted loader that holds `lock` while any base it loaded lives.
fn holding(lock: PathBuf) -> ScriptedLoader {
    ScriptedLoader::new(|_: &str, _: Option<&Path>| "I do not know.".into()).observed(
        move |event| match event {
            ScriptedEvent::Loaded(_) => std::fs::write(&lock, b"held").unwrap(),
            ScriptedEvent::Freed(_) => std::fs::remove_file(&lock).unwrap(),
            _ => {}
        },
    )
}

fn loads(loader: &ScriptedLoader) -> usize {
    loader
        .events()
        .iter()
        .filter(|e| matches!(e, ScriptedEvent::Loaded(_)))
        .count()
}

/// A model on the policy base with an adapter no fixture hands in, so the
/// context loads it on the residency.
fn on_the_base(ctx: &Context, adapter: &str) -> ModelRef {
    arm(ctx.config(), Some(&ctx.root().path().join(adapter)))
}

#[test]
fn the_policy_and_models_on_its_base_share_one_resident_base() {
    let (scratch, ctx) = gate_context("residency-share", Brain::Missing);
    let loader = holding(device_lock(&scratch));
    let ctx = ctx.with_residency(Residency::new(Arc::new(loader.clone())));
    ctx.model(&policy()).unwrap();
    ctx.model(&on_the_base(&ctx, "one.safetensors")).unwrap();
    ctx.model(&on_the_base(&ctx, "two.safetensors")).unwrap();
    assert_eq!(loads(&loader), 1, "{:?}", loader.events());
    assert_eq!(ctx.resident_bases(), 1);
}

#[test]
fn training_and_the_serve_check_start_with_no_resident_base() {
    let (scratch, ctx) = gate_context("residency-phases", Brain::Honest);
    let loader = holding(device_lock(&scratch));
    let ctx = ctx.with_residency(Residency::new(Arc::new(loader.clone())));
    ctx.model(&policy()).unwrap();
    assert_eq!(ctx.resident_bases(), 1);

    let (candidate, trainer) = candidate(&ctx, "alpha", &["alpha"]);
    assert_eq!(*trainer.resident_at_start.lock().unwrap(), vec![0]);

    // The policy's base is resident again before the gate, as it is when
    // a run solves between training and releasing.
    ctx.model(&on_the_base(&ctx, "probe.safetensors")).unwrap();
    assert_eq!(ctx.resident_bases(), 1);
    assert_eq!(loads(&loader), 2, "the next use reloads the released base");

    let decided = decide(&ctx, &candidate);
    let serve = &decided.gate.serve;
    assert!(
        serve.passed,
        "brain serve must find the device free: {serve:#?}"
    );
}
