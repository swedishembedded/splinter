// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in rolling fine-tuned models out to a shared inference
// service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: when the gate runs on a brain endpoint, its serve check starts no
//! `brain` of its own: it passes when the endpoint lists the candidate under
//! the digest of the candidate's adapter file, and fails, naming the digests,
//! when it does not.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::{config, Scratch};
use splinter_agent::CancelToken;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::BrainEndpoints;
use splinter_orchestrator::Context;

/// What the scripted endpoint serves, as brain's store names it.
const SERVED: &str = "Qwen/Qwen3-0.6B";

/// A context whose local models are requested from `server`.
fn context(scratch: &Scratch, server: &ScriptedEndpoint) -> Context {
    let mut settings = config(scratch);
    settings.brain = Some(BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY).unwrap());
    Context::new(settings, false).unwrap()
}

/// The serve check of a candidate whose adapter file is `adapter`, against a
/// context on `server`.
fn serve_check(
    ctx: &Context,
    adapter: &std::path::Path,
) -> splinter_eval::gate::Check<splinter_eval::gate::Serve> {
    use splinter_pipelines::release::probe::Suite;
    use splinter_pipelines::release::serve::{check, ServeCheck};
    let digest =
        splinter_core::digest::Digest::sha256_of_reader(std::fs::File::open(adapter).unwrap())
            .unwrap()
            .to_string();
    let sample = Suite::of_tasks("sample", Vec::new());
    check(
        ctx,
        &ServeCheck {
            binary: None,
            base: std::path::Path::new("/models/Qwen/Qwen3-0.6B"),
            adapter: Some(adapter),
            adapter_digest: &digest,
            system: None,
            sample: &sample,
            in_process: &[],
            startup: Duration::from_secs(1),
        },
        &CancelToken::new(),
    )
    .unwrap()
}

#[test]
fn the_serve_check_passes_when_the_endpoint_lists_the_candidate_under_its_digest() {
    let scratch = Scratch::new("endpoint-serve-listed");
    let adapter = scratch.0.join("adapter.brain.safetensors");
    std::fs::write(&adapter, b"candidate").unwrap();
    let digest =
        splinter_core::digest::Digest::sha256_of_reader(std::fs::File::open(&adapter).unwrap())
            .unwrap();
    let server = ScriptedEndpoint::start(vec![
        Card::base(SERVED),
        Card::adapter("Qwen/Qwen3-0.6B:splinter:candidate:v1", digest.as_str()),
    ]);
    let ctx = context(&scratch, &server);

    let check = serve_check(&ctx, &adapter);

    assert!(check.passed, "{:?}", check.reason);
    let measured = check.measured.unwrap();
    assert_eq!(measured.agreement, splinter_eval::gate::Agreement::Listed);
    assert!(
        measured.startup_line.contains("candidate:v1"),
        "{}",
        measured.startup_line
    );
}

#[test]
fn the_serve_check_fails_naming_the_digests_when_the_endpoint_does_not_list_the_candidate() {
    let scratch = Scratch::new("endpoint-serve-missing");
    let adapter = scratch.0.join("adapter.brain.safetensors");
    std::fs::write(&adapter, b"candidate").unwrap();
    let server = ScriptedEndpoint::start(vec![Card::base(SERVED)]);
    let ctx = context(&scratch, &server);

    let check = serve_check(&ctx, &adapter);

    assert!(!check.passed);
    assert!(check.reason.unwrap().contains("sha256:"));
}
