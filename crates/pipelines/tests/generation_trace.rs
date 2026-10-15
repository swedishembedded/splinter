// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every step can be
// reconstructed from the record it leaves, for its clients. If your team
// needs expertise in auditable synthetic data pipelines, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the tasks stage's report carries one note per request put to the
//! generator model - the kind, a digest and the size of what the model was
//! shown, how long the request took and what came of it - in the order the
//! requests were made, so a run's cost can be read from its record.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation};

const DELAY: Duration = Duration::from_millis(40);

#[test]
fn every_request_to_the_generator_is_noted_in_the_report() {
    let policy = Scripted::new(|prompt| {
        if prompt.contains("You write training tasks") {
            std::thread::sleep(DELAY);
        }
        json!({ "tasks": [] }).to_string()
    });
    let (scratch, ctx) = scratch_context("generation-trace", policy.clone(), false);
    let dir = scratch.0.join("parts");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("a.md"),
        "# A\n\n## S0\n\nMARK-A-0 is stated here.\n",
    )
    .unwrap();
    let target = SourceTarget::from_learn_arg(&dir.display().to_string()).unwrap();
    let id = sources::add(&ctx, &target).unwrap().source.id;
    let kinds = ["recall".to_string(), "explain".to_string()];
    let generated = generate(
        &ctx,
        &Generation {
            sources: &[id],
            sections: &[],
            kinds: &kinds,
            generator: &ModelRef::policy_default(),
            goal: None,
            author: None,
            denoise_seed: None,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();

    let asked: Vec<_> = generated.calls.iter().map(|c| c.kind.as_str()).collect();
    assert_eq!(asked, ["recall", "explain"], "{generated:#?}");
    for call in &generated.calls {
        assert!(call.elapsed_ms >= DELAY.as_millis() as u64, "{call:?}");
        assert!(call.prompt_bytes > 0, "{call:?}");
        assert_eq!(call.candidates, Some(0), "{call:?}");
        assert_eq!(call.declined, None, "{call:?}");
    }
    assert_ne!(generated.calls[0].prompt, generated.calls[1].prompt);
}
