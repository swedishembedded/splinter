// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that tell you what is
// wrong before an expensive run starts, for its clients. If your team needs
// expertise in making long-running ML jobs fail early and clearly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a `learn` that cannot write its state refuses before it starts, and
//! says what to change, instead of failing after the first stage's work.

#![allow(clippy::unwrap_used)]

mod common;

use common::gate::FakeTrainer;
use common::manual::{manual_policy as policy, MANUAL};
use common::scratch_context;
use splinter_orchestrator::OrchestratorError;
use splinter_pipelines::learn::{learn, LearnRequest};

#[cfg(unix)]
#[test]
fn a_run_on_an_unwritable_state_root_is_refused_before_it_starts() {
    use std::os::unix::fs::PermissionsExt;
    let (scratch, ctx) = scratch_context("learn-preflight", policy(), false);
    let manual = std::env::temp_dir().join(format!("preflight-manual-{}.md", std::process::id()));
    std::fs::write(&manual, MANUAL).unwrap();
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o555)).unwrap();
    // Root ignores the mode; the refusal cannot be provoked there.
    if std::fs::write(scratch.0.join("probe"), b"x").is_ok() {
        std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }

    let outcome = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            kinds: vec!["recall".into()],
            ..LearnRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    );
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::remove_file(&manual).ok();

    let Err(OrchestratorError::Refused(why)) = outcome else {
        panic!("the run must be refused, not attempted: {outcome:?}");
    };
    assert!(
        why.contains("state-writable") && why.contains("--state"),
        "the refusal names the check and the fix: {why}"
    );
    assert!(
        !ctx.root().path().exists(),
        "nothing was written under the state root"
    );
}
