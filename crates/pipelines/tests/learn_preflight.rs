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
    // Root ignores the mode, and then nothing here can be provoked: the test
    // fails rather than pass without having checked anything.
    let ignored = std::fs::write(scratch.0.join("probe"), b"x").is_ok();
    if ignored {
        std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert!(
        !ignored,
        "the file mode is ignored for this user (root?): run this test as an ordinary user"
    );

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

#[test]
fn one_source_text_is_refused_before_any_rollout_when_the_trainer_divides_by_group() {
    let (_scratch, ctx) = scratch_context("learn-one-group", policy(), false);
    let manual = std::env::temp_dir().join(format!("one-group-manual-{}.md", std::process::id()));
    std::fs::write(&manual, MANUAL).unwrap();
    let trainer = FakeTrainer::knowing(&[]).holding_out_by_group();
    let outcome = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            kinds: vec!["recall".into()],
            ..LearnRequest::default()
        },
        &trainer,
    );
    std::fs::remove_file(&manual).ok();
    let Err(OrchestratorError::Refused(why)) = outcome else {
        panic!("one text cannot be divided, so the run is refused: {outcome:?}");
    };
    assert!(
        why.contains("1 independent source text") && why.contains("add sources"),
        "the refusal counts the groups and names the fix: {why}"
    );
    assert!(
        trainer.plans.lock().unwrap().is_empty(),
        "nothing was trained"
    );
}
